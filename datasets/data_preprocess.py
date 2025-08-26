import os
import copy

import numpy
import torch
import numpy as np
import cdflib
import json
import h5py
import sys
import pickle
import joblib
import hashlib
import argparse
import zipfile
import time
import pickle as pkl
import pandas as pd
from shutil import rmtree
from PIL import Image
from glob import glob
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root)

from common.smpl_wrapper import SMPL
from common.geometry import *
import common.densepose_methods as dp_utils

# output_dir = os.path.join(prj_dir, 'data')
h36m_path = os.path.join(prj_root, r'./data/h36m')
pw3d_path = os.path.join(prj_root, r'./data/3dpw')
surreal_path = os.path.join(prj_root, r'./data/SURREAL_v1')
up3d_path = os.path.join(prj_root, r'./data/up-3d')
bedlam_path = os.path.join(prj_root, r'./data/bedlam/')

coco_path = os.path.join(prj_root, r'./data/coco')
hp3d_path = os.path.join(prj_root, r'./data/3dhp')

ssp_path = os.path.join(prj_root, r'./data/ssp_3d')


def kp2bbox(kps, img_wh, scale=1.0):
    """ use skeleton to estimate bbox
    :param kps:  bs,n,2
    :param img_wh: bs,2
    :param scale:
    :return: bs,[u_min,v_min,w,h]
    """
    left_up = kps.min(dim=1)[0]
    left_up = torch.max(left_up, torch.zeros_like(left_up))

    right_down = kps.max(dim=1)[0]
    right_down = torch.min(right_down, torch.tensor(img_wh, device=kps.device, dtype=kps.dtype))

    bbox_wh = (right_down - left_up) * scale
    bbox_mid = (left_up + right_down) / 2
    bbox = torch.cat([bbox_mid - (bbox_wh / 2), bbox_wh], dim=1)
    return bbox


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def camera2uv(xyz_cam, cx, cy, fx, fy):
    """ convert camera coordinate to pixel coordinate
    :param xyz_cam: 3d position in camera, with shape(...,3)
    :param cx: float u0 principle point
    :param cy: float v0 principle point
    :param fx: float αx
    :param fy: float αy
    :return:
    """
    u = fx * (xyz_cam[..., 0] / xyz_cam[..., 2]) + cx
    v = fy * (xyz_cam[..., 1] / xyz_cam[..., 2]) + cy
    return torch.stack([u, v], dim=-1)


def smpl2segmentation(triangles, bbox, device):
    """  generate dp_masks in coco_densepose style
    :param triangles: (bs,n_tri, 3, 2) projected 2D triangles of SMPL.mesh.faces
    :param bbox: (bs,4)
    :param device:
    :return: RLE encoded string. {'counts': res[i], 'size': [256,256]}
    """
    # return '00'
    bs = triangles.shape[0]
    res = []
    if torch.is_tensor(bbox): bbox = bbox.cpu().numpy()
    for frame_idx in range(bs):
        x, y, w, h = bbox[frame_idx]
        mask = np.zeros((256, 256), dtype=bool)
        for u_idx in range(256):
            # for v_idx in range(256):
            #     u, v = x + (u_idx + 0.5) / 256 * w, y + (v_idx + 0.5) / 256 * h
            #     tri_num = point_in_triangles(
            #         triangles[None], torch.tensor([u, v], dtype=triangles.dtype, device=device)[None, None]
            #     ).sum().item()
            #     masks[frame_idx, v_idx, u_idx] = (tri_num != 0)

            # Parallel processing of row data
            v_idxes = list(range(256))
            u, v_s = x + (u_idx + 0.5) / 256 * w, [y + (v_idx + 0.5) / 256 * h for v_idx in v_idxes]
            tri_num = point_in_triangles(
                triangles[frame_idx][None].to(device),
                torch.tensor([[u, v] for v in v_s], dtype=triangles.dtype, device=device)[None]
            )[0].sum(dim=0)
            mask[:, u_idx] = (tri_num != 0).tolist()
        RLE_counts = mask_utils.encode(mask.copy(order='F'))['counts']
        res.append(RLE_counts)
    return res


def mesh2visibility(vertices_uvd, faces, img_wh, device):
    # vertices visibility calculation
    vis = []
    for frame_idx in range(vertices_uvd.shape[0]):
        triangles_uvd = vertices_uvd[frame_idx:frame_idx + 1, faces]  # 1, 13776, 3, 3
        triangles = triangles_uvd[:, :, :, :2]  # 1, n_tri,3,2
        triangles_d_min = triangles_uvd[:, :, :, -1].min(dim=-1, keepdim=True)[0]  # bs, n_tri, 1
        occlusion_in = point_in_triangles(  # 2d point is in 2d triangle
            triangles.to(device), vertices_uvd[frame_idx:frame_idx + 1, :, :2].to(device))
        occlusion_dis = (triangles_d_min.to(device) < vertices_uvd[frame_idx:frame_idx + 1, :, -1].to(
            device))  # 3d triangle is close to camera than 3d point
        occlusion_triangles_num = (occlusion_in * occlusion_dis).sum(dim=1)
        not_self_occ = (occlusion_triangles_num == 0).to('cpu').numpy()
        occlusion_in, occlusion_dis, occlusion_triangles_num = None, None, None

        uv = vertices_uvd[frame_idx:frame_idx + 1, :, :2].cpu().numpy()
        in_frame = np.all((uv > 0) * (uv < img_wh[frame_idx][None, None]), axis=-1)

        vis.append(in_frame * not_self_occ)
        # print(f'visible vertices num: {vis[-1].sum().item()}')
    return vis


def prepare_3dpw(split='train', device='cuda:0', dtype=torch.float64, dataset_path=pw3d_path, demo=False):
    out_file = os.path.join(dataset_path, f'densepose_{split}.pt' if not demo else f'densepose_{split}_demo.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return

    print(f'\n\nPrepare 3DPW {split}set {dataset_path}')

    ann_files = glob(os.path.join(dataset_path, f'sequenceFiles/{split}/*'))
    ann_files.sort()  # to assure seq_idxes are stable
    if demo: ann_files = ann_files[:2]
    output = {}
    for seq_idx, pkl_file in enumerate(ann_files):
        output[seq_idx] = {}
        seq = pickle.load(open(pkl_file, 'rb'), encoding='latin-1')
        for actor_idx in range(len(seq['betas'])):
            img_seq_name = f"{seq['sequence']}_{actor_idx}"
            seq_len = seq['campose_valid'][actor_idx].shape[0]

            beta = seq['betas'][actor_idx][:10]  # shape=(10,) #smplx supports a maximum of 10
            beta = beta[None].repeat(seq_len, 0)  # shape=(1257,10)
            gender = 'male' if seq['genders'][actor_idx] == 'm' else 'female'
            theta_world = seq['poses'][actor_idx]  # shape=(1257,72)
            transl_world = seq['trans'][actor_idx]  # meter bs,3

            cam_extrinsic = torch.tensor(seq['cam_poses'], device='cpu', dtype=dtype)
            cam_quat = mat2quat(cam_extrinsic[:, :3, :3])
            cam_t = cam_extrinsic[:, :3, -1]

            smpl = SMPL(gender=gender, batch_size=seq_len, device='cpu', dtype=dtype, hybrik_joints=True)
            theta, beta, transl = smpl.pose_world2camera(cam_quat, cam_t, beta, theta_world, transl_world)
            joints, vertices, joints17 = smpl(beta=beta, pose=theta, transl=transl)
            # ortho_twist = smpl.cal_ortho_twist(beta=beta, pose=theta)  # bs,29,3

            cam_intrinsics = torch.tensor(seq['cam_intrinsics'], dtype=dtype, device='cpu')  # 3,3

            uvd = cam_intrinsics.reshape(1, 1, 3, 3).matmul(vertices.unsqueeze(-1)).squeeze(-1)
            uvd[:, :, :2] /= uvd[:, :, 2:]  # bs, 6890, 3

            # for idx in range(0,seq_len,100):
            #     from common.visualize import draw_joints2D
            #     img_path = 'imageFiles/' + f"{seq['sequence']}"
            #     draw_joints2D(uvd[idx, :, :2].cpu().numpy(), f'{dataset_path}/{img_path}/image_{idx:05d}.jpg', color='blue',
            #                   suffix='_smpl_2d')
            #     draw_joints2D(seq['poses2d'][actor_idx][idx].T[:, :2], f'{dataset_path}/{img_path}/image_{idx:05d}.jpg',
            #                   color='green', suffix='_coco_2d')

            image_id = [int(f'7{seq_idx:03d}{actor_idx:01d}{frame_idx:05d}') for frame_idx in range(seq_len)]
            image_path = [f"imageFiles/{seq['sequence']}/image_{frame_idx:05d}.jpg" for frame_idx in range(seq_len)]
            img_wh = np.array([Image.open(os.path.join(dataset_path, each)).size for each in image_path])
            # kp_detected_coco = torch.tensor(seq['poses2d'][actor_idx]).permute([0, 2, 1])
            valid_frame = torch.tensor(seq['campose_valid'][actor_idx], dtype=torch.bool)
            # print(seq_len - valid_frame.sum().item(), 'invalid frames in', img_seq_name)

            bbox = kp2bbox(uvd[:, :, :2], img_wh)

            output[seq_idx][actor_idx] = {
                'valid_frame': valid_frame.numpy(),  # torch.bool
                # '3dpw_coco_kp_detected': kp_detected_coco.numpy(),  # bs,18,3 coco kp

                'id': np.array(image_id),  # int64 # [7000000000, 7000000001,
                'video_name': np.array([img_seq_name] * seq_len),
                'frame_idx': np.array([frame_idx for frame_idx in range(seq_len)]),
                'file_name': np.array(image_path),  # not tensor ['imageFiles/courtyard_arguing_00/image_00000.jpg',
                'image_wh': img_wh,  # [(1080, 1920), (1080, 1920),
                'bbox': bbox.cpu().numpy(),
                'cam_intrinsics': cam_intrinsics.cpu().reshape(1, 3, 3).repeat(seq_len, 1, 1).numpy(),

                'gender': np.array([gender] * seq_len),  # not tensor
                'smpl_theta': theta.cpu().numpy().reshape(-1, 24, 3),
                'smpl_beta': beta.cpu().numpy(),
                'smpl_transl': transl.cpu().numpy(),
                # 'smpl_ortho_twist': ortho_twist.cpu().numpy(),
                # 'smpl_j29': joints.cpu().numpy(),  # meter as unit
                # 'smpl_j17': joints17.cpu().numpy(),  # meter as unit
                'smpl_vertices': vertices.cpu().numpy(),
            }
    for seq_idx in range(len(ann_files)):
        print(f'start vertices visibility calculation for seq{seq_idx}/{len(ann_files)}')
        actor_num = len(output[seq_idx].keys())
        bbox = [output[seq_idx][actor_idx]['bbox'] for actor_idx in range(actor_num)]
        vertices = [output[seq_idx][actor_idx]['smpl_vertices'] for actor_idx in range(actor_num)]
        vertices = [torch.tensor(each, dtype=dtype, device='cpu') for each in vertices]
        cam_intrinsics = torch.tensor(output[seq_idx][0]['cam_intrinsics'], dtype=dtype, device='cpu')
        uvd = [cam_intrinsics.unsqueeze(1).matmul(each.unsqueeze(-1)).squeeze(-1) for each in vertices]
        for each in uvd: each[:, :, :2] /= each[:, :, 2:]
        triangles_uvd = torch.cat([each[:, smpl.faces.astype(np.int32)] for each in uvd], dim=1)
        triangles = triangles_uvd[:, :, :, :2]  # bs, n_tri,3,2
        triangles_d_min = triangles_uvd[:, :, :, -1].min(dim=-1, keepdim=True)[0]  # bs, n_tri, 1
        # uvd = torch.cat(uvd, dim=1)

        # vis = [torch.rand(actor_num, 1, 6890) > 0.2 for i in range(triangles.shape[0])]
        vis = [[] for each in range(actor_num)]
        smpl_mask = [[] for each in range(actor_num)]
        for frame_idx in range(triangles.shape[0]):
            for actor_idx in range(actor_num):
                occlusion_in = point_in_triangles(  # 2d point is in 2d triangle
                    triangles[frame_idx:frame_idx + 1].to(device).float(),  # [1, 27552, 3, 2]
                    uvd[actor_idx][frame_idx:frame_idx + 1, :, :2].to(device).float()  # [1, 6890, 2]
                )
                occlusion_dis = (
                        triangles_d_min[frame_idx:frame_idx + 1].to(device) < uvd[actor_idx][frame_idx:frame_idx + 1, :,
                                                                              -1].to(
                    device))  # 3d triangle is close to camera than 3d point
                occlusion_triangles_num = (occlusion_in * occlusion_dis).sum(dim=1)  # 1,6890
                not_self_occ = (occlusion_triangles_num == 0).to('cpu').numpy()  ## 1,6890

                uv = uvd[actor_idx][frame_idx:frame_idx + 1, :, :2].cpu().numpy()  # 1,6890,2
                wh = output[seq_idx][0]['image_wh'][frame_idx][None, None]  # 1,1,2
                in_frame = np.all((uv > 0) * (uv < wh), axis=-1)
                vis_each = in_frame * not_self_occ
                vis[actor_idx].append(vis_each)
                occlusion_in, occlusion_dis, occlusion_triangles_num = None, None, None
                # print(f'start vertices visibility calculation for {seq_idx} {vis[-1].sum().item() / actor_num}')

                triangles_each = uvd[actor_idx][frame_idx:frame_idx + 1, smpl.faces.astype(np.int32),
                                 :2]  # 1, 13776, 3, 2
                visible_triangles_each = np.any(vis_each[0][smpl.faces.astype(np.int32)], axis=-1)
                smpl_mask[actor_idx].append(smpl2segmentation(
                    triangles_each[:, visible_triangles_each],  # 1,5980,3,2
                    bbox[actor_idx][frame_idx:frame_idx + 1], device)[0])
        for actor_idx in range(actor_num):
            output[seq_idx][actor_idx]['vis'] = np.concatenate(vis[actor_idx])  # seq_len,6890
            output[seq_idx][actor_idx]['smpl_mask'] = np.array(smpl_mask[actor_idx])  # seq_len,

    data = {}
    for seq_idx in range(len(ann_files)):
        actor_num = len(output[seq_idx].keys())
        for actor_idx in range(actor_num):
            for attr in output[seq_idx][actor_idx].keys():
                if attr not in data.keys():
                    data[attr] = []
                data[attr].append(output[seq_idx][actor_idx][attr])
    for attr in data.keys():
        data[attr] = np.concatenate(data[attr], axis=0)
    # filter invalid ones
    valid_frame = data.pop('valid_frame')
    data.pop('smpl_vertices')

    for attr in data.keys():
        data[attr] = data[attr][valid_frame]

    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')


def prepare_h36m(interval=20, split='train', device='cuda:0', dtype=torch.float64, dataset_path=h36m_path):
    # TODO segmentation 文件可以用来调整transl 以及平移mask
    assert interval % 5 == 0
    out_file = os.path.join(dataset_path, f'densepose_{split}_{interval}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return

    print(f'\n\nPrepare Human3.6M {split}set interval:{interval} {dataset_path}')
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8'] if split == 'train' else ['S9', 'S11']

    smpl = SMPL(gender='neutral', batch_size=2048, device=device, dtype=dtype, hybrik_joints=True)

    output = {}
    smpl_param_mosh = {}
    for subject in subjects:
        subject_idx = int(subject[1:])
        print('start to prepare ', subject)
        # smpl param of about one fifth frames estimated by mosh
        smpl_param_file = os.path.join(dataset_path, 'annotations', 'smpl_param_mosh_hybrik',
                                       f'Human36M_subject{subject_idx}_smpl.json')
        smpl_param_all = json.load(open(smpl_param_file, 'r'))
        cam_param_file = os.path.join(dataset_path, f'annotations/Human36M_subject{subject_idx}_camera.json')
        cam_param_all = json.load(open(cam_param_file, 'r'))
        j3d_file = os.path.join(dataset_path, f'annotations/Human36M_subject{subject_idx}_joint_3d.json')
        j3d_world_all = json.load(open(j3d_file, 'r'))
        ann_file = os.path.join(dataset_path, f'annotations/Human36M_subject{subject_idx}_data.json')
        ann_all = json.load(open(ann_file, 'r'))
        # ann_idx2img_idx = np.array([ann_all['annotations'][i]['image_id'] for i in range(len(ann_all['annotations']))])

        for each_item in ann_all['images']:
            s_i, a_i, sa_i, c_i, f_i = each_item['subject'], each_item['action_idx'], \
                each_item['subaction_idx'], each_item['cam_idx'], each_item['frame_idx']
            seq_name = f's_{s_i:02d}_act_{a_i:02d}_subact_{sa_i:02d}_ca_{c_i:02d}'

            if f_i % interval != 0: continue
            if s_i == 11 and a_i == 2: continue  # Discard corrupted video 'S11' 'Directions'  following MixSTE
            if [s_i, a_i, sa_i] in [[9, 5, 2], [9, 10, 2], [9, 13, 1], [11, 6, 2]]: continue  # following HybrIK

            if seq_name not in output.keys(): output[seq_name] = []

            j17_world = torch.tensor(j3d_world_all[str(a_i)][str(sa_i)][str(f_i)],
                                     dtype=dtype, device=device) / 1000  # 17,3
            R = torch.tensor(cam_param_all[str(c_i)]['R'], dtype=dtype, device=device)
            t = torch.tensor(cam_param_all[str(c_i)]['t'], dtype=dtype, device=device) / 1000
            fx, fy = torch.tensor(cam_param_all[str(c_i)]['f'], dtype=dtype, device=device)
            cx, cy = torch.tensor(cam_param_all[str(c_i)]['c'], dtype=dtype, device=device)
            cam_intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device)
            j17_cam = R.matmul(j17_world[:, :, None])[:, :, 0] + t[None]
            # draw_joints2D(camera2uv(j17_cam,cx,cy,fx,fy),os.path.join(r'D:\dataset\h36m\images', each_item['file_name']))
            # ann_idx = np.where(ann_idx2img_idx == each_item['id'])[0][0]

            output[seq_name].append({
                # 'keypoints_vis': np.array(ann_all['annotations'][ann_idx]['keypoints_vis'])[None],  # fake 99%+ visible
                'h36m_gt_j17': j17_cam.cpu()[None].numpy(),  # 1,17,3

                'id': np.array([int(f'3{s_i:02d}{a_i:02d}{sa_i}{c_i}{f_i:05d}')]),  # 301021100000
                'video_name': np.array([seq_name]),
                'frame_idx': np.array([f_i]),
                'file_name': np.array([each_item['file_name']]),  # 'seq_name/seq_name_000001.jpg'
                'image_wh': np.array((each_item['width'], each_item['height']))[None],
                # 'bbox': np.array(ann_all['annotations'][ann_idx]['bbox'])[None],  # xywh
                'cam_intrinsics': cam_intrinsics.cpu()[None].numpy(),  # 1,3,3
            })

        for each_item in smpl_param_all['images']:
            s_i, a_i, sa_i, c_i, f_i = each_item['subject'], each_item['action_idx'], \
                each_item['subaction_idx'], each_item['cam_idx'], each_item['frame_idx']
            seq_name = f's_{s_i:02d}_act_{a_i:02d}_subact_{sa_i:02d}_ca_{c_i:02d}'

            if f_i % interval != 0: continue
            if s_i == 11 and a_i == 2: continue  # Discard corrupted video 'S11' 'Directions'  following MixSTE
            if [s_i, a_i, sa_i] in [[9, 5, 2], [9, 10, 2], [9, 13, 1], [11, 6, 2]]: continue  # following HybrIK

            beta = torch.tensor(each_item['betas'], dtype=dtype, device=device).reshape(1, 10)
            theta = torch.tensor(each_item['thetas'], dtype=dtype, device=device).reshape(1, 24, 3)

            if seq_name not in smpl_param_mosh.keys(): smpl_param_mosh[seq_name] = []

            smpl_param_mosh[seq_name].append({
                'mosh_frame_idx': torch.tensor([f_i]).numpy(),
                'smpl_theta': theta.cpu().numpy(),  # 1,24,3
                'smpl_beta': beta.cpu().numpy(),  # 1,10
            })
    for video_name in output.keys():
        all_attr_seq = {}
        for attr in output[video_name][0].keys():
            all_attr_seq[attr] = np.concatenate([each[attr] for each in output[video_name]], axis=0)
        output[video_name] = all_attr_seq
    for video_name in smpl_param_mosh.keys():
        print(f'prepare smpl param and vertices visibility for {video_name}')
        all_attr_seq = {}
        for attr in smpl_param_mosh[video_name][0].keys():
            all_attr_seq[attr] = np.concatenate([each[attr] for each in smpl_param_mosh[video_name]], axis=0)
        # check smpl-data index
        mosh_idx2data_idx = np.array(
            [np.where(output[video_name]['frame_idx'] == each)[0][0] for each in all_attr_seq['mosh_frame_idx']])
        assert np.all(mosh_idx2data_idx == range(len(mosh_idx2data_idx)))
        assert np.all(all_attr_seq['mosh_frame_idx'] == output[video_name]['frame_idx'][mosh_idx2data_idx])
        assert (np.all(np.array(
            [len(np.where(output[video_name]['frame_idx'] == each)[0]) == 1 for each in
             all_attr_seq['mosh_frame_idx']]))), "h36m smpl ann match error"
        frame_num_without_smpl = output[video_name]['frame_idx'].shape[0] - all_attr_seq['mosh_frame_idx'].shape[0]
        if frame_num_without_smpl > 0:
            print(f'{video_name} {frame_num_without_smpl} frames do not have smpl parameter(mosh)')

        all_attr_seq.pop('mosh_frame_idx')
        for key in all_attr_seq.keys():
            all_attr_seq[key] = all_attr_seq[key][mosh_idx2data_idx]
        # prepare smpl related data
        mosh_smpl_j29, mosh_smpl_vertices, mosh_smpl_j17 = smpl(
            beta=all_attr_seq['smpl_beta'], pose=all_attr_seq['smpl_theta'])
        # mosh_smpl_ortho_twist = smpl.cal_ortho_twist(
        #     beta=all_attr_seq['smpl_beta'], pose=all_attr_seq['smpl_theta'])
        j17 = torch.tensor(output[video_name]['h36m_gt_j17'], device=device, dtype=dtype)
        mosh_smpl_transl = j17.mean(dim=1) - mosh_smpl_j17.mean(dim=1)
        mosh_smpl_j29_ = (mosh_smpl_j29 + mosh_smpl_transl[:, None])  # 118,29,3
        mosh_smpl_vertices_ = (mosh_smpl_vertices + mosh_smpl_transl[:, None])
        cam_intrinsics = torch.tensor(output[video_name]['cam_intrinsics'], dtype=dtype, device=device)  # 118,3,3

        vertices_uvd = cam_intrinsics[:, None].matmul(mosh_smpl_vertices_[:, :, :, None])[:, :, :, 0]
        vertices_uvd[:, :, :2] /= vertices_uvd[:, :, 2:]  # 118,6890,3
        bbox = kp2bbox(vertices_uvd[:, :, :2], output[video_name]['image_wh'])  # [u_min,v_min,w,h]

        vertices_visibility = mesh2visibility(
            vertices_uvd, smpl.faces.astype(np.int32), output[video_name]['image_wh'], device)

        triangles = vertices_uvd[:, smpl.faces.astype(np.int32), :2]  # bs, 13776, 3, 2
        smpl_mask = smpl2segmentation(triangles, bbox, device)

        all_attr_seq.update({
            'gender': np.array(['neutral'] * mosh_idx2data_idx.shape[0]),
            'smpl_transl': mosh_smpl_transl.cpu().numpy(),  # bs,3
            # 'smpl_j29': mosh_smpl_j29_.cpu().numpy(),  # bs,29,3 # meter as unit
            # 'smpl_j17': (mosh_smpl_j17 + mosh_smpl_transl[:, None]).cpu().numpy(),  # bs,17,3  meter as unit
            # 'smpl_ortho_twist': mosh_smpl_ortho_twist.cpu().numpy(),  # bs,29,3
            # 'smpl_vertices': mosh_smpl_vertices_.cpu().numpy(),  # bs,6890,3 # meter as unit

            'bbox': bbox.cpu().numpy(),  # bs,4
            'vis': np.concatenate(vertices_visibility),
            'smpl_mask': smpl_mask,
        })
        output[video_name].update(all_attr_seq)

    # convert seq_name->data dict to list
    data = {}
    for video_name in output.keys():
        for attr in output[video_name].keys():
            if attr not in data.keys():
                data[attr] = []
            data[attr].append(output[video_name][attr])
    for attr in data.keys():
        data[attr] = np.concatenate(data[attr], axis=0)
    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')


def prepare_surreal(split='train', device='cuda:0', dtype=torch.float64, dataset_path=surreal_path):
    out_file = os.path.join(dataset_path, f'densepose_{split}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare surreal {split}set {dataset_path}')

    pkl_file = os.path.join(dataset_path, rf'annotations/SURREAL_{split}.pkl')
    # https://github.com/ShirleyMaxx/VirtualMarker/issues/3
    pkl_data = pkl.load(open(pkl_file, 'rb'))  # list
    # print(torch.all(torch.tensor([each['root_cam'] is None for each in pkl_data]))==True)

    file_name = [each['img_name'] for each in pkl_data]
    frame_idx = [f"{int(each['img_name'][-7:-4]) + 1:03d}" for each in pkl_data]
    for i in range(len(file_name)):
        # the frame index starts from 1
        file_name[i] = file_name[i][:-7] + frame_idx[i] + '.jpg'
    image_wh = np.array([each['img_hw'][::-1] for each in pkl_data])

    gender = np.array([each['smpl_param']['gender'] for each in pkl_data])
    smpl_theta = torch.tensor(
        [each['smpl_param']['pose'] for each in pkl_data], dtype=dtype, device=device).reshape(-1, 24, 3)
    smpl_beta = torch.tensor([each['smpl_param']['shape'] for each in pkl_data], dtype=dtype, device=device)
    smpl_transl = torch.tensor([each['smpl_param']['trans'] for each in pkl_data], dtype=dtype, device=device)

    cam_intrinsics = torch.tensor([
        [[600., 0., 160.], [0., 600., 120.], [0., 0., 1.]] for each in pkl_data], dtype=dtype, device=device)

    smpl_male = SMPL(gender='male', batch_size=2048, device=device, dtype=dtype, hybrik_joints=True)
    smpl_female = SMPL(gender='female', batch_size=2048, device=device, dtype=dtype, hybrik_joints=True)
    bbox = []
    vertices_visibility = []
    smpl_mask = []
    for idx in range(len(file_name)):
        smpl = smpl_male if gender[idx] == 'male' else smpl_female
        _smpl_j29, smpl_vertices, _smpl_j17 = smpl(
            beta=smpl_beta[idx:idx + 1], pose=smpl_theta[idx:idx + 1], transl=smpl_transl[idx:idx + 1])

        each_k = cam_intrinsics[idx:idx + 1]

        vertices_uvd = each_k[:, None].matmul(smpl_vertices[:, :, :, None])[:, :, :, 0]
        vertices_uvd[:, :, :2] /= vertices_uvd[:, :, 2:]  # 1,6890,3
        bbox.append(kp2bbox(vertices_uvd[:, :, :2], image_wh[idx:idx + 1]).cpu().numpy())  # [u_min,v_min,w,h]
        vertices_visibility.append(mesh2visibility(
            vertices_uvd, smpl.faces.astype(np.int32), image_wh[idx:idx + 1], device)[0])
        triangles = vertices_uvd[:, smpl.faces.astype(np.int32), :2]  # bs, 13776, 3, 2
        smpl_mask.append(smpl2segmentation(triangles, bbox[-1], device)[0])

    bbox = np.concatenate(bbox)
    vertices_visibility = np.concatenate(vertices_visibility)
    smpl_mask = np.array(smpl_mask)
    data = {
        'id': np.arange(len(file_name)),
        # video_name  frame_idx
        'file_name': np.array(file_name),
        'image_wh': image_wh,
        'cam_intrinsics': cam_intrinsics.cpu().numpy(),
        'gender': gender,
        'smpl_theta': smpl_theta.cpu().numpy(),
        'smpl_beta': smpl_beta.cpu().numpy(),
        'smpl_transl': smpl_transl.cpu().numpy(),
        'bbox': bbox,
        'vis': vertices_visibility,
        'smpl_mask': smpl_mask,
    }
    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')


def prepare_up3d(device='cuda:0', dtype=torch.float64, dataset_path=up3d_path, demo=False):
    out_file = os.path.join(dataset_path, f'densepose_high.pt' if not demo else f'densepose_demo.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare the whole up-3d dataset {dataset_path}')

    img_file = sorted(glob(os.path.join(dataset_path, 'up-3d', '*_image.png')))
    crop_file = sorted(glob(os.path.join(dataset_path, 'up-3d', '*_fit_crop_info.txt')))
    smpl_file = sorted(glob(os.path.join(dataset_path, 'up-3d', '*_body.pkl')))
    quality_file = sorted(glob(os.path.join(dataset_path, 'up-3d', '*_quality_info.txt')))
    assert len(img_file) == len(crop_file) and len(img_file) == len(smpl_file)  # 8515
    if demo:
        img_file, crop_file, smpl_file, quality_file = (
            img_file[:100], crop_file[:100], smpl_file[:100], quality_file[:100])

    file_name = np.array([os.path.basename(each) for each in img_file])
    image_wh = np.array([get_image_size(each) for each in img_file])

    smpl_info = [pkl.load(open(each, 'rb'), encoding='latin1') for each in smpl_file]
    # print(np.unique(np.array([each['rt'] for each in smpl_info]))) # fake extrinsic rotation
    # print(np.unique(np.array([each['f'] for each in smpl_info]))) # == 5000 fixed focal length
    # print(np.unique(np.array([each['trans'] for each in smpl_info]))) # fake
    gender = np.array(['neutral' for each in smpl_info])
    smpl_theta = torch.tensor([each['pose'] for each in smpl_info], dtype=dtype, device=device).reshape(-1, 24, 3)
    smpl_beta = torch.tensor([each['betas'] for each in smpl_info], dtype=dtype, device=device).reshape(-1, 10)
    smpl_transl = torch.tensor([each['t'] for each in smpl_info], dtype=dtype, device=device).reshape(-1, 3)

    crop_info = np.array([open(each, 'r').read().split() for each in crop_file], dtype=np.int64)  # 8515,6
    u_min, v_min, u_max, v_max = crop_info[:, -2], crop_info[:, -4], crop_info[:, -1], crop_info[:, -3]
    resolution_w, resolution_h = crop_info[:, 1], crop_info[:, 0]
    # up-3d dataset uses fixed local length and scales bbox
    # calculate equivalent camera intrinsic parameters to crop and scale bbox
    fx, fy = (u_max - u_min) / resolution_w * 5000, (v_max - v_min) / resolution_h * 5000
    cx, cy = (u_max + u_min) / 2, (v_max + v_min) / 2
    cam_intrinsics = np.array([[fx[i], 0, cx[i], 0, fy[i], cy[i], 0, 0, 1] for i in range(len(cx))]).reshape(-1, 3, 3)

    smpl = SMPL(gender='neutral', batch_size=8515, device='cpu', dtype=dtype, hybrik_joints=False)
    smpl_j24, smpl_vertices, smpl_j17 = smpl(beta=smpl_beta, pose=smpl_theta, transl=smpl_transl)

    uvd = torch.tensor(cam_intrinsics[:, None], dtype=dtype, device=device
                       ).matmul(smpl_vertices[:, :, :, None].to(device))[:, :, :, 0]
    uvd[:, :, :2] /= uvd[:, :, 2:]
    bbox = kp2bbox(uvd[:, :, :2], image_wh).cpu().numpy()  # [u_min,v_min,w,h]

    smpl_j24, smpl_vertices, smpl_j17 = None, None, None
    smpl_mask = []
    for frame_idx in range(uvd.shape[0]):
        triangles = uvd[frame_idx:frame_idx + 1, smpl.faces.astype(np.int32), :2]  # bs, 13776, 3, 2
        smpl_mask.append(smpl2segmentation(triangles, bbox[frame_idx:frame_idx + 1], device)[0])
    smpl_mask = np.array(smpl_mask)

    vertices_visibility = []
    for idx in range(len(file_name)):
        # if idx % 1000 == 0: print(idx, '/8515')
        vertices_visibility.append(mesh2visibility(
            uvd[idx:idx + 1], smpl.faces.astype(np.int32), image_wh[idx:idx + 1], device)[0])
    vertices_visibility = np.concatenate(vertices_visibility)

    quality = np.array([open(each, 'r').read().split() for each in quality_file])
    high_quality = np.where(quality == 'high')[0]  # (8128,)
    data = {
        'id': np.arange(len(file_name))[high_quality],
        # video_name  frame_idx
        'file_name': file_name[high_quality],
        'image_wh': image_wh[high_quality],
        'cam_intrinsics': cam_intrinsics[high_quality],
        'gender': gender[high_quality],
        'smpl_theta': smpl_theta[high_quality].cpu().numpy(),
        'smpl_beta': smpl_beta[high_quality].cpu().numpy(),
        'smpl_transl': smpl_transl[high_quality].cpu().numpy(),
        'bbox': bbox[high_quality],
        'vis': vertices_visibility[high_quality],
        'smpl_mask': smpl_mask[high_quality],
    }
    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')


def prepare_bedlam(interval=1, split='train', device='cuda:0', dtype=torch.float64, dataset_path=bedlam_path,
                   demo=False):
    assert split in ['train', 'val']
    out_file = os.path.join(dataset_path, f'densepose_{split}_{interval}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare bedlam {split}set {dataset_path}')

    processed_f = sorted(glob(os.path.join(dataset_path, 'processed_labels', '*.npz')))  # 30 files
    # https://github.com/pixelite1201/BEDLAM/tree/master/data_processing
    closeup_names = ['20221011_1_250_batch01hand_closeup_suburb_a',
                     '20221011_1_250_batch01hand_closeup_suburb_b',
                     '20221011_1_250_batch01hand_closeup_suburb_c',
                     '20221011_1_250_batch01hand_closeup_suburb_d',
                     '20221012_1_500_batch01hand_closeup_highSchoolGym',
                     '20221019_1_250_highbmihand_closeup_suburb_b',
                     '20221019_1_250_highbmihand_closeup_suburb_c', ]
    # split https://bedlam.is.tue.mpg.de/imagesgt.html
    # val_split_names = ['20221018_1_250_batch01hand_zoom_suburb_b',
    #                    '20221018_3-8_250_batch01hand',
    #                    '20221018_3_250_batch01hand_orbit_archVizUI3_time15',
    #                    '20221019_3-8_250_highbmihand_orbit_stadium', ]
    # due to train and val splits have complete overlap subjects, we reselect val split
    val_split_idx = list(range(21, 27))
    train_split_idx = list(range(21)) + list(range(27, 30))
    train_sub, val_sub = [], []
    for scene_idx, each_f in enumerate(processed_f):
        processed_data = np.load(each_f)
        if scene_idx in train_split_idx:
            train_sub = train_sub + list(set(processed_data['sub']))
        else:
            val_sub = val_sub + list(set(processed_data['sub']))
    all_sub = sorted(list(set(train_sub + val_sub)))  # 271
    print('n of overlap subjects:', len(set(train_sub)) + len(set(val_sub)) - len(all_sub))

    smpl = SMPL(gender='neutral', batch_size=3, device=device, dtype=dtype, hybrik_joints=False, num_betas=11)
    data = {
        'id': [],
        'file_name': [],
        'video_name': [],
        'frame_idx': [],
        'image_wh': [],
        'bbox': [],
        'cam_intrinsics': [],
        'gender': [],
        'smpl_theta': [],
        'smpl_beta': [],
        'smpl_transl': [],

        'vis': [],
        'smpl_mask': [],
        # 'mask_files_pattern': [],
    }
    for scene_idx, each_f in enumerate(processed_f):
        # is_val_split = any([val_name + '.npz' in each_f for val_name in val_split_names])
        if split == 'train' and scene_idx in val_split_idx:
            continue
        if split == 'val' and scene_idx in train_split_idx:
            continue
        print('start processing scene:', scene_idx)
        processed_data = np.load(each_f)
        processed_data = {key: processed_data[key] for key in processed_data}

        scene_name = os.path.basename(each_f)[:-4]
        seq_info = pd.read_csv(os.path.join(dataset_path, 'data', scene_name, 'be_seq.csv')).to_dict('list')
        for subject_name in sorted(set(processed_data['sub'])):
            sub_idx = all_sub.index(subject_name)
            sub_mask = np.where(processed_data['sub'] == subject_name)
            seq_names = sorted(set([each.split('/')[0] for each in processed_data['imgname'][sub_mask]]))
            for seq_name in seq_names:
                seq_idx = int(seq_name[-6:])

                seq_idx_span = np.where(np.array(seq_info['Type']) == 'Group')[0][seq_idx:seq_idx + 2].tolist()
                seq_body_all = np.array(seq_info['Body'])[seq_idx_span[0] + 1:seq_idx_span[1]]
                assert sum([subject_name in each for each in seq_body_all]) == 1
                # https://github.com/pixelite1201/BEDLAM/issues/18
                per_idx_local = np.where([subject_name in each for each in seq_body_all])[0][0]

                sub_seq_mask = np.where((processed_data['sub'] == subject_name) * np.array(
                    [seq_name in each for each in processed_data['imgname']]))
                file_names = processed_data['imgname'][sub_seq_mask]
                file_names = [f'{scene_name}/png/{each}' for each in file_names]
                frame_idxes = [int(os.path.basename(each)[-8:-4]) for each in file_names]
                frame_valid = [each % interval == 0 for each in frame_idxes]

                ids = [int(f'7{scene_idx:02d}{sub_idx:03d}{seq_idx:03d}{frame_idx:04d}') for frame_idx in frame_idxes]
                video_name = [f'{scene_name}-{subject_name}-{seq_name}' for each in ids]

                if scene_name in closeup_names:
                    image_wh = [[720, 1280] for each in ids]
                else:
                    image_wh = [[1280, 720] for each in ids]

                cam_intrinsics = processed_data['cam_int'][sub_seq_mask]
                gender = [f'neutral' for each in ids]
                smpl_theta = processed_data['pose_cam'][sub_seq_mask]
                smpl_beta = processed_data['shape'][sub_seq_mask]
                smpl_transl = (processed_data['trans_cam'][sub_seq_mask] +
                               processed_data['cam_ext'][sub_seq_mask][:, :3, 3])

                _smpl_j29, smpl_vertices, _smpl_j17 = smpl(
                    beta=torch.tensor(smpl_beta[frame_valid], dtype=dtype, device=device),
                    pose=torch.tensor(smpl_theta[frame_valid], dtype=dtype, device=device),
                    transl=torch.tensor(smpl_transl[frame_valid], dtype=dtype, device=device), )

                vertices_uvd = torch.tensor(cam_intrinsics[frame_valid], dtype=dtype, device=device)[:, None].matmul(
                    smpl_vertices[:, :, :, None]).squeeze(-1)
                vertices_uvd[:, :, :2] /= vertices_uvd[:, :, 2:]  # 1,6890,3
                bbox = kp2bbox(vertices_uvd[:, :, :2], np.array(image_wh)[frame_valid]).cpu().numpy()

                mask_clothed_list = []  #
                mask_files_pattern = [
                    os.path.join(scene_name, 'masks', seq_name, f'*_{frame:04d}_{per_idx_local:02d}_*.png')
                    for frame in frame_idxes]
                for each in np.array(mask_files_pattern)[frame_valid]:
                    mask_files = glob(os.path.join(dataset_path, 'data', each))
                    masks = [np.array(Image.open(each)) for each in mask_files]
                    mask_clothed_each = np.any(np.array(masks) > 0, axis=0).astype(np.float64)
                    mask_clothed_each = mask_clothed_each if type(mask_clothed_each) is np.ndarray else np.zeros(
                        image_wh[0][::-1])
                    mask_clothed_list.append(mask_clothed_each)
                mask_clothed = np.stack(mask_clothed_list)[:, None]  # bs,1,1280,720

                not_self_occluded = np.concatenate(mesh2visibility(vertices_uvd, smpl.faces.astype(np.int32),
                                                                   np.array(image_wh)[frame_valid], device))
                not_occluded = torch.nn.functional.grid_sample(
                    torch.tensor(mask_clothed),  # bs, 1, 1280, 720
                    -1 + 2 * vertices_uvd[:, None, :, :2].cpu() / torch.tensor(image_wh)[frame_valid][:, None, None],
                    # bs,1,6890,2
                    mode='bilinear', padding_mode='zeros', align_corners=False,
                ).squeeze(1).squeeze(1)
                vis = not_self_occluded * (not_occluded.numpy() > 0.3)
                triangles = vertices_uvd[:, smpl.faces.astype(np.int32), :2]  # bs, 13776, 3, 2
                triangles_vis = vis[:, smpl.faces.astype(np.int32)].mean(axis=-1) > 0.5
                smpl_mask = np.array([
                    smpl2segmentation(triangles[i][triangles_vis[i]][None], bbox[i][None], device)[0]
                    for i in range(vis.shape[0])])
                # smpl_mask = np.array(smpl2segmentation(triangles, bbox, device))

                vis_valid = (vis.sum(-1) > 500)  # bs,
                data['id'].append(np.array(ids)[frame_valid][vis_valid])
                data['file_name'].append(np.array(file_names)[frame_valid][vis_valid])
                data['video_name'].append(np.array(video_name)[frame_valid][vis_valid])
                data['frame_idx'].append(np.array(frame_idxes)[frame_valid][vis_valid])
                data['image_wh'].append(np.array(image_wh)[frame_valid][vis_valid])
                data['bbox'].append(bbox[vis_valid])
                data['cam_intrinsics'].append(cam_intrinsics[frame_valid][vis_valid])
                data['gender'].append(np.array(gender)[frame_valid][vis_valid])
                data['smpl_theta'].append(smpl_theta[frame_valid][vis_valid])
                data['smpl_beta'].append(smpl_beta[frame_valid][vis_valid])
                data['smpl_transl'].append(smpl_transl[frame_valid][vis_valid])

                # data['mask_files_pattern'].append(np.array(mask_files_pattern)[frame_valid][vis_valid])
                data['vis'].append(vis[vis_valid])
                data['smpl_mask'].append(smpl_mask[vis_valid])
                if demo: break
            if demo: break
    for k in data.keys():
        data[k] = np.concatenate(data[k])
    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}\n\n')


def compress_bedlam():
    # compress png file to WEBP format (1300kb to 300kb).  It takes about 70 hours to finish.
    new_img_path = os.path.join(bedlam_path, 'img_webp')
    for i, each_scene in enumerate(sorted(list(glob(os.path.join(bedlam_path,'data/*'))))):
        for j, each_img_seq in enumerate(sorted(list(glob(os.path.join(each_scene,'png/*'))))):
            print(f'start compress images of {i}th scene,{j}th sequence')
            new_img_path_seq = os.path.join(new_img_path, os.sep.join(os.path.normpath(each_img_seq).split(os.sep)[-3:]))
            if not os.path.exists(new_img_path_seq): os.makedirs(new_img_path_seq)
            for each_img in sorted(list(glob(os.path.join(each_img_seq,'*.png')))):
                new_file = os.path.join(new_img_path, os.sep.join(os.path.normpath(each_img).split(os.sep)[-4:]) + '.webp')
                Image.open(each_img).save(new_file, 'WEBP', quality=100)  # still loss a little detail
                # Image.open(each_img).save(new_file, 'WEBP', lossless=True)


def prepare_ssp_3d(device='cuda:0', dtype=torch.float64, dataset_path=ssp_path):
    out_file = os.path.join(dataset_path, f'densepose.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare ssp_3d dataset {dataset_path}')
    labels = np.load(os.path.join(dataset_path,'labels.npz'))
    file_name = labels['fnames']
    gender = labels['genders'].astype('<U6')
    gender[gender=='f'] = 'female'
    gender[gender == 'm'] = 'male'

    smpl_theta = labels['poses']
    smpl_beta = labels['shapes']
    smpl_transl = labels['cam_trans']

    image_wh = np.array([get_image_size(os.path.join(dataset_path,'images',each)) for each in file_name]) # [[512,512],***]
    cam_intrinsics = np.array([[5000.0, 0, 512.0/2, 0, 5000.0, 512.0/2, 0, 0, 1] for i in range(len(file_name))]).reshape(-1, 3, 3)

    video_name = np.array([each.split('frame')[0] for each in file_name])

    smpl_male = SMPL(gender='male', batch_size=len(file_name), device=device, dtype=dtype, hybrik_joints=True)
    smpl_female = SMPL(gender='female', batch_size=len(file_name), device=device, dtype=dtype, hybrik_joints=True)

    smpl_j24_male, smpl_vertices_male, smpl_j17_male = smpl_male(beta=smpl_beta, pose=smpl_theta, transl=smpl_transl)
    smpl_j24_female, smpl_vertices_female, smpl_j17_female = smpl_female(beta=smpl_beta, pose=smpl_theta, transl=smpl_transl)
    smpl_vertices = (smpl_vertices_female * torch.tensor((gender=='female')[:,None,None],dtype=dtype, device=device) +
                     smpl_vertices_male * torch.tensor((gender=='male')[:,None,None],dtype=dtype, device=device))

    uvd = torch.tensor(cam_intrinsics[:, None], dtype=dtype, device=device
                       ).matmul(smpl_vertices[:, :, :, None].to(device))[:, :, :, 0]
    uvd[:, :, :2] /= uvd[:, :, 2:]
    bbox = kp2bbox(uvd[:, :, :2], image_wh).cpu().numpy()  # [u_min,v_min,w,h]

    smpl_mask = []
    for frame_idx in range(uvd.shape[0]):
        triangles = uvd[frame_idx:frame_idx + 1, smpl_male.faces.astype(np.int32), :2]  # bs, 13776, 3, 2
        smpl_mask.append(smpl2segmentation(triangles, bbox[frame_idx:frame_idx + 1], device)[0])
    smpl_mask = np.array(smpl_mask)

    vertices_visibility = []
    for idx in range(len(file_name)):
        # if idx % 1000 == 0: print(idx, '/8515')
        vertices_visibility.append(mesh2visibility(
            uvd[idx:idx + 1], smpl_male.faces.astype(np.int32), image_wh[idx:idx + 1], device)[0])
    vertices_visibility = np.concatenate(vertices_visibility)

    data = {
        'id': np.arange(len(file_name)),
        'file_name': file_name,
        'image_wh': image_wh,
        'bbox': bbox,
        'video_name':video_name,
        'frame_idx':np.array([int(each.split('frame')[-1].split('.')[0][1:]) for each in file_name]),
        'cam_intrinsics': cam_intrinsics,
        'gender': gender,
        'smpl_theta': smpl_theta,
        'smpl_beta': smpl_beta,
        'smpl_transl': smpl_transl,

        'vis': vertices_visibility,
        'smpl_mask': smpl_mask,
    }
    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')


def prepare_coco_densepose(split='train', dataset_path=coco_path):
    assert split in ['train2014', 'valminusminival2014', 'minival2014']
    out_file = os.path.join(dataset_path, f'densepose_{split}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare coco {split}set {dataset_path}')

    ann_file = os.path.join(os.path.normpath(dataset_path), 'annotations', f'densepose_{split}.json')
    coco = COCO(ann_file)

    image_ids = sorted(coco.getImgIds())
    imgs = coco.loadImgs(image_ids)

    DP = dp_utils.DensePoseMethods()

    output = []

    for image_ann in imgs:
        obj_ann_ids = coco.getAnnIds(imgIds=image_ann['id'])  # , iscrowd=False)
        obj_ann = coco.loadAnns(obj_ann_ids)
        for obj in obj_ann:
            if obj['area'] <= 0 or obj['category_id'] != 1:  # not human
                print('coco ann error ', image_ann['id'])
                continue
            bbox = obj['bbox']  # list:4 xywh

            if 'dp_masks' not in obj.keys():
                continue
                dp_masks_decoded = None
                dp_x_global, dp_y_global = None, None
                dp_I, dp_U, dp_V = None, None, None
                dp_face_vertices, dp_bcs = None, None
            else:
                dp_masks_decoded = np.zeros((256, 256), dtype=np.int8)
                for i, RLE_i in enumerate(obj.get('dp_masks')):
                    if RLE_i != []:
                        mask_i = mask_utils.decode(RLE_i)
                        assert mask_i.shape == (256, 256), f"RLE size error {obj['id']}"
                        dp_masks_decoded[mask_i > 0] = i + 1

                dp_x_global = np.array([each / 256 * bbox[2] + bbox[0] for each in obj['dp_x']], dtype=np.float64)
                dp_y_global = np.array([each / 256 * bbox[3] + bbox[1] for each in obj['dp_y']], dtype=np.float64)
                dp_I = np.round(obj.get('dp_I')).astype(np.int8)
                dp_U = np.array(obj.get('dp_U'), dtype=np.float64)
                dp_V = np.array(obj.get('dp_V'), dtype=np.float64)
                dp_face_vertices, dp_bcs = [], []
                for i, (ii, uu, vv) in enumerate(zip(dp_I, dp_U, dp_V)):
                    # Convert IUV to FBC (faceIndex and barycentric coordinates.)
                    face_index, bc1, bc2, bc3 = DP.IUV2FBC(ii, uu, vv)
                    face_vertex = DP.All_vertices[DP.FacesDensePose[face_index]] - 1  # [ 609, 1297,  591] ∈[1-1,6890-1]
                    bc = [bc1, bc2, bc3]  # (0.422, 0.254, 0.323)
                    dp_face_vertices.append(face_vertex)
                    dp_bcs.append(bc)
                dp_face_vertices = np.array(dp_face_vertices)
                dp_bcs = np.array(dp_bcs, dtype=np.float64)

                def pad_ary(ary, n=200):
                    pad_len = n - ary.shape[0]
                    pad_ary = np.repeat(ary[:1] * 0, pad_len, axis=0)
                    return np.concatenate([ary, pad_ary], axis=0)

                dp_n = dp_x_global.shape[0]
                dp_x_global, dp_y_global = pad_ary(dp_x_global), pad_ary(dp_y_global)
                dp_I, dp_U, dp_V = pad_ary(dp_I), pad_ary(dp_U), pad_ary(dp_V)
                dp_face_vertices, dp_bcs = pad_ary(dp_face_vertices), pad_ary(dp_bcs)

            output.append({
                'id': obj['id'],  # 'image_id': obj['image_id'],
                'file_name': image_ann['file_name'],
                'image_wh': np.array([image_ann['width'], image_ann['height']]),
                'bbox': np.array(bbox, dtype=np.float64),

                # The segmentation format depends on whether the instance represents  a single object (iscrowd=0 in
                # which case polygons are used) or a collection of objects (iscrowd=1 in which case RLE is used).
                # Note that a single object (iscrowd=0) may require multiple polygons, for example if occluded. Crowd
                # annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people).
                # 'iscrowd': obj['iscrowd'],  # 0 or 1

                # 'coco_kp17': np.array(obj['keypoints']),  # list:51 0not_visible 1labeled 2 visible
                # 'num_keypoints': obj['num_keypoints'],  # e.g. 2 or 17

                # 'segmentation': np.array(obj['segmentation'], dtype=object),  # ListOfList or dict
                # 'area': obj['area'],  # 27789.11055

                # dp_masks : list len(14)  {'counts': str,'size':[256,256](fixed)}
                # dp_masks: RLE encoded dense masks (`dict` containing keys `counts` and `size`).
                # The masks are typically of size `256x256`, they define segmentation within the bounding box.
                # 'dp_masks_decoded': dp_masks_decoded,
                # dp_mask: bg/fg segmentation within the bounding box. the encoding size is also [256,256]
                'dp_mask': mask_utils.encode((dp_masks_decoded != 0).astype(np.uint8).copy(order='F'))['counts'],

                'dp_n': dp_n,  # 144 # num of annotated points
                # # 'dp_I' : The patch index that indicates which of the 24 surface patches the point is on.
                # 'dp_I': dp_I,  # list (len=144) ∈ [1,24]
                # # 'dp_U', 'dp_V' :  Coordinates in the UV space. Each surface patch has a separate 2D parameterization.
                # 'dp_U': dp_U,  # list (len=144) ∈ [0,1]
                # 'dp_V': dp_V,  # list (len=144) ∈ [0,1]
                'dp_face_vertices': dp_face_vertices,  # (144, 3)
                'dp_bcs': dp_bcs,  # (144, 3)

                # 'dp_x', 'dp_y' :  The spatial coordinates of collected points on the image. The coordinates are scaled such that the bounding box size is 256x256.
                'dp_x_global': dp_x_global,  # list(len=144) pixel coordinates on original image
                'dp_y_global': dp_y_global,
            })
    data = {}
    for key in output[0].keys():
        data[key] = np.array([each[key] for each in output])

    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')
    return data


def prepare_coco(split='train2017', dataset_path=coco_path):
    assert split in ['train2017']
    out_file = os.path.join(dataset_path, f'coco_{split}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare coco {split}set {dataset_path}')

    ann_file = os.path.join(os.path.normpath(dataset_path), 'annotations', f'person_keypoints_{split}.json')

    coco = COCO(ann_file)
    classes = [c['name'] for c in coco.loadCats(coco.getCatIds())]
    assert classes == ['person'], "Incompatible category names with COCO. "

    image_ids = sorted(coco.getImgIds())

    output = []
    for image_ann in coco.loadImgs(image_ids):
        obj_ann_ids = coco.getAnnIds(imgIds=image_ann['id'], iscrowd=False)
        obj_ann = coco.loadAnns(obj_ann_ids)
        for obj in obj_ann:
            if obj['area'] <= 0 or obj['category_id'] != 1:  # not human
                continue
            coco_kp17 = np.array(obj['keypoints']).reshape((17, 3))
            coco_eval_joints = [7, 8, 9, 10, 13, 14, 15, 16]
            if (coco_kp17[coco_eval_joints, -1] > 0).sum() < 1:
                continue
            output.append({
                'id': obj['id'],
                'file_name': image_ann['file_name'],
                'image_wh': [image_ann['width'], image_ann['height']],
                'bbox': obj['bbox'],  # list:4 xywh
                'coco_kp17': coco_kp17,  # list:51 0not_visible 1labeled 2 visible
            })
    data = {}
    for key in output[0].keys():
        data[key] = np.stack([each[key] for each in output])

    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')
    return data


def prepare_3dhp(split='train_v2', dataset_path=hp3d_path):
    assert split in ['train_v2', 'test']
    out_file = os.path.join(dataset_path, f'3dhp_{split}.pt')
    if os.path.exists(out_file):
        print('file exists, skip', out_file)
        return
    print(f'\n\nPrepare 3DHP {split}set {dataset_path}')

    _ann_file_from_hybrik = os.path.join(dataset_path, f'annotation_mpi_inf_3dhp_{split}.json')
    test_split_joint_idx = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]]
    hp3d2h36m = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]
    output = []

    with open(_ann_file_from_hybrik, 'r') as fid:
        database = json.load(fid)
    # iterate through the annotations
    for ann_image, ann_annotations in zip(database['images'], database['annotations']):
        ann = dict()
        for k, v in ann_image.items():
            assert k not in ann.keys()
            ann[k] = v
        for k, v in ann_annotations.items():
            ann[k] = v

        j3d_cam = np.array(ann['keypoints_cam']) / 1000
        if not split == 'test':
            j3d_cam = j3d_cam[test_split_joint_idx]
        j3d_cam = j3d_cam[hp3d2h36m]  # convert to h36m joint order  #joint 8,9,10 should be discarded
        cam_intrinsics = np.array(ann['cam_param']['intrinsic_param'], dtype=np.float32)[:3, :3]

        video_name = ann['file_name'].split('/')[0 if split == 'test' else 3]

        output.append({  # np.array([]),
            # 'keypoints_vis': np.array(ann['keypoints_vis']),  # fake annotation, all are visible
            '3dhp_j17': j3d_cam,  # h36m joint order  #joint 8,9,10 should be discarded

            'id': ann['image_id'],
            'video_name': video_name,
            'frame_idx': ann['frame_idx'],
            'bbox': np.array(ann['bbox']),
            'image_wh': np.array([ann['width'], ann['height']]),
            'file_name': ann['file_name'],
            'cam_intrinsics': cam_intrinsics,
        })
        # if split == 'test':
        #     output[-1]['activity_id'] = ann['activity_id']

    data = {}
    for key in output[0].keys():
        data[key] = np.stack([each[key] for each in output])

    joblib.dump(data, out_file)
    print(f'Done len:{data["id"].shape[0]}')
    return output


if __name__ == '__main__':
    '''           
    'dataset_name', 'id', 'file_name', 'image_wh', 'bbox', 'video_name', 'frame_idx', 
    'cam_intrinsics',  'gender', 'smpl_theta', 'smpl_beta', 'smpl_transl', 'vis', 'smpl_mask',
    'dp_n', 'dp_face_vertices', 'dp_bcs', 'dp_x_global', 'dp_y_global', 'dp_mask',
    '''
    print()
    # prepare_bedlam(interval=3, split='val')  #, demo=True)
    # prepare_bedlam(interval=3, split='train')  # ,demo=True)
    # prepare_ssp_3d()
    # prepare_up3d(demo=True)
    # prepare_surreal(split='test')
    # prepare_3dpw(split='train', demo=True)
    # prepare_h36m(interval=500, split='test')

    # prepare_coco_densepose(split='minival2014')
    # prepare_coco_densepose(split='valminusminival2014')
    # prepare_coco_densepose(split='train2014')

    # prepare_up3d()

    # prepare_surreal(split='test')
    # prepare_surreal(split='train')

    # prepare_3dpw(split='train')
    # prepare_3dpw(split='validation')
    # prepare_3dpw(split='test')
    # prepare_3dpw(split='all')

    # prepare_h36m(interval=20, split='train')
    # prepare_h36m(interval=20, split='test')
    # prepare_h36m(interval=5, split='train')
    # prepare_h36m(interval=5, split='test')

    # prepare_coco(split='train2017')

    # prepare_3dhp(split='test')
    # prepare_3dhp(split='train_v2')
