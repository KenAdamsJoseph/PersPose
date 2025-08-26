import os
import shutil

import joblib
import itertools
import bisect
import sys
import numpy as np
import torch
import pickle as pkl
import pycocotools.mask as mask_utils
from torch.utils.data import Dataset

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root)
from common import bbox
from common.smpl_wrapper import SMPL

# In these videos, actors are not occluded by objects or pedestrians.
multi_person = [
    "courtyard_arguing_00", "courtyard_capoeira_00", "courtyard_captureSelfies_00", "courtyard_dancing_00",
    "courtyard_dancing_01", "courtyard_giveDirections_00", "courtyard_goodNews_00", "courtyard_hug_00",
    "courtyard_rangeOfMotions_00", "courtyard_rangeOfMotions_01", "courtyard_shakeHands_00", "courtyard_warmWelcome_00",
    "downtown_warmWelcome_00", "downtown_upstairs_00", ]
single_person = [
    "courtyard_bodyScannerMotions_00", "courtyard_relaxOnBench_00", "courtyard_relaxOnBench_01", "downtown_arguing_00",
    "downtown_walkDownhill_00", "downtown_walkUphill_00", "outdoors_climbing_00", "outdoors_climbing_01",
    "outdoors_fencing_01", "courtyard_laceShoe_00", "courtyard_jumpBench_01", ]
pw3d_video_with_valid_masks = multi_person + single_person
# The actor is occluded by a chair.
h36m_video_with_invalid_masks = ['_act_04_', '_act_06_', '_act_09_', '_act_11_', ]


class HumanMesh(Dataset):
    def __init__(self, dataset_name, db_file, img_path, interval=1):
        self.name = dataset_name
        self.data = joblib.load(os.path.join(prj_root, db_file), 'r')
        self.img_path = img_path
        self.interval = interval

        num_betas = 11  # bedlam
        self.smpl = {g: SMPL(gender=g, batch_size=1, dtype=torch.float64, device='cpu', num_betas=num_betas) for g in
                     ['male', 'female', 'neutral']}
        self.smpl_part = pkl.load(
            open(os.path.join(prj_root, "./data/smpl_file/segm_per_v_overlap_from_surreal.pkl"), 'rb')
        )

        freq_file = os.path.join(prj_root, db_file + '_freq.npy')
        if not os.path.exists(freq_file):
            freq = self.data['vis'].mean(axis=0)
            np.save(freq_file, freq)
        else:
            freq = np.load(freq_file)
        dp_weight = 1 / (freq + 0.1)  # smooth the weight
        self.dp_weight = dp_weight / dp_weight.mean()

    def __len__(self):
        return len(self.data['id'])//self.interval

    def __getitem__(self, idx, visualize=False):
        idx *= self.interval
        res = {'idx': idx}
        for attr in self.data.keys():
            res[attr] = self.data[attr][idx].copy()
        res['dataset_name'] = 'hps_' + self.name
        res['file_name'] = os.path.join(prj_root, self.img_path, res['file_name'])
        if self.name == 'bedlam':
            res['smpl_theta'] = res['smpl_theta'].reshape(24, 3)
            if not os.path.exists(res['file_name']):
                res['file_name'] = res['file_name'][:-4] + '.jpg'
        else:
            res['smpl_beta'] = np.concatenate([res['smpl_beta'], res['smpl_beta'][:1] * 0])
        res['gender'] = str(res['gender'])
        res['mask'] = mask_utils.decode({'counts': bytes(res.pop('smpl_mask')), 'size': [256, 256]})
        res['bbox_mask'] = res['bbox'].copy()
        if self.name == 'up3d':
            res['mask_valid'] = False
        elif self.name == 'surreal':
            res['mask_valid'] = True
        elif self.name == '3dpw':
            if res['video_name'][:-2] in pw3d_video_with_valid_masks:
                res['mask_valid'] = True
            else:
                res['mask_valid'] = False
        elif self.name == 'h36m':
            if any([each in res['video_name'] for each in h36m_video_with_invalid_masks]):
                res['mask_valid'] = False
            else:
                res['mask_valid'] = True
        elif self.name == 'bedlam':
            res['mask_valid'] = True

        ortho_twist = self.smpl[res['gender']].cal_ortho_twist(beta=res['smpl_beta'][None], pose=res['smpl_theta'][None])
        res['smpl_ortho_twist'] = ortho_twist.cpu().numpy()[0]

        j24, vertices, j17 = self.smpl[res['gender']](
            beta=res['smpl_beta'][None], pose=res['smpl_theta'][None], transl=res['smpl_transl'][None])
        res['smpl_j24'] = j24.cpu().numpy()[0]
        res['smpl_j17'] = j17.cpu().numpy()[0]
        res['smpl_vertices'] = vertices.cpu().numpy()[0]

        cam_intrinsics = torch.tensor(res['cam_intrinsics'], dtype=torch.float64, device='cpu')  # 3,3
        uvd = cam_intrinsics.reshape(1, 1, 3, 3).matmul(vertices.unsqueeze(-1)).squeeze(-1)
        uvd[:, :, :2] /= uvd[:, :, 2:]  # bs, 6890, 3

        res['dp_uv'] = uvd[0, :, :2].cpu().numpy()  # 6890,2
        res['dp_face'] = np.arange(6890).reshape(6890, 1).repeat(3, 1)  # 6890,3
        res['dp_bc'] = np.ones((6890, 3)) / 3  # barycentric coordinates.
        res['dp_valid'] = res['vis']  # (6890,)
        res['dp_weight'] = self.dp_weight[:, None].repeat(3, 1)
        if self.name in ['up3d', '3dpw', 'h36m', 'surreal', 'bedlam']:
            for k in ['leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']:
                res['dp_valid'][self.smpl_part[k]] = False
        if self.name in ['up3d', '3dpw', ]:
            for k in ['leftToeBase', 'rightToeBase', 'leftFoot', 'rightFoot']:
                res['dp_valid'][self.smpl_part[k]] = False

        rest_pose_v = self.smpl[res['gender']].cal_rest_pose(res['smpl_beta'][None], True)[1]
        res['height'] = np.array([(rest_pose_v[:, :, 1].max() - rest_pose_v[:, :, 1].min()).item(),])

        xyxy = np.concatenate([res['dp_uv'].min(axis=0), res['dp_uv'].max(axis=0)])
        xyxy = np.clip(xyxy, -0.2 * max(res['image_wh']), 1.2 * max(res['image_wh']))
        res['bbox'] = np.array(bbox.xyxy2xywh(xyxy))

        if visualize:
            res['vertices'] = vertices.cpu().numpy()
        return res

    def visualize(self, idx):
        data = self.__getitem__(idx, visualize=True)

        from common.visualize import draw_joints2D
        draw_joints2D(data['dp_uv'][data['dp_valid']], data['file_name'], radius=1, suffix='_densepose')
        if data['mask_valid']:
            draw_joints2D(
                np.stack([
                    [each / 256 * data['bbox'][2] + data['bbox'][0]
                     for each in np.where(data['mask'] > 0)[1]],
                    [each / 256 * data['bbox'][3] + data['bbox'][1]
                     for each in np.where(data['mask'] > 0)[0]]
                ], axis=-1),
                data['file_name'], radius=1, suffix='_mask')

        draw_joints2D(
            np.stack([data['bbox'][:2], data['bbox'][:2] + data['bbox'][2:]]),
            data['file_name'], radius=5, suffix='_bbox')

        # import trimesh
        # vertices = data['vertices'][0]
        # vertices[:, 1:] *= -1
        # m = trimesh.Trimesh(vertices=data['vertices'][0], faces=self.smpl[data['gender']].faces)
        # face_vis = (data['vis'][self.smpl['neutral'].faces.astype(np.long)].sum(axis=-1) == 3)
        # f_color = torch.tensor(trimesh.visual.color.interpolate(face_vis, color_map="viridis")[:, :3])
        # m.visual.face_colors = f_color
        # m.show()
        print(os.path.normpath(data['file_name']))


if __name__ == '__main__':
    # h36m_train = HumanMesh('h36m', r"./data/h36m/densepose_train_5.pt", r"./data/h36m/images")
    # h36m_test = HumanMesh('h36m', r"./data/h36m/densepose_test_5.pt", r"./data/h36m/images")
    # pw3d_train = HumanMesh('3dpw', r"./data/3dpw/densepose_train.pt", r"./data/3dpw")
    # pw3d_validation = HumanMesh('3dpw', r"./data/3dpw/densepose_validation.pt", r"./data/3dpw")
    # pw3d_test = HumanMesh('3dpw', r"./data/3dpw/densepose_test.pt", r"./data/3dpw")
    # surreal_train = HumanMesh('surreal', r"./data/SURREAL_v1/densepose_train.pt", r"./data/SURREAL_v1")
    # surreal_test = HumanMesh('surreal', r"./data/SURREAL_v1/densepose_test.pt", r"./data/SURREAL_v1")
    # bedlam_train = HumanMesh('bedlam', r"./data/bedlam/densepose_train_3.pt", r"./data/bedlam/data")
    # bedlam_val = HumanMesh('bedlam', r"./data/bedlam/densepose_val_3.pt", r"./data/bedlam/data")
    up3d = HumanMesh('up3d', r"./data/up-3d/densepose_high.pt", r"./data/up-3d/up-3d")

    # ssp_3d = HumanMesh('ssp_3d', r"./data/ssp_3d/densepose.pt", r"./data/ssp_3d/images")

    def get_fov(dataset):
        f = dataset.data['cam_intrinsics'][:, 0, 0]
        bbox_size = dataset.data['bbox'][:, 2:].max(axis=1)
        f_crop = f * 256 / bbox_size
        fov_crop = 2 * np.arctan(256 * 1 / f_crop / 2) / 3.14159 * 180
        return fov_crop

    def get_pr_degree(dataset):
        bbox_center = dataset.data['bbox'][:, :2] + dataset.data['bbox'][:, 2:] / 2
        uv1 = np.concatenate((bbox_center, bbox_center[:,:1]*0+1),axis=-1)
        xy1 = np.linalg.inv(dataset.data['cam_intrinsics']) @ uv1[:, :, None]
        pr_degree = np.arctan(np.linalg.norm(xy1[:, :2, 0], axis=-1)) / 3.14159 * 180
        return pr_degree

    # torch.save(get_fov(h36m_train), 'h36m_train_fov_crop.torchsave')
    # torch.save(get_pr_degree(h36m_train), 'h36m_train_pr_degree.torchsave')
    # h36m_train, h36m_test, pw3d_train, pw3d_validation, pw3d_test, surreal_train, surreal_test, bedlam_train, bedlam_val,  up3d
    for dataset in [up3d, ]:
        print(dataset.name, dataset.__len__())
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=32, num_workers=0,
            shuffle=True)
        batch = iter(loader)._next_data()
        print(batch.keys())

        for idx in torch.randint(0, len(dataset), (128,)).tolist():  # np.random.randint(0, len(dataset), (50,)):
            dataset.visualize(idx)
    print()
