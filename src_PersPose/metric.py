import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.uv_heatmap import uv2heatmap
from common.loss import p_mpjpe, swing_radian
from common.visualize import draw_joints2D
from common.geometry import rot6d_to_rotmat, batch_rodrigues


def xyz2uv_density(xyz, cam_intrinsic):
    """
    :param xyz: tensor(bs,j_num,3) joint position in camera coordinate
    :param cam_intrinsic: tensor(bs,3,3)  camera intrinsic
    :return: tensor(bs,j_num,2) pixel coordinate and  tensor(bs,j_num,1) density(d/f)
    """
    uv1_z = cam_intrinsic.matmul(xyz.permute(0, 2, 1)).permute(0, 2, 1)
    f = (cam_intrinsic[:, 0, 0] + cam_intrinsic[:, 1, 1]) / 2  # (bs,) f=(fx+fy)/2
    return uv1_z[:, :, :2] / uv1_z[:, :, -1:], 1000 * uv1_z[:, :, -1:] / f.reshape(-1, 1, 1)


def mask_mean(inp, mask):
    """
    :param inp: tensor of any shape, e.g., (bs, d1, ...)
    :param mask: list or bool tensor, shape should match the leading dimensions of inp, e.g., (bs, d1, ...)
    :return: weighted mean
    """
    if type(mask) is list:
        mask = torch.tensor(mask, dtype=inp.dtype, device=inp.device)
    mask = mask.to(inp.device)

    # Ensure mask's shape is compatible with inp's shape
    if mask.shape != inp.shape[:len(mask.shape)]:
        raise ValueError("The shape of mask must match the leading dimensions of inp")

    # Expand mask dimensions to match inp dimensions
    expanded_mask = mask.reshape(list(mask.shape) + [1] * (len(inp.shape) - len(mask.shape))).expand_as(inp)

    # Compute the weighted mean
    weighted_sum = (inp * expanded_mask).sum()
    total_weight = expanded_mask.sum().float()
    return weighted_sum / (total_weight + 1e-7)


class MaskMse(nn.Module):
    def __init__(self, ):
        super(MaskMse, self).__init__()
        self.SE = nn.MSELoss(reduction='none')

    def forward(self, inp1, inp2, mask):
        """
        :param inp1: tensor(bs,...)
        :param inp2: tensor(bs,...)
        :param mask: see func mask_mean
        :return: weighted mean square error
        """
        assert inp1.shape == inp2.shape
        se = self.SE(inp1, inp2)
        return mask_mean(se, mask)


def scaled_diff(val_from, val_to):
    """ min_s \sigma_i ||scale*val_from_i-val_to_i||^2
    :param val_from: bs,n,d
    :param val_to: bs,n,d
    """
    scale = (val_from * val_to).sum((-1, -2)) / (val_from * val_from).sum((-1, -2))
    return scale[:,None,None] * val_from - val_to


def normed_diff(val_from, val_to):
    """
    Aligns val_from to val_to by finding the optimal scalar scale s and translation t
    that minimize the squared error:

         min_{s,t} sum_i || s*val_from_i + t - val_to_i ||^2.

    The optimum is given by:
         s = (sum_i (val_from_i - mean(val_from))·(val_to_i - mean(val_to)))
             / (sum_i ||val_from_i - mean(val_from)||^2)
         t = mean(val_to) - s * mean(val_from)

    :param val_from: Tensor of shape (bs, n, d)
    :param val_to:   Tensor of shape (bs, n, d)
    :return:         The difference after alignment, i.e.
                      (s*val_from + t) - val_to, with shape (bs, n, d)
    """
    # Compute the centroids along the n dimension (keeping dims for broadcasting)
    mean_from = val_from.mean(dim=1, keepdim=True)  # shape: (bs, 1, d)
    mean_to = val_to.mean(dim=1, keepdim=True)  # shape: (bs, 1, d)

    # Center the data
    val_from_centered = val_from - mean_from
    val_to_centered = val_to - mean_to

    # Compute optimal scale for each batch.
    # Sum over both n and d dimensions.
    numerator = (val_from_centered * val_to_centered).sum(dim=(1, 2))  # shape: (bs,)
    denominator = (val_from_centered ** 2).sum(dim=(1, 2))  # shape: (bs,)
    scale = numerator / denominator  # shape: (bs,)

    # Compute optimal translation for each batch.
    t = mean_to - scale.view(-1, 1, 1) * mean_from  # shape: (bs, 1, d)

    # Apply the transformation to val_from
    aligned_val_from = scale.view(-1, 1, 1) * val_from + t  # shape: (bs, n, d)

    # Return the difference (residual)
    return aligned_val_from - val_to


def cal_loss_metric(output, labels, inp, model):
    """
    :param output: 'uv', 'depth', 'density', 'xyz', 'xyz0'
    :param labels:
    :param inp:
    :return:
    """
    bs, _3, inp_h, inp_w = inp['img'].shape
    device = inp['img'].device

    # the sample index of each data set
    dataset_name = labels['dataset_name']
    coco_idx = torch.tensor([each == 'coco' for each in dataset_name], dtype=torch.bool, device=device)
    hp3d_idx = torch.tensor([each == '3dhp' for each in dataset_name], dtype=torch.bool, device=device)
    hps_idx = torch.tensor([each.startswith('hps_') for each in dataset_name], dtype=torch.bool, device=device)
    bedlam_idx = torch.tensor(['bedlam' in each for each in dataset_name], dtype=torch.bool, device=device)
    h_gt, h_pred = labels['height'][:, :, None], labels['height'][:, :, None]  # *0+1.8

    # ############hps-datasets(with smpl parameters)##################
    metric_joints = list(range(24))

    j3d_gt = torch.cat([labels['smpl_j24'], labels['smpl_j17'][:, :1]], dim=1)
    uv_gt, density_gt = xyz2uv_density(j3d_gt, labels['cam_intrinsics'])
    valid_hps = (uv_gt > 0).all(dim=-1) * (uv_gt[..., 0] < inp_w) * (uv_gt[..., 1] < inp_h) * hps_idx.unsqueeze(-1)

    loss_uv = mask_mean((output['uv'] - uv_gt).norm(dim=-1), (valid_hps*(0.1*bedlam_idx[:,None]+~bedlam_idx[:,None])))
    err_uv = mask_mean((output['uv'] - uv_gt).norm(dim=-1)[:, metric_joints], valid_hps[:, metric_joints])

    loss_density_root = 1.8 * mask_mean((output['density'][:, :1] / h_pred - density_gt[:, :1] / h_gt).abs(), hps_idx)
    err_density_root = mask_mean((output['density'][:, :1] - density_gt[:,:1]).abs(), hps_idx)

    depth_gt = j3d_gt[:, :, -1:] - j3d_gt[:, :1, -1:]
    t = 0 * (output['depth']/h_pred - depth_gt/h_gt).mean(dim=1,keepdim=True)
    loss_depth = 1.8 * mask_mean(100 * (output['depth']/h_pred - depth_gt/h_gt - t).abs() + 0.1 * t.abs(), valid_hps)
    err_depth = mask_mean(100 * (output['depth'] - depth_gt).abs(), valid_hps)
    err_depth_aligned = mask_mean(100 * normed_diff(output['depth'], depth_gt).abs(), valid_hps)

    # joint 3d position error of smpl datasets
    xyz_pred = 100 * output['xyz']
    xyz_gt = 100 * (j3d_gt - j3d_gt[:, :1])

    loss_xyz = 1.8 * mask_mean((xyz_pred/h_pred - xyz_gt/h_gt).norm(dim=-1), valid_hps)
    err_xyz_aligned = mask_mean(normed_diff(xyz_pred[:, metric_joints],xyz_gt[:, metric_joints]).norm(dim=-1),hps_idx)
    err_xyz = mask_mean((xyz_pred - xyz_gt).norm(dim=-1)[:, metric_joints], hps_idx)
    err_x = mask_mean((xyz_pred - xyz_gt)[:, :, 0].abs()[:, metric_joints], hps_idx)
    err_y = mask_mean((xyz_pred - xyz_gt)[:, :, 1].abs()[:, metric_joints], hps_idx)
    err_z = mask_mean((xyz_pred - xyz_gt)[:, :, 2].abs()[:, metric_joints], hps_idx)

    joints_with_twist = model.smpl.joints_with_twist # len==17
    pred_twist = F.normalize(output['ortho_twist'][:, joints_with_twist], p=2, dim=-1)   # bs, 17, 3
    gt_twist = F.normalize(labels['smpl_ortho_twist'][:, joints_with_twist], p=2, dim=-1)
    loss_twist = mask_mean((pred_twist - gt_twist).norm(dim=-1, p='fro'), hps_idx)
    err_twist = 180/torch.pi * mask_mean(swing_radian(pred_twist, gt_twist,reduce='none'), hps_idx)

    loss_beta = mask_mean((labels['smpl_beta'] - output['beta']).abs(), hps_idx)
    gt_leaf = batch_rodrigues(labels['smpl_theta'][:, model.leaf_idxes].reshape(-1, 3)).reshape(bs, 5, 9)
    loss_leaf = mask_mean((output['leaf_rot'] - gt_leaf).norm(dim=-1, p='fro'), hps_idx)

    pred_j24, pred_v, pred_j17 = model.smpl(output['beta'], output['theta'], output['transl'])
    pred_v = pred_v - pred_j17[:, :1]
    gt_v = labels['smpl_vertices'] - labels['smpl_j17'][:, :1]
    MPVPE = mask_mean((pred_v - gt_v).norm(dim=-1), hps_idx)

    pred_j14 = (pred_j17 - pred_j17[:, :1])[:, model.smpl.h36m_j14_idx]
    gt_j14 = (labels['smpl_j17'] - labels['smpl_j17'][:, :1])[:, model.smpl.h36m_j14_idx]
    MPJPE = mask_mean((pred_j14-gt_j14).norm(dim=-1), hps_idx)
    try:
        PA_MPJPE = p_mpjpe(pred_j14.detach().cpu().double().numpy(), gt_j14.detach().cpu().double().numpy())
        PA_MPJPE = mask_mean(torch.tensor(PA_MPJPE, device=MPJPE.device, dtype=MPJPE.dtype), hps_idx)
    except RuntimeError as e:
        PA_MPJPE = 0.03
        print('PA_MPJPE: ', e)

    # ############3dhp#####################
    # after data preprocessing, 3dhp joints are in h36m style.
    hp3d_eval_joints = [0, 2, 3, 5, 6, 12, 13, 15, 16]
    idx_h36m2smpl = [24, 2, 5, 8, 1, 4, 7, -1, 12, -1, -1, 16, 18, 20, 17, 19, 21]
    hp3d_eval_joints_smpl = [idx_h36m2smpl[each] for each in hp3d_eval_joints]

    # uv density j3d loss of 3dhp datasets
    uv_gt_hp3d, density_gt_hp3d = xyz2uv_density(labels['3dhp_j17'], labels['cam_intrinsics'])
    valid_hp3d = (uv_gt_hp3d > 0).all(dim=-1) * (uv_gt_hp3d[..., 0] < inp_w) * (uv_gt_hp3d[..., 1] < inp_h) * hp3d_idx.unsqueeze(-1)
    valid_hp3d = valid_hp3d[:, hp3d_eval_joints]

    uv_diff_hp3d = output['uv'][:, hp3d_eval_joints_smpl] - uv_gt_hp3d[:, hp3d_eval_joints]
    loss_uv_hp3d = mask_mean(torch.relu(uv_diff_hp3d.norm(dim=-1) - 5), valid_hp3d)
    err_uv_hp3d = mask_mean(uv_diff_hp3d.norm(dim=-1), valid_hp3d)

    loss_density_hp3d_root = 1.8 * mask_mean(
        torch.relu((output['density'][:, hp3d_eval_joints_smpl][:,:1]/h_pred - density_gt_hp3d[:, hp3d_eval_joints][:,:1]/h_gt).abs()-0.04/800*1000/1.8),
        hp3d_idx)
    err_density_hp3d_root = mask_mean(
        (output['density'][:, hp3d_eval_joints_smpl][:,:1] - density_gt_hp3d[:, hp3d_eval_joints][:,:1]).abs(), hp3d_idx)

    depth_gt_hp3d = labels['3dhp_j17'][:, hp3d_eval_joints][:, :, -1:]  # bs,9,1
    depth_gt_hp3d = 100 * (depth_gt_hp3d - labels['3dhp_j17'][:, :1, -1:])
    depth_pred_hp3d = 100 * (output['depth'][:, hp3d_eval_joints_smpl] - output['depth'][:, -1:])  # bs,9,1
    t = 0 * (depth_pred_hp3d / h_pred - depth_gt_hp3d / h_gt).mean(dim=1, keepdim=True)
    loss_depth_hp3d = 1.8 * mask_mean(
        torch.relu((depth_pred_hp3d/h_pred - depth_gt_hp3d/h_gt - t).abs() - 4/1.8) + 0.1 * t.abs(),
        valid_hp3d)
    err_depth_hp3d = mask_mean((depth_pred_hp3d - depth_gt_hp3d).abs(), valid_hp3d)
    err_depth_aligned_hp3d = mask_mean(normed_diff(depth_pred_hp3d,depth_gt_hp3d).abs(), valid_hp3d)

    pred_j17 = pred_j17 - pred_j17[:, :1]
    gt_j17_hp3d = (labels['3dhp_j17'] - labels['3dhp_j17'][:, :1])
    MPJPE_hp3d = mask_mean((pred_j17 - gt_j17_hp3d).norm(dim=-1), hp3d_idx)
    try:
        PA_MPJPE_hp3d = p_mpjpe(pred_j17.detach().cpu().double().numpy(), gt_j17_hp3d.detach().cpu().double().numpy())
        PA_MPJPE_hp3d = mask_mean(torch.tensor(PA_MPJPE_hp3d, device=pred_j17.device, dtype=pred_j17.dtype), hp3d_idx)
    except RuntimeError as e:
        PA_MPJPE_hp3d = 0.03
        print('PA_MPJPE_hp3d: ', e)

    # from common.visualize import array_show
    # files = [f'./render_res/{each:.3f}.jpg' for each in torch.randn(bs)]
    # [array_show(inp['img'][i], files[i]) for i in range(bs)]
    # model.smpl.render(output['beta'], output['theta'], output['transl'], files, labels['cam_intrinsics'])
    #
    # from common.visualize import draw_joints2D
    # [draw_joints2D(output['uv'][i], image=inp['img'][i].permute(1, 2, 0).cpu().numpy(),
    #                suffix=f'{files[i]}') for i in range(bs)]

    # #############coco#####################
    coco_eval_joints = [7, 8, 9, 10, 13, 14, 15, 16]
    idx_coco2smpl = [-1, -1, -1, -1, -1, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    coco_eval_joints_smpl = [idx_coco2smpl[each] for each in coco_eval_joints]

    # get coco keypoints annotation and calculate uv loss
    uv_coco_gt = labels['coco_kp17'][:, coco_eval_joints][:, :, :2]
    in_frame = (uv_coco_gt > 0).all(dim=-1) * (uv_coco_gt[..., 0] < inp_w) * (uv_coco_gt[..., 1] < inp_h)
    valid_coco = coco_idx.unsqueeze(-1) * in_frame * (labels['coco_kp17'][:, coco_eval_joints][:, :, -1] > 0)

    uv_coco_pred = output['uv'][:, coco_eval_joints_smpl]
    loss_uv_coco = mask_mean(torch.relu((uv_coco_pred - uv_coco_gt).norm(dim=-1) - 5), valid_coco)
    err_uv_coco = mask_mean((uv_coco_pred - uv_coco_gt).norm(dim=-1), valid_coco)

    err_h = mask_mean(100 * (h_gt - h_pred).abs(), hps_idx + hp3d_idx)  # percentage
    loss_h = err_h

    loss = [loss_uv, loss_uv_coco, loss_uv_hp3d,
            loss_depth, loss_depth_hp3d,
            loss_density_root, loss_density_hp3d_root,
            loss_xyz,
            loss_h,
            loss_twist, loss_beta, loss_leaf,
            ]
    return loss, [each.item() if torch.is_tensor(each) else each for each in
                       loss + [
                           err_uv, err_uv_coco, err_uv_hp3d,
                           err_depth, err_depth_aligned, err_depth_hp3d, err_depth_aligned_hp3d,
                           err_density_root, err_density_hp3d_root,
                           err_xyz, err_x, err_y, err_z, err_xyz_aligned,
                           err_h,
                           err_twist,
                           MPJPE, PA_MPJPE, MPVPE, MPJPE_hp3d, PA_MPJPE_hp3d,
                       ]]
