import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.smpl_wrapper import SMPL
from common.model.HRNet.HRNet_hybrik import get_hrnet
from common.geometry import rot6d_to_rotmat, batch_rodrigues


class PersPose(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.joint_num = 25  # 24*smpl_joint + 1*h36m_j17_pelvis
        self.flip_pair = torch.tensor([[1, 2], [4, 5], [7, 8], [10, 11],
                                       [13, 14], [16, 17],
                                       [18, 19], [20, 21], [22, 23],])
        self.flip_joint_idx = torch.arange(self.joint_num)
        for l, r in self.flip_pair:
            self.flip_joint_idx[l] = r
            self.flip_joint_idx[r] = l

        self.backbone = get_hrnet(48, num_joints=self.joint_num,
                                  depth_dim=64, generate_feat=True, generate_hm=True,
                                  init_weights=True, pretrained='./data/ckpt/pose_hrnet_w48_256x192.pth', )
        self.hm_layer = nn.Conv2d(in_channels=48, out_channels=self.joint_num, kernel_size=1)
        self.depth_layer_global = nn.Linear(2048, self.joint_num)
        # self.depth_layer = nn.Conv2d(in_channels=48, out_channels=self.joint_num, kernel_size=1)
        self.rho0_layer = nn.Linear(2048, 1)
        self.twist_layer = nn.Linear(2048, 3 * self.joint_num)  # ortho twist
        self.shape_layer = nn.Linear(2048, 11)  # beta
        self.leaf_rot_layer = nn.Linear(2048, 6*5)
        self.leaf_idxes = [22, 23, 10, 11, 15]

        self.cam_layer = nn.Conv2d(2, 256, kernel_size=1, bias=False)
        nn.init.normal_(self.cam_layer.weight, mean=0, std=0.01)

        self.smpl = SMPL(gender='neutral', batch_size=96, dtype=torch.float32, device='cpu', num_betas=11)
        for param in self.smpl.parameters():
            param.requires_grad = False

    def freeze(self, freeze=True):
        pass
        # for param in self.backbone.parameters():
        #     param.requires_grad = not freeze

    def inner(self, img, labels, flip=False):
        bs, _3, inp_h, inp_w = img.shape
        if flip:
            img = img.flip([-1])
            labels['cam_intrinsics'][:, 0, 2] = inp_w - labels['cam_intrinsics'][:, 0, 2]

        v_index, u_index = torch.meshgrid(torch.arange(inp_h // 4, device=img.device),
                                          torch.arange(inp_w // 4, device=img.device), indexing='ij')
        uv1 = torch.stack([(u_index+0.5)*4,(v_index+0.5)*4,v_index*0+1], -1)  # 64,64,3
        normed_uv = torch.inverse(labels['cam_intrinsics'])[:, None] @ uv1.reshape(1, -1, 3, 1)
        normed_uv = normed_uv[:, :, :2, 0].permute(0, 2, 1).reshape(-1, 2, inp_h // 4, inp_w // 4)
        valid_cam_intrinsics = [each != 'coco' for each in labels['dataset_name']]
        normed_uv = normed_uv * torch.tensor(valid_cam_intrinsics, device=img.device).reshape(-1, 1, 1, 1)

        cam_info = self.cam_layer(normed_uv)

        hrn_feat, global_feat = self.backbone(img, cam_info)  # bs,48,64,64  bs,2048
        hm = self.hm_layer(hrn_feat)  # bs,j_num,64,64
        hm = torch.softmax(hm.reshape(bs, self.joint_num, -1), dim=-1).reshape(hm.shape)
        v_index, u_index = torch.meshgrid(torch.arange(hm.shape[-2], device=hm.device),
                                          torch.arange(hm.shape[-1], device=hm.device), indexing='ij')
        v_index, u_index = (v_index + 0.5) * inp_h / hm.shape[-2], (u_index + 0.5) * inp_w / hm.shape[-1]
        uv = torch.stack([
            (hm * u_index[None, None]).sum([-1, -2]), (hm * v_index[None, None]).sum([-1, -2])
        ], dim=-1)  # bs,j_num,2

        # depth_map = self.depth_layer(hrn_feat).sigmoid()  # bs,j_num,64,64
        # depth = (hm * depth_map).sum(dim=[-1, -2]).unsqueeze(-1)  # bs,j_num,1
        depth = self.depth_layer_global(global_feat).sigmoid().unsqueeze(-1)
        depth = 1.1 * (depth - 0.5) * 2  # ∈[-1.1,1.1] bs,j_num,1
        depth = torch.cat([0 * depth[:, :1], depth[:, 1:]], dim=1)

        rho0 = self.rho0_layer(global_feat).sigmoid().unsqueeze(-1)
        rho0 = 1e-3 * 7 * torch.exp((rho0 - 0.5) * 2 * 1.5)  # 7/(2.73**1.5), 7*(2.73**1.5)  # 1800mm/256=7

        ortho_twist = self.twist_layer(global_feat).reshape(bs, self.joint_num, 3)
        beta = self.shape_layer(global_feat)  # bs,11
        leaf_rot = self.leaf_rot_layer(global_feat).reshape(bs, 5, 6)
        leaf_rot = rot6d_to_rotmat(leaf_rot).reshape(bs, 5, 9)

        f = (labels['cam_intrinsics'][:, 0, 0] + labels['cam_intrinsics'][:, 1, 1])[:, None, None] / 2

        if flip:
            labels['cam_intrinsics'][:, 0, 2] = inp_w - labels['cam_intrinsics'][:, 0, 2]

            depth = depth[:, self.flip_joint_idx]
            uv = uv[:, self.flip_joint_idx]
            uv = torch.cat([inp_w - uv[:, :, :1], uv[:, :, 1:]], dim=-1)
            ortho_twist = ortho_twist[:, self.flip_joint_idx]
            ortho_twist = torch.cat([-1*ortho_twist[:, :, :1], ortho_twist[:, :, 1:]], dim=-1)

            flip_x = torch.tensor([[-1., 0, 0], [0, 1, 0], [0, 0, 1]]).to(leaf_rot.device)[None]
            leaf_rot = (flip_x @ leaf_rot.reshape(-1, 3, 3) @ flip_x).reshape(bs, 5, 9)
            leaf_rot = leaf_rot[:, [1, 0, 3, 2, 4]]

        beta_init = torch.tensor([[-0.2, 0.59, 0.43, 0.74, 0.33, 0.1, -0.5, 0.0, 1.27, 0.49, 0]]).to('cuda')
        return depth, uv, rho0, f, ortho_twist, leaf_rot, beta # + beta_init

    def forward(self, x, labels):
        bs, _3, inp_h, inp_w = x['img'].shape

        if self.training:
            depth, uv, rho0, f, ortho_twist, leaf_rot, beta = self.inner(x['img'], labels, flip=(random.random() < 0.5))
        else:
            depth, uv, rho0, f, ortho_twist, leaf_rot, beta = self.inner(x['img'], labels, flip=False)
            if True:  # flip_test
                depth_, uv_, rho0_, f_, ortho_twist_, _, beta_ = self.inner(x['img'], labels, flip=True)
                depth, uv, rho0, f, ortho_twist, beta = (depth+depth_)/2, (uv+uv_)/2, (rho0+rho0_)/2, (f+f_)/2, (ortho_twist+ortho_twist_)/2, (beta+beta_)/2

        density = (depth / f) + rho0

        center_uv = labels['cam_intrinsics'][:, :2, 2]
        xy = (uv - center_uv[:, None]) * density  # bs,j_num,2
        xyz = torch.cat([xy-xy[:, :1], depth], dim=-1)

        xyz0 = torch.cat([xy[:, :1], f * density[:, :1]], dim=-1)

        with torch.cuda.amp.autocast(enabled=False):  # run without AMP
            leaf_rot_dict = {self.leaf_idxes[i]: leaf_rot[:, i].reshape(bs, 3, 3) for i in range(len(self.leaf_idxes))}
            beta, theta, transl = self.smpl.ik_w_ortho_twist(
                xyz[:, :24]+xyz0, ortho_twist[:, :24], beta, leaf_rot=leaf_rot_dict)

        output = {
            'uv': uv,  # bs,j_num,2
            'depth': depth,  # bs,j_num,1
            'density': 1000 * density,  # bs,j_num,1
            'xyz': xyz,  # bs,j_num,3
            'xyz0': xyz0,  # bs,1,3  used for projection
            'ortho_twist': ortho_twist,  # bs,j_num,3
            'leaf_rot': leaf_rot,  # (bs, 5, 9)

            'beta': beta,  # bs,11
            'theta': theta,  # bs,24, 3, 3
            'transl': transl,  # bs,3
        }
        return output
