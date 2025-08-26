# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry import *


def batch_pelvis_R_t(p1, p2, p3, p4, p5, p6):
    """minimize ||Rp1+t-p4||+||Rp2+t-p5||+||Rp3+t-p6||
    :param pi:bs,3
    :return: R:(bs,3,3), t:(bs,3,1)
    """

    # Centralization
    c1 = (p1 + p2 + p3) / 3.0
    c2 = (p4 + p5 + p6) / 3.0

    # Compute normals
    n1 = torch.cross(p2 - p1, p3 - p1, dim=-1)
    n2 = torch.cross(p5 - p4, p6 - p4, dim=-1)

    # Compute rotation matrix making the two triangles in the same plane
    r1 = cal_swing(n1, n2)

    def func(v1, v2):
        angle = cal_radian(v1, v2)
        return (((torch.cross(v1, v2) * n2).sum(dim=-1) > 0) * 2 - 1) * angle

    def cal_mean(angle_1, angle_2):
        """(-179+179)/2=180 or -180 """
        mean_val = (angle_1 + angle_2) / 2
        flip_mean = mean_val - ((mean_val > 0) * 2 - 1) * 3.141592653589793
        flip_flap = (angle_1 - angle_2).abs() > 3.141592653589793
        return mean_val * ~flip_flap + flip_mean * flip_flap

    # use n2 as twist axis ,and calculate the average twist angle
    angle1 = func(r1.matmul((p1 - c1).unsqueeze(-1)).squeeze(-1), p4 - c2)
    angle2 = func(r1.matmul((p2 - c1).unsqueeze(-1)).squeeze(-1), p5 - c2)
    angle3 = func(r1.matmul((p3 - c1).unsqueeze(-1)).squeeze(-1), p6 - c2)
    twist_angle = cal_mean(cal_mean(angle1, angle2), angle3)
    r2 = batch_rodrigues(F.normalize(n2, p=2, dim=-1) * twist_angle.unsqueeze(-1))

    R = r2.matmul(r1)

    # Compute translation  Rc1+t=c2
    t = c2.unsqueeze(-1) - R.matmul(c1.unsqueeze(-1))

    return R, t


def batch_orient_svd(c1, c2, c3, c1_aim, c2_aim, c3_aim):
    rest_mat = torch.cat([c3.unsqueeze(-1), c1.unsqueeze(-1), c2.unsqueeze(-1)], dim=2)
    target_mat = torch.cat([c3_aim.unsqueeze(-1), c1_aim.unsqueeze(-1), c2_aim.unsqueeze(-1)], dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    # U, _, V = torch.svd(S_non_zero)
    device = S_non_zero.device
    U, _, V = torch.svd(S_non_zero.cpu())
    U = U.to(device=device)
    V = V.to(device=device)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device, dtype=c1.dtype)

    # rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device, dtype=c1.dtype).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat, 0 * rot_mat[:, :, :1]


def batch_chest_rotation(p1, p2, p3, p4):
    """minimize ||Rp1-p3||+||Rp2+p4||
    :param pi:bs,3
    :return: R:(bs,3,3)
    """

    # Compute normals
    n1 = torch.cross(p1, p2, dim=-1)
    n2 = torch.cross(p3, p4, dim=-1)

    # Compute rotation matrix making the two segment in the same plane
    r1 = cal_swing(n1, n2)

    def func(v1, v2):
        angle = cal_radian(v1, v2)
        return (((torch.cross(v1, v2) * n2).sum(dim=-1) > 0) * 2 - 1) * angle

    def cal_mean(angle_1, angle_2):
        """(-179+179)/2=180 or -180 """
        mean_val = (angle_1 + angle_2) / 2
        flip_mean = mean_val - ((mean_val > 0) * 2 - 1) * 3.141592653589793
        flip_flap = (angle_1 - angle_2).abs() > 3.141592653589793
        return mean_val * ~flip_flap + flip_mean * flip_flap

    # use n2 as twist axis ,and calculate the average twist angle
    twist_angle = cal_mean(func(r1.matmul(p1.unsqueeze(-1)).squeeze(-1), p3),
                           func(r1.matmul(p2.unsqueeze(-1)).squeeze(-1), p4))
    r2 = batch_rodrigues(F.normalize(n2, p=2, dim=-1) * twist_angle.unsqueeze(-1))
    return r2.matmul(r1)


def batch_hybrIK(aim_pose, rest_pose, xy_priority=False, big_diff=True):
    """ An Explicit inverse kinematics algorithm proposed in HybrIK
    Parameters
    ----------
    aim_pose : torch.tensor BxNx3
        Estimated pose or skeleton.
    rest_pose : torch.tensor BxNx3
        Template skeleton.
    Returns
    -------
    result_pose: torch.tensor BxNx3
        final joint position
    rel_result_pose: torch.tensor BxNx3
        The direction of each bone
    root_R: torch.tensor Bx3x3
        root_orientation
    root_t: torch.tensor Bx3x1
        root_position  # in SMPL root_position!=root_translation
    chest_global_orientation: torch.tensor Bx3x3
        chest orientation change with root orientation
    """
    batch_size = aim_pose.shape[0]
    dtype = aim_pose.dtype
    device = aim_pose.device
    joint_num = aim_pose.shape[-2]

    parents = torch.tensor(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
         15, 22, 23, 10, 11], device=device)
    if joint_num == 24:
        parents = parents[:24]

    # make root as center
    aim_pose_root = aim_pose[:, :1]
    rest_pose_root = rest_pose[:, :1]
    aim_pose = aim_pose - aim_pose_root
    rest_pose = rest_pose - rest_pose_root

    # calculate relative joint i.e. bone vector
    rel_rest_pose = rest_pose[:, 1:] - rest_pose[:, parents[1:]]
    rel_rest_pose = torch.cat([rest_pose[:, :1], rel_rest_pose], dim=-2)
    rel_aim_pose = aim_pose[:, 1:] - aim_pose[:, parents[1:]]
    rel_aim_pose = torch.cat([aim_pose[:, :1], rel_aim_pose], dim=-2)

    # rotate the T pose i.e. perform hybrik
    result_pose_list = []
    for joint_i in range(joint_num):
        if joint_i in [1, 2, 3, 13, 14]:
            # the position of joints 0,1,2,3 or 12,13,14 is determined in the same iteration.
            continue
        if joint_i == 0:
            # make children of pelvis get close to aim
            c1, c2, c3 = rest_pose[:, 1], rest_pose[:, 2], rest_pose[:, 3]
            c1_aim, c2_aim, c3_aim = aim_pose[:, 1], aim_pose[:, 2], aim_pose[:, 3]
            root_R, root_t = batch_pelvis_R_t(c1, c2, c3, c1_aim, c2_aim, c3_aim)
            # root_R, root_t = batch_orient_svd(c1, c2, c3, c1_aim, c2_aim, c3_aim)
            result_c1 = (root_R.matmul(c1.unsqueeze(-1)) + root_t).squeeze(-1)
            result_c2 = (root_R.matmul(c2.unsqueeze(-1)) + root_t).squeeze(-1)
            result_c3 = (root_R.matmul(c3.unsqueeze(-1)) + root_t).squeeze(-1)
            result_pose_list += [root_t.squeeze(-1), result_c1, result_c2, result_c3]
        elif joint_i == 12:
            # make collarbone get close to aim
            c_m, c_l, c_r = rel_rest_pose[:, 12], rel_rest_pose[:, 13], rel_rest_pose[:, 14]
            c_m_aim, c_l_aim, c_r_aim = (aim_pose[:, 12] - result_pose_list[9],
                                         aim_pose[:, 13] - result_pose_list[9],
                                         aim_pose[:, 14] - result_pose_list[9])
            # chest_global_orientation = batch_chest_rotation(c_l, c_r, c_l_aim, c_r_aim)
            chest_global_orientation = batch_orient_svd(c_m, c_l, c_r, c_m_aim, c_l_aim, c_r_aim)[0]
            c_m_result = result_pose_list[9] + chest_global_orientation.matmul(c_m.unsqueeze(-1)).squeeze(-1)
            c_l_result = result_pose_list[9] + chest_global_orientation.matmul(c_l.unsqueeze(-1)).squeeze(-1)
            c_r_result = result_pose_list[9] + chest_global_orientation.matmul(c_r.unsqueeze(-1)).squeeze(-1)
            result_pose_list += [c_m_result, c_l_result, c_r_result]
        else:
            direction = aim_pose[:, joint_i] - result_pose_list[parents[joint_i]]
            direction_normed = F.normalize(direction, p=2, dim=-1)  # bs,3
            bone_len = rel_rest_pose[:, joint_i].norm(dim=-1).unsqueeze(-1)

            result_rel = bone_len * direction_normed
            if xy_priority:
                delta_z = torch.relu(bone_len**2-(direction[:,:2]**2).sum(dim=-1,keepdim=True)).sqrt()
                result_rel_ = torch.cat([direction[:, :2], direction[:, -1:].sign() * delta_z], dim=-1)
                result_rel = (delta_z > 0) * result_rel_ + (delta_z <= 0) * result_rel

            if big_diff:
                # if the result bone direction is far different from the estimated pose, use the bone direction of the estimated pose.
                result_rel_v2 = bone_len * F.normalize(rel_aim_pose[:, joint_i], p=2, dim=-1)
                big_diff_idx = torch.where(((result_rel_v2 - result_rel).norm(dim=-1) >= 0.015))[0]
                result_rel[big_diff_idx] = result_rel_v2[big_diff_idx]

            result_position = result_pose_list[parents[joint_i]] + result_rel

            result_pose_list.append(result_position)

    result_pose = torch.stack(result_pose_list, dim=1)
    result_pose = result_pose + aim_pose_root
    rel_result_pose = result_pose[:, 1:] - result_pose[:, parents[1:]]
    rel_result_pose = torch.cat([result_pose[:, :1], rel_result_pose], dim=-2)
    return result_pose, rel_result_pose, root_R, chest_global_orientation


if __name__ == '__main__':

    def test_global_transform():
        # test the function global_transform
        def draw(point1, point2, point3, p4, p5, p6):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            # 创建3D图形对象
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 将p1, p2, p3定义为第一个三角形的顶点，p4, p5, p6定义为第二个三角形的顶点
            # 绘制三角形的三个顶点
            color1 = 'red'
            color2 = 'green'
            color3 = 'blue'

            ax.scatter(point1[0] * 0, point1[1] * 0, point1[2] * 0)
            ax.scatter(point1[0], point1[1], point1[2], c=color1)
            ax.scatter(point2[0], point2[1], point2[2], c=color2)
            ax.scatter(point3[0], point3[1], point3[2], c=color3)

            # 绘制三角形的边
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])
            ax.plot([point1[0], point3[0]], [point1[1], point3[1]], [point1[2], point3[2]])
            ax.plot([point2[0], point3[0]], [point2[1], point3[1]], [point2[2], point3[2]])

            # 绘制第二个三角形的三个顶点
            ax.scatter(p4[0], p4[1], p4[2], c=color1)
            ax.scatter(p5[0], p5[1], p5[2], c=color2)
            ax.scatter(p6[0], p6[1], p6[2], c=color3)

            # 绘制第二个三角形的边
            ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]])
            ax.plot([p4[0], p6[0]], [p4[1], p6[1]], [p4[2], p6[2]])
            ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]])

            plt.show()

        bs = 2
        p1 = torch.randn(bs, 3)
        p2 = torch.randn(bs, 3)
        p3 = torch.randn(bs, 3)
        p4 = torch.randn(bs, 3)
        p5 = torch.randn(bs, 3)
        p6 = torch.randn(bs, 3)
        # Compute initial objective function
        initial_obj = torch.norm(p1 - p4, dim=-1) + torch.norm(p2 - p5, dim=-1) + torch.norm(p3 - p6, dim=-1)

        # Compute initial distance between c1 and c2
        c1 = (p1 + p2 + p3) / 3.0
        c2 = (p4 + p5 + p6) / 3.0
        initial_distance = torch.norm(c1 - c2, dim=-1)

        # Compute transformation
        R, t = batch_pelvis_R_t(p1, p2, p3, p4, p5, p6)

        # Compute transformed points
        p1_transformed = (R.matmul(p1[:, :, None]) + t)[:, :, 0]
        p2_transformed = (R.matmul(p2[:, :, None]) + t)[:, :, 0]
        p3_transformed = (R.matmul(p3[:, :, None]) + t)[:, :, 0]

        # Compute final objective function
        final_obj = (torch.norm(p1_transformed - p4, dim=-1) +
                     torch.norm(p2_transformed - p5, dim=-1) +
                     torch.norm(p3_transformed - p6, dim=-1))
        for i in range(bs):
            draw(p1[i], p2[i], p3[i], p4[i], p5[i], p6[i])
            draw(p1_transformed[i], p2_transformed[i], p3_transformed[i], p4[i], p5[i], p6[i])

        print("Initial Objective Function: ", initial_obj)
        print("Final Objective Function: ", final_obj)
        print("Initial Distance between c1 and c2: ", initial_distance)


    def test_chest_rotation():
        # test the function global_transform
        def draw(point1, point2, p4, p5):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            # 创建3D图形对象
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 将p1, p2, p3定义为第一个三角形的顶点，p4, p5, p6定义为第二个三角形的顶点
            # 绘制三角形的三个顶点
            ax.scatter(point1[0] * 0, point1[1] * 0, point1[2] * 0)
            ax.scatter(point1[0], point1[1], point1[2])
            ax.scatter(point2[0], point2[1], point2[2])

            # 绘制三角形的边
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

            # 绘制第二个三角形的三个顶点
            ax.scatter(p4[0], p4[1], p4[2])
            ax.scatter(p5[0], p5[1], p5[2])

            # 绘制第二个三角形的边
            ax.plot([p4[0], p5[0]], [p4[1], p5[1]], [p4[2], p5[2]])

            plt.show()

        bs = 2
        p1 = torch.randn(bs, 3)
        p2 = torch.randn(bs, 3)
        p3 = torch.randn(bs, 3)
        p4 = torch.randn(bs, 3)

        # Compute transformation
        R = batch_chest_rotation(p1, p2, p3, p4)

        # Compute transformed points
        p1_transformed = R.matmul(p1[:, :, None])[:, :, 0]
        p2_transformed = R.matmul(p2[:, :, None])[:, :, 0]

        for i in range(bs):
            draw(p1[i], p2[i], p3[i], p4[i])
            draw(p1_transformed[i], p2_transformed[i], p3[i], p4[i], )


    # pose_skeleton2 = torch.load(r"D:\tmp\pose_skeleton2", map_location='cpu')
    # rest_pose2 = torch.load(r"D:\tmp\rest_J2", map_location='cpu')
    #
    # skeleton2mesh(pose_skeleton2, rest_pose2)

    test_global_transform()
    test_chest_rotation()
