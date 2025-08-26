# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np


# import quaternion
# https://quaternion.readthedocs.io/en/latest/Package%20API%3A/quaternion/#rotate_vectors
# https://zhuanlan.zhihu.com/p/608482354
# q1 = np.quaternion(1,2,3,4)
# q2 = quaternion.from_float_array([1,2,3,4])
# q3 = quaternion.from_rotation_matrix([[1,2,3],[1,2,3],[1,2,3]])
# q4 = quaternion.from_euler_angles([1,2,3])

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # quaternion.rotate_vectors(cam_quat, smpl_tran)  #用的是矩阵。。
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def axisang2quat(axisang):
    # quaternion.from_rotation_vector()
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return quat


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def mat2quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    this function is borrowed from pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
           ].reshape(batch_dim + (4,))


# import cv2
# def mat2quat_opencv(mat):
#     return axisang2quat(torch.tensor(cv2.Rodrigues(mat)[0][:, 0])[None])
#     # return quaternion.from_rotation_matrix(mat)


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    quat = axisang2quat(axisang)
    rot_mat = quat2mat(quat)
    return rot_mat


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        # >>> input = torch.rand(4, 3, 4)  # Nx3x4
        #  >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def rot6d_to_rotmat_spin(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(dim=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def cal_radian(v1, v2):
    """
    Args:
        v1: bs,3
        v2: bs,3
    Returns:
        Angle between vectors.
    """
    dot = (v1 * v2).sum(dim=-1)
    cos_value = dot / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
    eps = 1e-7
    cos_value = (cos_value >= 1 - eps) * (1 - eps) + (cos_value < 1 - eps) * cos_value
    cos_value = (cos_value <= -1 + eps) * (-1 + eps) + (cos_value > -1 + eps) * cos_value
    radian = torch.acos(cos_value)
    return radian


def cal_swing(v1, v2, mat=True):
    """ swing_R.matmul(V1_normalized) == V2_normalized
    :param v1: bs,3,
    :param v2: bs,3
    :return: swing rotation matrix R
    """
    axis = v1.cross(v2, dim=-1)  # -1,3
    if torch.all(axis.norm(dim=-1) == 0):
        angle = v1[:, 0] * 0
    else:
        angle = cal_radian(v1, v2)
    # print(angle[0].item()/3.14*180)
    axis_angle = F.normalize(axis, p=2, dim=-1) * angle.unsqueeze(-1)  # -1,3
    if mat:
        return batch_rodrigues(axis_angle)
    else:
        return axis_angle


def rotation2swing_twist(twist_axis, rot):
    """
    rot: (B, 3, 3) rotation to be decomposed
    twist_axis: (B, 3) twist axis
    return: swing matrix and twist angle
    """
    twist_axis = F.normalize(twist_axis, p=2, dim=-1)
    twist_axis_ = rot.matmul(twist_axis[..., None]).squeeze(dim=-1)
    # swing twist_axis to twist_axis_
    swing_axis = torch.cross(twist_axis, twist_axis_, dim=-1)
    dot_res = (twist_axis * twist_axis_).sum(dim=-1)
    # arccos is not differentiable at -1 and 1
    eps = 1e-7
    dot_res = ((dot_res > 1 - eps) * (1 - eps) + (dot_res < -1 + eps) * (-1 + eps) +
               (dot_res <= 1 - eps) * (dot_res >= -1 + eps) * dot_res)
    swing_ang = torch.arccos(dot_res)
    swing_axisang = swing_ang[:, None] * F.normalize(swing_axis, p=2, dim=-1)
    swing = batch_rodrigues(swing_axisang).reshape(-1, 3, 3)
    # calculate the remain twist
    twist = rot.matmul(swing.transpose(-1, -2))
    twist_axisang = quaternion_to_angle_axis(mat2quat(twist))
    # twist_axisang should be on same direction with twist_axis_
    tmp = (((twist_axis_ * F.normalize(twist_axisang, p=2, dim=-1)).sum(dim=-1) > 0) * 2 - 1)
    twist_radian = tmp * twist_axisang.norm(dim=-1)
    return twist_radian, swing


def adjust_perspective(cam_intrinsic, obj_center, aim_center, rot_angle=0):
    """move  position by adjusting perspective
    :param cam_intrinsic:np.array(3,3) fx fy cx cy are used
    :param obj_center: object's center [cx,cy]
    :param aim_center:  aim pixel position
    :param rot_angle: rotation augmentation
    :return: 3d rotation, 2d transformation
    """
    original_axis_vector = np.linalg.inv(cam_intrinsic) @ np.array([obj_center[0], obj_center[1], 1])
    new_axis_vector = np.linalg.inv(cam_intrinsic) @ np.array([aim_center[0], aim_center[1], 1])
    original_axis_vector = torch.tensor(original_axis_vector / np.linalg.norm(original_axis_vector))
    new_axis_vector = torch.tensor(new_axis_vector / np.linalg.norm(new_axis_vector))

    # calculate 3d rotation of object in camera
    R_swing = cal_swing(original_axis_vector.reshape(1, 3), new_axis_vector.reshape(1, 3))[0]
    R_twist_axisang = rot_angle / 180 * torch.pi * new_axis_vector
    R_twist = batch_rodrigues(R_twist_axisang.reshape(1, 3))[0]
    R = R_twist.matmul(R_swing).numpy()
    perspective_k = cam_intrinsic @ R @ np.linalg.inv(cam_intrinsic)  # (3,3)
    return R, perspective_k


def point_in_triangles(triangles, points):
    """
    triangles: Tensor of shape (bs, n_tri, 3, 2) representing triangles.
    points: Tensor of shape (bs,n_points, 2) representing points.
    """
    bs, n_tri = triangles.shape[:2]
    n_points = points.shape[1]

    # triangles_expanded = triangles.unsqueeze(2).expand(bs, n_tri, n_points, 3, 2)  # bs, 13776, 2048, 3, 2
    # points_expanded = points.unsqueeze(0).unsqueeze(0).expand(bs, n_tri, n_points, 2).unsqueeze(3)  # bs, 13776, 2048, 1, 2
    triangles_expanded = triangles.unsqueeze(2)  # bs, n_tri, 1, 3, 2
    points_expanded = points.unsqueeze(-2).unsqueeze(1)  # bs,1,n_points,1,2
    edge_vecs = triangles_expanded[:, :, :, [1, 2, 0]] - triangles_expanded  # bs, n_tri, 1, 3, 2
    point_vecs = points_expanded - triangles_expanded  # bs, n_tri, n_points, 3, 2
    cross_products = (edge_vecs[:, :, :, :, 0] * point_vecs[:, :, :, :, 1] -
                      edge_vecs[:, :, :, :, 1] * point_vecs[:, :, :, :, 0])  # bs,n_tri, n_points, 3
    sign_check = cross_products.sign()
    res = (sign_check[:, :, :, 0] == sign_check[:, :, :, 1]) * (sign_check[:, :, :, 0] == sign_check[:, :, :, 2])

    return res  # bs, n_tri,n_points
