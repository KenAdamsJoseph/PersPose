import torch


# def heatmap2uv_(hm, heatmap_hw=(64, 64), image_hw=(256, 256)):
#     """ get pixel coordinate
#     :param hm: (bs, j_num, hm_h, hm_w)  heatmap tensor
#     :return: (bs,j_num,u,v)  u∈(0.5,hm_w-0.5)
#     """
#     bs, j_num, hm_h, hm_w = hm.shape
#     max_val, max_idx = hm.reshape(bs, j_num, -1).max(dim=-1)
#     u, v = max_idx % hm_w, max_idx // hm_w
#     u_new, v_new = u.clone().float(), v.clone().float()
#     for sample_idx in range(bs):
#         for joint_idx in range(j_num):
#             hm_each = hm[sample_idx][joint_idx]
#             u_each = u[sample_idx][joint_idx]
#             v_each = v[sample_idx][joint_idx]
#             if u_each - 1 >= 0 and u_each + 1 <= hm_w - 1:
#                 # when hm[v,u+1]==h[v,u-1], u_decimal=0;
#                 # when hm[v,u+1]-hm[v,u-1]==2*(hm[v,u]-(hm[v,u+1]+hm[v,u-1])/2), u_decimal=0.5
#                 u_plus1 = hm_each[v_each][u_each + 1]
#                 u_minus1 = hm_each[v_each][u_each - 1]
#                 u_decimal = 0.25 * (u_plus1 - u_minus1) / (hm_each[v_each][u_each] - (u_plus1 + u_minus1) / 2 + 1e-7)
#                 u_new[sample_idx][joint_idx] += u_decimal
#             if v_each - 1 >= 0 and v_each + 1 <= hm_h - 1:
#                 v_plus1 = hm_each[v_each + 1][u_each]
#                 v_minus1 = hm_each[v_each - 1][u_each]
#                 v_decimal = 0.25 * (v_plus1 - v_minus1) / (hm_each[v_each][u_each] - (v_plus1 + v_minus1) / 2 + 1e-7)
#                 v_new[sample_idx][joint_idx] += v_decimal
#     uv = torch.stack([u_new, v_new], dim=-1)  # bs,j_num,2
#     uv = uv + 0.5  # 0.5-63.5
#     uv[:, :, 0] *= image_hw[1] / heatmap_hw[1]
#     uv[:, :, 1] *= image_hw[0] / heatmap_hw[0]
#     return uv  # 2-254
# def uv2heatmap_(uv, heatmap_hw=(64, 64), image_hw=(256, 256), sigma=2):
#     """
#     :param uv: bs,j_num,2 pixel coordinate  #dtype= torch.float32, uv are continuous, u∈[0,image_wh[0]]
#     :param heatmap_hw:
#     :param image_hw:
#     :param sigma:
#     :return: bs,j_num,hm_h,hm_w
#     """
#     bs, num_joints = uv.shape[:2]
#     tmp_size = sigma * 3
#     feat_stride = [image_hw[0] / heatmap_hw[0], image_hw[1] / heatmap_hw[1]]
#     target = torch.zeros(bs, num_joints, heatmap_hw[0], heatmap_hw[1], dtype=uv.dtype, device=uv.device)
#
#     uv_scaled = uv.clone()
#     uv_scaled[..., 0] /= feat_stride[1]
#     uv_scaled[..., 1] /= feat_stride[0]
#     for sample_idx in range(bs):
#         for joint_id in range(num_joints):
#             mu_x, mu_y = uv_scaled[sample_idx][joint_id]
#             hm_uv_ul = [int(mu_x) - tmp_size, int(mu_y) - tmp_size]
#             hm_uv_br = [int(mu_x) + tmp_size + 1, int(mu_y) + tmp_size + 1]
#
#             # calculate probability density
#             hm_x = torch.arange(hm_uv_ul[0], hm_uv_br[0], 1, dtype=uv.dtype, device=uv.device) + 0.5
#             hm_y = torch.arange(hm_uv_ul[1], hm_uv_br[1], 1, dtype=uv.dtype, device=uv.device) + 0.5
#             x_grid, y_grid = torch.meshgrid(hm_x, hm_y, indexing='xy')
#             # xy = torch.stack((x_grid, y_grid), dim=-1)
#             g = torch.exp(- ((x_grid - mu_x) ** 2 + (y_grid - mu_y) ** 2) / (2 * sigma ** 2))  # 13,13
#
#             #
#             hm_u_min, hm_u_max = min(max(0, hm_uv_ul[0]), heatmap_hw[1]), max(0, min(hm_uv_br[0], heatmap_hw[1]))
#             hm_v_min, hm_v_max = min(max(0, hm_uv_ul[1]), heatmap_hw[1]), max(0, min(hm_uv_br[1], heatmap_hw[0]))
#             g_u_min = max(0, hm_u_min - hm_uv_ul[0])
#             g_u_max = g_u_min + hm_u_max - hm_u_min
#             g_v_min = max(0, hm_v_min - hm_uv_ul[1])
#             g_v_max = g_v_min + hm_v_max - hm_v_min
#             target[sample_idx, joint_id, hm_v_min:hm_v_max, hm_u_min:hm_u_max] = g[g_v_min:g_v_max, g_u_min:g_u_max]
#
#     return target


def heatmap2uv(hm, heatmap_hw=(64, 64), image_hw=(256, 256)):
    bs, j_num, hm_h, hm_w = hm.shape
    max_val, max_idx = hm.view(bs, j_num, -1).max(dim=-1)
    u, v = max_idx % hm_w, max_idx // hm_w
    u_new, v_new = u.float(), v.float()

    mask_u = ((u - 1 >= 0) & (u + 1 <= hm_w - 1))
    mask_v = ((v >= 1) & (v <= hm_h - 2))
    u_plus1 = hm[torch.arange(bs)[:, None], torch.arange(j_num), v, (u + 1) * mask_u]  # to prevent Index Out of Range
    u_minus1 = hm[torch.arange(bs)[:, None], torch.arange(j_num), v, (u - 1) * mask_u]
    v_plus1 = hm[torch.arange(bs)[:, None], torch.arange(j_num), (v + 1) * mask_v, u]
    v_minus1 = hm[torch.arange(bs)[:, None], torch.arange(j_num), (v - 1) * mask_v, u]
    u_decimal = 0.25 * (u_plus1 - u_minus1) / (
            hm[torch.arange(bs)[:, None], torch.arange(j_num), v, u] - (u_plus1 + u_minus1) / 2 + 1e-7)
    v_decimal = 0.25 * (v_plus1 - v_minus1) / (
            hm[torch.arange(bs)[:, None], torch.arange(j_num), v, u] - (v_plus1 + v_minus1) / 2 + 1e-7)

    u_new += u_decimal * mask_u
    v_new += v_decimal * mask_v
    uv = torch.stack([u_new, v_new], dim=-1)  # bs,j_num,2
    uv = uv + 0.5  # 0.5-63.5
    uv *= torch.tensor([image_hw[1] / heatmap_hw[1], image_hw[0] / heatmap_hw[0]], device=uv.device)[None, None, :]
    return uv  # 2-254


def uv2heatmap(uv, heatmap_hw=(64, 64), image_hw=(256, 256), sigma=2):
    bs, num_joints = uv.shape[:2]
    tmp_size = sigma * 3
    feat_stride = torch.tensor([image_hw[0] / heatmap_hw[0], image_hw[1] / heatmap_hw[1]], device=uv.device)
    target = torch.zeros(bs, num_joints, heatmap_hw[0], heatmap_hw[1], dtype=uv.dtype, device=uv.device)

    # generate gauss grid
    uv_scaled = uv / feat_stride[None, None, :]
    hm_x = 0.5 + torch.arange(heatmap_hw[1], dtype=uv.dtype, device=uv.device).view(1, 1, 1, -1)
    hm_y = 0.5 + torch.arange(heatmap_hw[0], dtype=uv.dtype, device=uv.device).view(1, 1, -1, 1)
    g = torch.exp(- (
            (hm_x - uv_scaled[..., 0, None, None]) ** 2 + (hm_y - uv_scaled[..., 1, None, None]) ** 2
    ) / (2 * sigma ** 2))

    # calculate area
    hm_uv_ul = (uv_scaled - tmp_size).floor().long()
    hm_uv_br = (uv_scaled + tmp_size + 1).floor().long()
    hm_u_min = torch.clamp(hm_uv_ul[..., 0, None, None], 0, heatmap_hw[1])
    hm_u_max = torch.clamp(hm_uv_br[..., 0, None, None], 0, heatmap_hw[1])
    hm_v_min = torch.clamp(hm_uv_ul[..., 1, None, None], 0, heatmap_hw[0])
    hm_v_max = torch.clamp(hm_uv_br[..., 1, None, None], 0, heatmap_hw[0])

    # fill heatmap
    v_index, u_index = torch.meshgrid(torch.arange(heatmap_hw[0], device=uv.device),
                                      torch.arange(heatmap_hw[1], device=uv.device), indexing='ij')
    v_index = v_index[None, None, :, :].expand(bs, num_joints, -1, -1)
    u_index = u_index[None, None, :, :].expand(bs, num_joints, -1, -1)
    mask = (v_index >= hm_v_min) & (v_index < hm_v_max) & (u_index >= hm_u_min) & (u_index < hm_u_max)

    target[mask] = g[mask]
    return target


if __name__ == '__main__':
    uv_inp = torch.rand(32, 29, 2) * 200 + 20  # 20-220
    hm = uv2heatmap(uv_inp)
    uv_out = heatmap2uv(hm)
    print('reconstruct uv pixel distance ', (uv_inp - uv_out).norm(dim=-1).max())

    # hm_ = uv2heatmap_(uv_inp)
    # uv_out_ = heatmap2uv_(hm_)
    # print('reconstruct uv pixel distance ', (uv_inp - uv_out_).norm(dim=-1).max())
