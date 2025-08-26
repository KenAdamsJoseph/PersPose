import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_v2 import SwinTransformerV2
from .swin_v1 import SwinTransformer


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=[1, 2, 4]):
        super().__init__()
        self.layers = nn.ModuleList()
        for o_size in pool_scales:
            self.layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=o_size),
                ConvBNReLU(in_channels, out_channels, kernel_size=1),
            ))
        self.bottleneck = ConvBNReLU(
            out_channels * len(pool_scales) + in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        pyramid = [each(x) for each in self.layers]
        f_ = [F.interpolate(each, size=x.shape[-2:], mode='bilinear', align_corners=False) for each in pyramid]
        ppm_outs = torch.cat([x, *f_], dim=1)
        return self.bottleneck(ppm_outs)


class UperHead(nn.Module):
    def __init__(self, backbone_dim_list=[96, 192, 384, 768], out_channels=512, **kwargs):
        super().__init__()
        self.dim_list = backbone_dim_list
        self.d = out_channels
        self.n_level = len(self.dim_list)

        self.lateral_convs = nn.ModuleList()
        for i in range(self.n_level - 1):
            self.lateral_convs.append(ConvBNReLU(self.dim_list[i], self.d, kernel_size=1))
        self.lateral_convs.append(PPM(self.dim_list[-1], self.d))

        self.fpn_convs = nn.ModuleList()
        for i in range(self.n_level - 1):
            self.fpn_convs.append(ConvBNReLU(self.d, self.d, kernel_size=3, padding=1))
        self.fpn_convs.append(nn.Identity())
        self.fpn_bottleneck = ConvBNReLU(self.d * self.n_level, self.d, kernel_size=3, padding=1)

    def forward(self, x_list):
        laterals = [each(x_list[i]) for i, each in enumerate(self.lateral_convs)]

        # fpn top-down path
        for i in range(self.n_level - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(self.n_level)]

        for i in range(1, self.n_level):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return self.fpn_bottleneck(fpn_outs)


class UpernetSwin(nn.Module):
    def __init__(self, name='swin_v2_base_window16', img_size=(384, 128), drop_path_rate=0.2, out_channel=48):
        super(UpernetSwin, self).__init__()
        swin_cfg = {
            'swin_v1_large': dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
            'swin_v1_base': dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
            'swin_v1_small': dict(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
            'swin_v1_tiny': dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
            'swin_v2_base_window16': dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                          window_size=16, ),  # img_size=(256, 256)
            'swin_v2_base_window8': dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                         window_size=8, ),  # img_size=(256, 256)
            'swin_v2_large': dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                                  window_size=16, ),  # img_size=(256, 256)
        }
        swin_ckpt = {
            'swin_v2_base_window16': r"./data/ckpt/swin/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth",
            'swin_v2_large_window16': r"./data/ckpt/swin/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth",
            'swin_v2_base_window8': r"./data/ckpt/swin/swinv2_base_patch4_window8_256.pth",
        }[name]
        self.swin = SwinTransformerV2(img_size=img_size, drop_path_rate=drop_path_rate, **swin_cfg[name])

        ckpt = torch.load(swin_ckpt, map_location='cpu')
        if 'model' in ckpt.keys():
            ckpt = ckpt['model']
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = {k[7:]: v for k, v in ckpt.items()}
        if list(ckpt.keys())[0].startswith('encoder.'):
            ckpt_new = {}
            for k, v in ckpt.items():
                if k.startswith('encoder.'):
                    ckpt_new[k[8:]] = v
            ckpt = ckpt_new
        rpe_mlp_keys = [k for k in ckpt.keys() if "rpe_mlp" in k]
        for k in rpe_mlp_keys:
            ckpt[k.replace('rpe_mlp', 'cpb_mlp')] = ckpt.pop(k)

        to_remove = [k for k in ckpt.keys() if
                     "relative_coords_table" in k or 'relative_position_index' in k or '.attn_mask' in k]
        for k in to_remove:
            ckpt.pop(k)

        print(self.swin.load_state_dict(ckpt, strict=False))

        dim = swin_cfg[name]['embed_dim']
        self.head = UperHead(backbone_dim_list=[dim, dim * 2, dim * 4, dim * 8], out_channels=out_channel)

    def forward(self, x):
        f_list = self.swin(x)
        return self.head(f_list)

# m = UpernetSwin()
# f = m(torch.randn(2, 3, 384, 128))
# print()
