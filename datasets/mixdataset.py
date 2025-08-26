import os
import sys
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.transform import Transform
from common import bbox


class MixDataset(Dataset):
    def __init__(self, datasets, args=None, train=True, ratio=None, n=None):
        """
        :param datasets: list of sub-dataset
        :param ratio: list, Proportion for each sub-dataset
        :param n: len of mix dataset
        """
        self.train = train
        self.datasets = datasets
        self.dataset_size = [len(x) for x in self.datasets]
        self.n = n if n is not None else sum(self.dataset_size)
        if ratio is not None:
            self.ratio = [x / sum(ratio) for x in ratio]
        else:
            self.ratio = [x / sum(self.dataset_size) for x in self.dataset_size]
        self.cumulative_r = [0]
        for val in self.ratio:
            self.cumulative_r += [self.cumulative_r[-1] + val]

        self.epoch = -1  # which is necessary to traverse all samples

        rand_gen = torch.Generator()
        rand_gen.manual_seed(2147483647)
        self.fixed_shuffle = [(
                                      torch.randperm(each_size * 50, generator=rand_gen) % each_size
                              ).tolist() for each_size in self.dataset_size]

        self.transform = Transform(args)

        self.data_domain = set(list((
            'dataset_name', 'id', 'file_name', 'image_wh', 'bbox',
            'mask', 'mask_valid', 'bbox_mask',
            'dp_uv', 'dp_face', 'dp_bc', 'dp_valid', 'dp_weight',
            'smpl_theta', 'smpl_beta', 'smpl_transl', 'gender', 'cam_intrinsics', 'cam_intrinsics_origin',
            'smpl_j24','smpl_j17','smpl_vertices', 'smpl_ortho_twist', 'height',
            'coco_kp17', '3dhp_j17',
            'rotation_3d',
        )))
        self.default_val = {
            'mask': np.zeros((256,256)).astype(np.uint8),
            'mask_valid': False,
            'bbox_mask': np.zeros(4,),

            'dp_uv': np.zeros((6890, 2)),
            'dp_face': np.zeros((6890, 3), dtype=np.int32),
            'dp_bc': np.ones((6890, 3)) / 3,
            'dp_valid': np.zeros((6890,), dtype=bool),
            'dp_weight': np.ones((6890, 3)),

            'smpl_theta': np.zeros((24, 3)),
            'smpl_beta': np.zeros(11),
            'smpl_transl': np.ones(3),
            'gender': 'neutral',
            'cam_intrinsics': np.array([[600, 0, 128], [0, 600, 128], [0, 0, 1.]]),
            'cam_intrinsics_origin': np.array([[600, 0, 128], [0, 600, 128], [0, 0, 1.]]),

            'smpl_j24': np.ones((24, 3)),
            'smpl_j17': np.random.rand(17, 3)+10,
            'smpl_vertices': np.ones((6890, 3)),
            'smpl_ortho_twist': np.ones((24, 3)),
            'height': np.ones(1)*1.8,

            'coco_kp17': np.ones((17, 3)),
            '3dhp_j17': np.random.rand(17, 3)+10,
            'rotation_3d': np.eye(3),
        }

    def __len__(self):
        return self.n

    def get_sample_idx(self, idx):
        assert self.epoch != -1
        p = idx / self.n  # p∈[0,1]  # position in mix dataset
        dataset_idx = bisect.bisect_right(self.cumulative_r, p) - 1
        p_ = (p - self.cumulative_r[dataset_idx]) / self.ratio[dataset_idx]  # p_∈[0,1] # position in sub dataset
        sample_idx = round(self.ratio[dataset_idx] * self.n * (p_ + self.epoch))  # cumulative index
        if self.train:
            # shuffle to avoid samples in the same epoch being continuous
            tmp_idx = sample_idx % len(self.fixed_shuffle[dataset_idx])
            sample_idx = self.fixed_shuffle[dataset_idx][tmp_idx]
        else:
            sample_idx = sample_idx % self.dataset_size[dataset_idx]
        return dataset_idx, sample_idx

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.get_sample_idx(idx)
        data = self.datasets[dataset_idx][sample_idx]
        for key in list(data.keys()):  # remove redundant attribute
            if key not in self.data_domain:
                data.pop(key)
        for key in self.data_domain:  # fill empty
            if key not in data.keys():
                data[key] = self.default_val[key]
                if type(data[key]) == np.ndarray:
                    data[key] = data[key].copy()

        for key in self.data_domain:  # check datatype
            if type(data[key]) == np.memmap:
                data[key] = np.array(data[key]).copy()
            if type(data[key]) == np.ndarray:
                if data[key].dtype == np.float32:
                    data[key] = data[key].astype(np.float64)
        assert set(data.keys()) == self.data_domain

        img, target = self.transform(data, train=self.train)

        for key in self.data_domain:  # check datatype
            if type(data[key]) == np.ndarray:
                if data[key].dtype == np.float64:
                    data[key] = data[key].astype(np.float32)

        return {'img': img}, target


if __name__ == '__main__':
    from datasets.cocodp import CocoDp
    from datasets.humanmesh import HumanMesh

    coco_dp = CocoDp(split='train2014')

    h36m = HumanMesh('h36m', r"./data/h36m/densepose_train_5.pt", r"./data/h36m/images")
    pw3d = HumanMesh('3dpw', r"./data/3dpw/densepose_train.pt", r"./data/3dpw")
    surreal = HumanMesh('surreal', r"./data/SURREAL_v1/densepose_train.pt", r"./data/SURREAL_v1")
    up3d = HumanMesh('up3d', r"./data/up-3d/densepose_high.pt", r"./data/up-3d/up-3d")

    mix_dataset = MixDataset([coco_dp, h36m, pw3d, surreal, up3d], args=None,
                             train=True, ratio=[3, 2, 1, 2, 1], n=int(3e5))  # [3, 2, 1, 2, 1]
    mix_loader = torch.utils.data.DataLoader(
        dataset=mix_dataset, batch_size=32, num_workers=0,
        shuffle=True)
    mix_loader.dataset.epoch = 3
    res = []
    for step, [inp, labels] in enumerate(mix_loader):
        print(step)
        # for idx in range(32):
        #     cv2.imwrite(
        #         os.path.join(r"D:\prj\ddp_reid\ddp_reid\render_res", f'{step}_{idx}.jpg'),
        #         np.clip(
        #             (inp['img'][idx].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 0.225 + 0.5) * 200,
        #             0, 255
        #         ).astype(np.uint8)
        #     )
