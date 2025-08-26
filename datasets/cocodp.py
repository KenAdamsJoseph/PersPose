import json
import os
import sys
import itertools
import json
import torch
import cv2
import joblib
import numpy as np
import pycocotools.mask as mask_utils
from torch.utils.data import Dataset

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root)
from common import bbox


class CocoDp(Dataset):
    def __init__(self, split='train2014', dataset_path=r'./data/coco/'):
        # assert split in ['train2014', 'valminusminival2014', 'minival2014']
        self.split = split
        self.dataset_path = dataset_path
        self.data = joblib.load(os.path.join(prj_root, dataset_path, f'densepose_{split}.pt'), 'r')
        valid = self.data['dp_n'] != 0
        for attr in self.data.keys():
            self.data[attr] = self.data[attr][valid]

        freq_file = os.path.join(prj_root, dataset_path, f'densepose_{split}.pt_freq.npy')
        if not os.path.exists(freq_file):
            all_vertices = np.concatenate(
                [each[:self.data['dp_n'][i]] for i, each in enumerate(self.data['dp_face_vertices'])]
            ).reshape(-1)
            freq = np.array([(all_vertices == v_idx).sum() for v_idx in range(6890)]) / len(self.data['id'])
            np.save(freq_file, freq)
        else:
            freq = np.load(freq_file)
        dp_weight = 1 / (freq + 0.1)
        self.dp_weight = dp_weight / dp_weight.mean()

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, idx):
        res = {'idx':idx}
        for attr in self.data.keys():
            res[attr] = self.data[attr][idx].copy()

        res['dataset_name'] = 'coco_dp'

        if 'train' in self.split:
            img_path = os.path.join(self.dataset_path, 'train2014')
        else:
            img_path = os.path.join(self.dataset_path, 'val2014')
        res['file_name'] = os.path.join(prj_root, img_path, res['file_name'])

        res['mask'] = mask_utils.decode({'counts': bytes(res.pop('dp_mask')), 'size': [256, 256]})
        res['mask_valid'] = True
        res['bbox_mask'] = res['bbox'].copy()

        res['dp_uv'] = np.zeros((6890, 2))
        res['dp_uv'][:res['dp_n'], 0] = res['dp_x_global'][:res['dp_n']]
        res['dp_uv'][:res['dp_n'], 1] = res['dp_y_global'][:res['dp_n']]

        res['dp_face'] = np.zeros((6890, 3), dtype=np.int32)
        res['dp_face'][:res['dp_n']] = res['dp_face_vertices'][:res['dp_n']]

        res['dp_bc'] = np.ones((6890, 3)) / 3
        res['dp_bc'][:res['dp_n']] = res['dp_bcs'][:res['dp_n']]

        res['dp_valid'] = np.zeros((6890,), dtype=bool)
        res['dp_valid'][:res['dp_n']] = True

        res['dp_weight'] = np.zeros((6890, 3))
        res['dp_weight'][:res['dp_n']] = self.dp_weight[res['dp_face_vertices'][:res['dp_n']]]

        for k in ['dp_face_vertices', 'dp_bcs', 'dp_x_global', 'dp_y_global', ]:
            res.pop(k)
        return res

    def visualize(self, idx):
        data = self.__getitem__(idx)

        from common.visualize import draw_joints2D
        draw_joints2D(data['dp_uv'][data['dp_valid']], data['file_name'], radius=2, suffix='_densepose')

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

        return
        from common.smpl_wrapper import SMPL
        import trimesh

        smpl = SMPL(gender='neutral', batch_size=1, dtype=torch.float64, device='cpu')
        joints, vertices = smpl.cal_rest_pose(np.zeros((1, 10)), return_vertices=True)
        vertices = vertices[0] - joints[0][[2, 3]].mean(dim=0, keepdim=True)

        val = vertices.norm(dim=-1)
        val = val * 0
        val[np.unique(data['dp_face'][:data['dp_n']])] = 1

        m = trimesh.Trimesh(vertices=vertices, faces=smpl.faces)
        face_val = val[smpl.faces.astype(np.long)].mean(dim=-1)
        face_val = face_val - face_val.min()
        face_val = face_val / face_val.max()

        f_color = torch.tensor(trimesh.visual.color.interpolate(face_val, color_map="viridis")[:, :3])
        m.visual.face_colors = f_color
        m.show()


if __name__ == '__main__':
    data_set = CocoDp(split='train2014')  # train2014 valminusminival2014  minival2014
    print(data_set.__len__())

    loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=32, num_workers=0, shuffle=False)
    batch = iter(loader)._next_data()
    print(batch.keys())

    for idx in np.random.randint(0, len(data_set), (32,)):
        data_set.visualize(idx)
