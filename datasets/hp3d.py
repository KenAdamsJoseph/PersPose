import json
import os
import shutil
import sys
import itertools
import json
import torch
import cv2
import joblib
import numpy as np
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import bbox


class HP3D(Dataset):
    def __init__(self, split='train_v2', dataset_path=r'./data/3dhp/'):
        assert split in ['train_v2', 'test']
        self.split = split
        self.dataset_path = dataset_path
        self.data = joblib.load(os.path.join(dataset_path, f'3dhp_{split}.pt'), 'r')
        self.h = [1.7994, 1.8039, 1.7621, 1.6546, 1.6756, 1.6977, 1.7823, 1.7548]  # from SMPL param. (NeuralAnnot)

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, idx):
        res = {}
        for attr in self.data.keys():
            res[attr] = self.data[attr][idx].copy()
        if 'train' in self.split:
            res['file_name'] = os.path.join(os.path.normpath(self.dataset_path),
                                            'mpi_inf_3dhp_train_set', res['file_name'])
        else:
            res['file_name'] = os.path.join(os.path.normpath(self.dataset_path),
                                            'mpi_inf_3dhp_test_set', res['file_name'])
        res['dataset_name'] = '3dhp'
        # res['bbox'] = np.array(bbox.scale_xywh(res['bbox'].tolist(), 1.1))
        # cam_intrinsics = torch.tensor(res['cam_intrinsics'], dtype=torch.float64, device='cpu')  # 3,3
        # uvd = cam_intrinsics.reshape(1, 1, 3, 3).matmul(torch.tensor(res['3dhp_j17'])[...,None]).squeeze(-1)
        # uvd[:, :, :2] /= uvd[:, :, 2:]  # bs, 6890, 3
        # uv = uvd[0, :, :2]
        #
        # from common.visualize import draw_joints2D
        # draw_joints2D(np.concatenate([np.array(bbox.xywh2xyxy(res['bbox'].tolist())).reshape(2,2),uv]),res['file_name'] )

        if 'train' in self.split:
            subject_id = int(os.path.basename(res['file_name']).split('_')[1][1:])
            res['height'] = np.array([self.h[subject_id-1]])
        return res

def preprocess_TS3_TS6():
    def interpolate_bbox(bbox,interval=10):  # bbox:bs,4
        interpolated = []
        for i in range(bbox.shape[0]-1):
            current_frame = bbox[i].unsqueeze(0)  # (1,4)
            next_frame = bbox[i+1].unsqueeze(0)  # (1,4)
            interpolated.append(current_frame)
            alphas = torch.linspace(1 / interval, (interval-1) / interval, interval-1)  # 生成9个alpha值
            interp_frames = (1 - alphas.unsqueeze(1)) * current_frame + alphas.unsqueeze(1) * next_frame
            interpolated.append(interp_frames)
        interpolated.append(bbox[-1].unsqueeze(0))
        result = torch.cat(interpolated, dim=0)
        return result

    data_set = HP3D(split='test')
    cam_intrinsics = [each for each in data_set.data['cam_intrinsics'][data_set.data['video_name'] == 'TS3'][31:41]][0]
    f,cx,cy = cam_intrinsics[0,0],cam_intrinsics[0,2],cam_intrinsics[1,2]
    bbox = torch.tensor([each for each in data_set.data['bbox'][data_set.data['video_name'] == 'TS3'][31:41]])
    bbox = interpolate_bbox(bbox,10)
    for i in range(bbox.shape[0]):
        file = f'./data/3dhp/mpi_inf_3dhp_test_set/TS3/imageSequence/img_000{501+i:03d}.jpg'
        target_file = os.path.basename(file)[:-4]+'_'+'_'.join([f'{int(each)}' for each in [f, cx, cy] + bbox[i].tolist()])+file[-4:]
        shutil.copy(file,os.path.join(r'./render_res/mip_inf_3dhp_test_TS3', target_file))

    cam_intrinsics = data_set.data['cam_intrinsics'][data_set.data['video_name'] == 'TS6'][0]
    f, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    bbox = torch.tensor([each for each in data_set.data['bbox'][data_set.data['video_name'] == 'TS6'][:284]])
    frame_idx = [int(each[-10:-4]) for each in data_set.data['file_name'][data_set.data['video_name'] == 'TS6'][:284]]
    # 373 - 51 - (98 - 69 - 1) - (171 - 159 - 1)
    # frame_idx[69 - 51:69 - 51 + 2]
    # frame_idx[80:82]
    bbox = torch.cat([
        bbox[:69 - 51],
        interpolate_bbox(bbox[69 - 51:69 - 51 + 2],98-69),
        bbox[69 - 51+2:80],
        interpolate_bbox(bbox[80:82], 171-159),
        bbox[82:]
    ])
    for i in range(bbox.shape[0]):
        file = f'./data/3dhp/mpi_inf_3dhp_test_set/TS6/imageSequence/img_000{51+i:03d}.jpg'
        target_file = os.path.basename(file)[:-4]+'_'+'_'.join([f'{int(each)}' for each in [f, cx, cy] + bbox[i].tolist()])+file[-4:]
        shutil.copy(file,os.path.join(r'./render_res/mip_inf_3dhp_test_TS6', target_file))


if __name__ == '__main__':
    data_set = HP3D(split='test')  # , dataset_path=)
    print(data_set.__len__())
    loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=32, num_workers=0,
        shuffle=True)
    print(iter(loader)._next_data().keys())
    res = []
    for step, each in enumerate(loader):
        res.append(each)
        # from common.visualize import draw_joints2D
        # for i in range(32):
        #     draw_joints2D(np.array(bbox.xywh2xyxy(each['bbox'][i].tolist())).reshape(2,2),each['file_name'][i])

