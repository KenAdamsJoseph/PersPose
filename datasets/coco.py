import json
import os
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


class COCO(Dataset):
    def __init__(self, split='train2017', dataset_path=r'./data/coco/'):
        assert split in ['train2017', ]
        self.split = split
        self.dataset_path = dataset_path
        self.data = joblib.load(os.path.join(dataset_path, f'coco_{split}.pt'), 'r')

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, idx):
        res = {}
        for attr in self.data.keys():
            res[attr] = self.data[attr][idx].copy()
        res['file_name'] = os.path.join(os.path.normpath(self.dataset_path), self.split, res['file_name'])
        res['dataset_name'] = 'coco'
        res['coco_kp17'] = res['coco_kp17'].reshape(-1, 3).astype(np.float32)
        # res['bbox'] = np.array(bbox.scale_xywh(res['bbox'].tolist(), 1.1))
        return res


if __name__ == '__main__':
    data_set = COCO(split='train2017')  # , dataset_path=)
    print(data_set.__len__())
    loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=32, num_workers=0, shuffle=False)
    print(iter(loader)._next_data().keys())
    res = []
    for step, each in enumerate(loader):
        res.append(each)
        # from common.visualize import draw_joints2D
        # for i in range(32):
        #     draw_joints2D(np.array(bbox.xywh2xyxy(each['bbox'][i].tolist())).reshape(2,2),each['file_name'][i])
