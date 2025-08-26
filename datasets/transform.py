import cv2
import numpy as np
import random
import os
import sys
import math
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import bbox
from common.visualize import array_show, draw_joints2D
from common.geometry import *


def spherical2cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def perspective_transform(uv, perspective_k):
    # uv:(n,2), perspective_k:(3,3)
    uv1 = np.concatenate((uv, uv[:, :1] * 0 + 1), axis=-1)
    uv1_ = (perspective_k @ (uv1.T)).T
    return uv1_[:, :2] / uv1_[:, 2:]


class Transform(object):
    def __init__(self, args=None):
        self.prob = args.TRANSFORM['PROB'] if args is not None else {
            'no_augmentation': 0.3,

            'cutout': 0.5,
            'perspective': 0.5,
            'rot': 0.5,
            'shift': 0.5,
            'scale': 0.5,
            'stretch': 0.5,

            'gaussian_noise': 0.3,
            'blur_sharp': 0.3,
            'contrast_brightness': 0.5,
            'saturation_hue': 0.5,
            'channel': 0.1,
        }
        self.factor = args.TRANSFORM['FACTOR'] if args is not None else {
            'cutout': 0.4,  # max(cutout area)==cutout*bbox_area
            'perspective_field': 30,
            'rot': 30,  # degree
            'shift': 0.2,  # 20%
            'scale': 0.2,  # 20%
            'stretch': 0.2,  # 20%
        }
        self.result_img_size = args.TRANSFORM['IMAGE_SIZE'] if args is not None else [128, 256]  # w,h
        self.dp_margin = args.TRANSFORM['DP_MARGIN'] if args is not None else 0
        self.perspective_center = args.TRANSFORM['PERSPECTIVE_CENTER'] if args is not None else False
        self.perspective_center_aug = args.TRANSFORM['PERSPECTIVE_CENTER_AUG']

    def add_gaussian_noise(self, image: np.array, mean=0, std=None):
        if random.random() < 1 - self.prob['gaussian_noise']: return image
        if std is None: std = image.mean() / 5
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def adjust_saturation_hue(self, image, factor_h=40, factor_s=1.0):
        if random.random() < 1 - self.prob['saturation_hue']: return image
        shift_h = int((random.random() * 2 - 1) * factor_h)
        scale_s = factor_s * random.random() + (1 - factor_s)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + shift_h) % 180  # -1%180==179
        hsv_image[:, :, 1] = np.clip(scale_s * 255 / (1 + hsv_image[:, :, 1].max()) * hsv_image[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def adjust_contrast_brightness(self, image, alpha_=0.6):
        if random.random() < 1 - self.prob['contrast_brightness']: return image
        rgb_mean = image.mean()
        alpha = 0.1 + 1 + alpha_ * (random.random() * 2 - 1)  # [0.5,1.7]
        beta = 0.5 * rgb_mean * (random.random() * 2 - 1)
        img_adjusted = np.clip((np.float32(image) - rgb_mean) * alpha + rgb_mean + beta, 0, 255).astype('uint8')
        # img_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return img_adjusted

    def adjust_blur_sharp(self, image):
        if random.random() < 1 - self.prob['blur_sharp']: return image
        behavior = random.choice(['resize_blur', 'gaussian_blur', 'sharpen'])
        if behavior == 'resize_blur':
            scale = random.choice([8 / 5, 8 / 4, 8 / 3])
            img_wh = image.shape[:2][::-1]
            new_wh = [int(each / scale) for each in img_wh]
            resized_image = cv2.resize(cv2.resize(image, new_wh), img_wh)
            return resized_image
        elif behavior == 'gaussian_blur':
            k = random.choice([7, 11, 15])
            return cv2.GaussianBlur(image, (k, k), 0)
        else:  # sharpen
            k = random.choice([3, 7, 11, 15])
            L = cv2.GaussianBlur(image, (k, k), k)
            H = cv2.subtract(image, L)
            return cv2.addWeighted(image, 1, H, 2, 0)

    def channel_aug(self, image):
        if random.random() < 1 - self.prob['channel']: return image
        if random.random() < 0.5:
            selected_channel = random.randrange(0, 3)
            return image[:, :, selected_channel, None].repeat(3, -1)
        else:
            w_ = [random.random(), random.random(), random.random()]
            w = [each / sum(w_) for each in w_]
            weighted_gray_img_ = (image.astype(np.float32) * np.float32(w)[None, None]).sum(axis=-1)
            weighted_gray_img = weighted_gray_img_[:, :, None].repeat(3, -1).astype(image.dtype)
            return weighted_gray_img

    def pixel_wise_aug(self, image):
        image = self.adjust_blur_sharp(image)
        image = self.add_gaussian_noise(image)
        image = self.adjust_contrast_brightness(image)
        image = self.adjust_saturation_hue(image)
        image = self.channel_aug(image)
        return image

    def generate_cutout_bbox(self, xywh, image_hw):
        x, y, w, h = xywh
        cutout_area = w * h * (0.2 + 0.8 * random.random()) * self.factor['cutout']
        r_min = 0.3  # min(cutout ratio) ∈[0,1]
        aspect_ratio = random.random() * (1 / r_min - r_min) + r_min
        cutout_width = (cutout_area * aspect_ratio) ** 0.5
        cutout_height = (cutout_area / aspect_ratio) ** 0.5

        x_min, y_min, x_max, y_max = bbox.xywh2xyxy(xywh)
        center_x = random.randint(int(x_min) + 2, int(x_max) - 2)
        center_y = random.randint(int(y_min) + 2, int(y_max) - 2)

        cutout_x_min = center_x - cutout_width / 2
        cutout_y_min = center_y - cutout_height / 2
        cutout_x_max = cutout_x_min + cutout_width
        cutout_y_max = cutout_y_min + cutout_height

        cutout_xyxy = [int(cutout_x_min), int(cutout_y_min), int(cutout_x_max), int(cutout_y_max)]
        cutout_xyxy = bbox.clip_xyxy(cutout_xyxy, [0, 0, image_hw[1], image_hw[0]])
        return cutout_xyxy

    def crop_aug(self, bbox, rot_angle, shift_ratio, scale_ratio, stretch_ratio):
        crop_w, crop_h = self.result_img_size
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        img_center = crop_w / 2, crop_h / 2
        bb_center = (bbox_x + bbox_w / 2, bbox_y + bbox_h / 2)
        tx = img_center[0] - bb_center[0] + shift_ratio[0] * bbox_w
        ty = img_center[1] - bb_center[1] + shift_ratio[1] * bbox_h
        affine_shift = np.float64([[1, 0, tx], [0, 1, ty]])

        affine_rot = cv2.getRotationMatrix2D(img_center, rot_angle, scale_ratio)

        if stretch_ratio > 1:
            stretch_A = np.float64([[stretch_ratio, 0], [0, 1]])
        else:
            stretch_A = np.float64([[1, 0], [0, 1 / stretch_ratio]])
        stretch_b = np.float64(img_center)[:, None] - stretch_A @ np.float64([[img_center[0]], [img_center[1]]])
        affine_stretch = np.concatenate([stretch_A, stretch_b], axis=-1)

        affine = np.concatenate([affine_stretch, np.float64([[0, 0, 1]])]) @ np.concatenate(
            [affine_rot, np.float64([[0, 0, 1]])]) @ np.concatenate([affine_shift, np.float64([[0, 0, 1]])])
        return affine

    def __call__(self, label, train=True):
        img = cv2.cvtColor(cv2.imread(label['file_name']), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        crop_w, crop_h = self.result_img_size

        do_aug = train and random.random() > self.prob['no_augmentation']

        # The original mask is stored in shape 256*256, regardless of the person bbox.
        # Perform mask_prepare_affine to transform the original mask into original image space.
        # array_show(cv2.warpAffine(label['mask'], mask_prepare_affine[:2], (img_w,img_h)))
        bbox_mask_u, bbox_mask_v, bbox_mask_w, bbox_mask_h = label['bbox_mask']
        mask_prepare_affine = np.array([
            [bbox_mask_w / 256, 0, bbox_mask_u],
            [0, bbox_mask_h / 256, bbox_mask_v],
            [0, 0, 1],
        ], dtype=np.float64)

        # #######rotate########
        rot_angle, perspective_R, perspective_k, affine_rot, M_rot = 0, np.eye(3), np.eye(3), np.eye(3), np.eye(3)
        if self.perspective_center:
            if label['dataset_name'] == '3dhp' or label['dataset_name'].startswith('hps_'):
                cx, cy, w_, h_ = bbox.xywh2center_wh(label['bbox'])
                image_center = label['cam_intrinsics'][:2, -1]
                if do_aug and self.perspective_center_aug:
                    image_center = image_center + (np.random.rand(2)-0.5)*2*0.1*max(w_,h_)
                perspective_R, perspective_k = adjust_perspective(
                    label['cam_intrinsics'], [cx, cy], image_center.tolist())
        if do_aug:
            # for k in self.prob.keys(): self.prob[k] = 1
            if random.random() < self.prob['rot']:
                rot_angle = (random.random() * 2 - 1) * self.factor['rot']  # ∈(-30,30)
            if random.random() < self.prob['perspective']:
                assert not self.perspective_center
                if label['dataset_name'] == '3dhp' or label['dataset_name'].startswith('hps_'):
                    # perspective orientation in spherical coordinate: perspective_theta∈(0,30°) perspective_phi∈(0,360°)
                    perspective_theta = random.random() * self.factor['perspective_field']
                    perspective_phi = random.random() * 360
                    xyz = spherical2cartesian(1, np.radians(perspective_theta), np.radians(perspective_phi))
                    cx_aim, cy_aim, _1 = np.dot(label['cam_intrinsics'], xyz) / xyz[-1]
                    cx, cy, w_, h_ = bbox.xywh2center_wh(label['bbox'])
                    perspective_R, perspective_k = adjust_perspective(
                        label['cam_intrinsics'], [cx, cy], [cx_aim, cy_aim], rot_angle=rot_angle)
                    # array_show(cv2.warpPerspective(img, perspective_k, label['image_wh']))
                    rot_angle = 0  # avoid overrotation
            affine_rot = np.concatenate((
                cv2.getRotationMatrix2D((img_w / 2, img_h / 2), -1 * rot_angle, 1),
                np.array([[0, 0, 1], ], dtype=np.float64)
            ), axis=0)
            M_rot = affine_rot.copy()
            M_rot[0, -1], M_rot[1, -1] = 0, 0

        # rotate bbox
        bbox_u, bbox_v, bbox_w, bbox_h = label['bbox']
        bbox_vertex = np.array([[bbox_u, bbox_v], [bbox_u, bbox_v + bbox_h],
                                [bbox_u + bbox_w, bbox_v], [bbox_u + bbox_w, bbox_v + bbox_h]])
        bbox_vertex_rotated = perspective_transform(bbox_vertex, affine_rot @ perspective_k)
        bbox_rotated = np.array(bbox.xyxy2xywh(
            np.concatenate((bbox_vertex_rotated.min(axis=0), bbox_vertex_rotated.max(axis=0)))))
        if any(label['dp_valid']):
            dp_uv_rotated = perspective_transform(label['dp_uv'], affine_rot @ perspective_k)
            if label['dataset_name'] == 'coco_dp':
                bbox_rotated = np.array(bbox.xyxy2xywh(
                    np.concatenate((
                        dp_uv_rotated[label['dp_valid']].min(axis=0),  # (2,)
                        dp_uv_rotated[label['dp_valid']].max(axis=0),  # (2,)
                    ))  # (4,)
                ))  # (4,)
            else:
                assert label['dataset_name'].startswith('hps_')
                bbox_rotated = np.array(bbox.xyxy2xywh(
                    np.concatenate((
                        dp_uv_rotated.min(axis=0),  # (2,)
                        dp_uv_rotated.max(axis=0),  # (2,)
                    ))  # (4,)
                ))  # (4,)
        if self.perspective_center:
            assert self.prob['shift'] == 0, 'bbox shift will ruin perspective center'
            if label['dataset_name'] == '3dhp' or label['dataset_name'].startswith('hps_'):
                # expand the bbox_rotated to keep perspective center.
                image_center = label['cam_intrinsics'][:2, -1][None]  # 1,2
                bbox_wh_ = 2*(np.array(bbox.xywh2xyxy(bbox_rotated)).reshape(2,2)-image_center).__abs__().max(axis=0)
                bbox_xyxy_ = np.concatenate([image_center - bbox_wh_/2, image_center + bbox_wh_/2]).reshape(4)
                bbox_rotated = bbox.xyxy2xywh(bbox_xyxy_)
        # img_rotated = cv2.warpPerspective(img, (affine_rot @ perspective_k), (img_w, img_h))
        # array_show(img_rotated)
        # draw_joints2D(bbox_vertex_rotated, 'bbox', image=img_rotated.astype(np.float32))
        # draw_joints2D(np.array(bbox.xywh2xyxy(bbox_rotated)).reshape(2, 2), 'bbox_rotated', image=img_rotated.astype(np.float32))
        # draw_joints2D(dp_uv_rotated[label['dp_valid']], 'dp_rotated', image=img_rotated.astype(np.float32))
        # #######rotate########

        # #######crop########
        bbox_rot_u, bbox_rot_v, bbox_rot_w, bbox_rot_h = bbox_rotated
        shift_ratio, scale_ratio = [0, 0], 1
        if label['dataset_name'] == '3dhp' or label['dataset_name'].startswith('hps_'):
            stretch_ratio_default = label['cam_intrinsics'][1, 1] / label['cam_intrinsics'][0, 0]
        else:
            stretch_ratio_default = 1
        stretch_ratio = stretch_ratio_default if self.prob['stretch'] == 0 else (crop_w / crop_h) / (bbox_rot_w / bbox_rot_h)
        crop_scale = min(crop_w / bbox_rot_w, crop_h / bbox_rot_h)
        if do_aug:
            # for k in self.prob.keys(): self.prob[k] = 1
            if random.random() < self.prob['shift']:
                shift_ratio = [(random.random() * 2 - 1) * self.factor['shift'],
                               (random.random() * 2 - 1) * self.factor['shift']]
            if random.random() < self.prob['scale']:
                scale_ratio = 0.9 + (random.random() * 2 - 1) * self.factor['scale']  # 0.9-0.3,0.9+0.3
            if random.random() < self.prob['stretch']:
                stretch_ratio = stretch_ratio + (random.random() * 2 - 1) * self.factor['stretch']
        stretch_ratio = np.clip(stretch_ratio, 1 / 2, 2)
        crop_affine = self.crop_aug(bbox_rotated, 0, shift_ratio, crop_scale * scale_ratio, stretch_ratio)
        # #######crop########

        if label['dataset_name'] == '3dhp' or label['dataset_name'].startswith('hps_'):
            label['cam_intrinsics_origin'] = label['cam_intrinsics'].copy()
            label['cam_intrinsics'] = crop_affine @ label['cam_intrinsics']

        label['rotation_3d'] = M_rot @ perspective_R
        if label['dataset_name'].startswith('hps_'):
            R = label['rotation_3d']
            j0_relative = label['smpl_j24'][0] - label['smpl_transl']
            label['smpl_j24'] = (R @ label['smpl_j24'].T).T
            label['smpl_j17'] = (R @ label['smpl_j17'].T).T
            label['smpl_vertices'] = (R @ label['smpl_vertices'].T).T
            smpl_orient = batch_rodrigues(torch.tensor(label['smpl_theta'][0][None]))[0]
            label['smpl_theta'][0] = quaternion_to_angle_axis(mat2quat(torch.tensor(R).matmul(smpl_orient)))
            label['smpl_transl'] = label['smpl_j24'][0] - j0_relative
            label['smpl_ortho_twist'] = (R @ label['smpl_ortho_twist'].T).T

        label['perspective_2d'] = crop_affine @ affine_rot @ perspective_k
        img_cropped = cv2.warpPerspective(img, label['perspective_2d'], (crop_w, crop_h))
        label['mask'] = cv2.warpPerspective(
            label['mask'].astype(np.float64), (label['perspective_2d'] @ mask_prepare_affine), (crop_w, crop_h))
        dp_uv_cropped = perspective_transform(label['dp_uv'], label['perspective_2d'])

        label['coco_kp17'][:, :2] = perspective_transform(label['coco_kp17'][:, :2], label['perspective_2d'])
        label['coco_kp17'][:, -1] *= ((label['coco_kp17'][:, 0] > 0) * (label['coco_kp17'][:, 0] < crop_w))
        label['coco_kp17'][:, -1] *= ((label['coco_kp17'][:, 1] > 0) * (label['coco_kp17'][:, 1] < crop_h))
        # draw_joints2D(label['coco_kp17'][:, :2][label['coco_kp17'][:, -1] > 0].astype(np.int64),
        #               label['file_name'][-8:], suffix='_dp', radius=2, image=img_cropped.astype(np.float32))

        if label['dataset_name'] == '3dhp':
            label['3dhp_j17'] = (label['rotation_3d'] @ label['3dhp_j17'].T).T
            # cam_intrinsics = torch.tensor(label['cam_intrinsics'], dtype=torch.float64, device='cpu')  # 3,3
            # uvd = (cam_intrinsics@label['3dhp_j17'].T).T
            # uvd[:, :2] /= uvd[:, 2:]  # bs, 6890, 3
            # draw_joints2D(uvd[:, :2], label['file_name'][-8:], suffix='_dp', radius=2, image=img_cropped.astype(np.float32))

        # filter points outside the frame
        margin = self.dp_margin  # corner grid cant perform interpolation for dp_pred_prob
        dp_in_frame = np.ones_like(dp_uv_cropped[:, 0]).astype(np.bool8)  # 6890
        dp_in_frame *= ((dp_uv_cropped[:, 0] > margin) * (dp_uv_cropped[:, 0] < crop_w - margin))
        dp_in_frame *= ((dp_uv_cropped[:, 1] > margin) * (dp_uv_cropped[:, 1] < crop_h - margin))

        if do_aug and random.random() < self.prob['cutout']:
            cutout_xyxy = self.generate_cutout_bbox([0, 0, crop_w, crop_h], image_hw=[crop_h, crop_w])

            cutout_xywh = bbox.xyxy2xywh(cutout_xyxy)
            fill1 = np.random.rand(cutout_xywh[-1], cutout_xywh[-2], 3) * 255
            fill2 = np.random.rand(3) * 255
            fill = fill1 if random.random() < 0.5 else fill2
            img_cropped[cutout_xyxy[1]:cutout_xyxy[3], cutout_xyxy[0]:cutout_xyxy[2]] = fill

            label['mask'][cutout_xyxy[1]:cutout_xyxy[3], cutout_xyxy[0]:cutout_xyxy[2]] = 0

            dp_in_cut_box = np.ones(dp_uv_cropped.shape[0]).astype(np.bool8)  # 6890
            dp_in_cut_box *= ((dp_uv_cropped[:, 0] > cutout_xyxy[0]) * (dp_uv_cropped[:, 0] < cutout_xyxy[2]))
            dp_in_cut_box *= ((dp_uv_cropped[:, 1] > cutout_xyxy[1]) * (dp_uv_cropped[:, 1] < cutout_xyxy[3]))
        else:
            dp_in_cut_box = np.zeros(dp_uv_cropped.shape[0]).astype(np.bool8)  # 6890

        # array_show(img_cropped,save_path='./render_res/'+label['file_name'][-8:]+'_maaaa.jpg')
        # array_show(label['mask'], save_path='./render_res/'+label['file_name'][-8:] + '_mask.jpg')
        # draw_joints2D(label['dp_uv'][label['dp_valid']], label['file_name'][-8:],
        #               suffix='_dp', radius=2, image=img.astype(np.float32))
        # draw_joints2D(label['dp_uv'][label['dp_valid']* dp_in_frame * (~dp_in_cut_box)], label['file_name'][-8:],
        #               suffix='_dp_keeped',radius=2, image=img.astype(np.float32))
        label['dp_valid'] = (label['dp_valid'] * dp_in_frame * (~dp_in_cut_box))
        label['dp_uv'] = np.stack([
            dp_uv_cropped[:, 0].clip(0, crop_w - 1).astype(np.int64),
            dp_uv_cropped[:, 1].clip(0, crop_h - 1).astype(np.int64),
        ], axis=-1)
        # draw_joints2D(label['dp_uv'][label['dp_valid']], label['file_name'][-8:],
        #               suffix='dp_cropped', image=img_cropped.astype(np.float32))

        if do_aug:
            img_transformed = self.pixel_wise_aug(img_cropped)
        else:
            img_transformed = img_cropped
        # if any(label['dp_valid']):
        #     array_show(img_transformed, save_path='./render_res/' + label['file_name'][-8:] + '_ma.jpg')

        # normalize
        img_transformed = torch.tensor(img_transformed / 255, dtype=torch.float32)
        img_transformed -= torch.tensor([0.485, 0.456, 0.406])
        img_transformed /= torch.tensor([0.229, 0.224, 0.225])
        img_transformed = img_transformed.permute(2, 0, 1)  # C*H*W
        return img_transformed, label
