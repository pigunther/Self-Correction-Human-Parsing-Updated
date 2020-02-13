#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   dataset.py
@Time    :   8/30/19 9:12 PM
@Desc    :   Dataset Definition
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import cv2
import random
import numpy as np

import torch
from torch.utils import data
import pandas as pd

from scipy.stats import multivariate_normal
from torch.utils import data
import torchvision.transforms as transforms
from target_generation import generate_edge


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_h = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_h * 0.5, dst_w * 0.5]
    dst[1, :] = np.array([dst_h * 0.5, dst_w * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),  # (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)

    return target_logits


class SCHPDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        self.file_list = os.listdir(self.root)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'name': img_name,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta


class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[384, 384], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=None, drop_factor=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = flip_prob
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset
        self.keypoints = pd.read_csv(root + f'/TrainVal_pose_annotations/lip_{dataset}_set.csv', header=None)
        self.keypoints[0] = self.keypoints[0].astype(str)

        list_path = os.path.join(self.root, self.dataset + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        if drop_factor is not None:
            self.im_list = self.im_list[::drop_factor]
        self.drop_factor = drop_factor
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([h * 1.0, w * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset + '_images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        #         if im.shape[1] < self.crop_size[0] or im.shape[2] < self.crop_size[1]:
        #             # resize_shape = (im.shape[0], self.crop_size[1], self.crop_size[0])
        #             im = cv2.resize(im, tuple(self.crop_size))
        if im.shape[-1] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, h - 1, w - 1])
        # print(center, s, 'center scale')
        r = 0

        if self.dataset != 'test':
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train' or self.dataset == 'val':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]

                    center[0] = im.shape[1] - center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(center, s, r, self.crop_size)
        kp = self.keypoints.loc[self.keypoints[0] == im_name + '.jpg']
        kp = np.array(kp.values[0])[1:]
        kp = np.where(kp == kp, kp, -1).astype(np.float64)
        heatmap_params = {
            'h': h,
            'w': w,
            'center': center,
            'r': r,
            's': s,
            'kp': kp,
            'trans': trans,
            'to_tensor_flag': self.transform is not None
        }
        # print(type(heatmap_params))
        # for k in heatmap_params:
        #     print(type(heatmap_params[k]))

        def get_heatmap_by_radius(radius):
            local_heatmap = get_train_kp_heatmap(h, w, kp, radius=radius)
            local_heatmap = cv2.warpAffine(
                local_heatmap,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))
            if self.transform:
                local_heatmap = transforms.ToTensor()(local_heatmap).type(torch.float32)
            return local_heatmap
        heatmap = get_train_kp_heatmap(h, w, kp)

        input_img = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        heatmap = cv2.warpAffine(
            heatmap,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        #         for i in range(heatmap.shape[-1]):
        #             heatmap[:,:, i] = cv2.warpAffine(
        #                 heatmap[:,:, i],
        #                 trans,
        #                 (int(self.crop_size[1]), int(self.crop_size[0])),
        #                 flags=cv2.INTER_LINEAR,
        #                 borderMode=cv2.BORDER_CONSTANT,
        #                 borderValue=(0, 0, 0))

        if self.transform:
            input_img = self.transform(input_img)
            heatmap = transforms.ToTensor()(heatmap).type(torch.float32)
        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r,
            'index': index
        }

        if self.dataset != 'train' and self.dataset != 'val': #todo: delete val
            return input_img, meta, heatmap
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_edge = generate_edge(label_parsing)

            label_parsing = torch.from_numpy(label_parsing)
            label_edge = torch.from_numpy(label_edge)
            return input_img, label_parsing, label_edge, meta, heatmap, heatmap_params


def get_train_kp_heatmap(h, w, kp, radius=40, kp_num=16):
    radius = int(radius)
    x = [kp[i] for i in range(0, len(kp), 3)]
    y = [kp[i] for i in range(1, len(kp), 3)]
    pos = np.dstack(np.mgrid[0:h:1, 0:w:1])
    heatmap = np.zeros((h, w, kp_num))
    for x_i, y_i, i in zip(x, y, list(range(kp_num))):
        if x_i != -1 and y_i != -1:
            if y_i >= h:
                y_i -= 1
            if x_i >= w:
                x_i -= 1
            if x_i >= w or y_i >= h:
                continue
            heatmap[int(y_i), int(x_i), i] = 1.0
            heatmap[:, :, i] = cv2.GaussianBlur(heatmap[:, :, i], (radius + radius % 2 - 1, radius + radius % 2 - 1), 0)
    return heatmap