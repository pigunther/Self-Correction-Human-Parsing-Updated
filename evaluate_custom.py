#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate_custom.py.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Evaluation Scripts
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import network
from datasets import SCHPDataset, transform_logits

from datasets import LIPDataSet


dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label':['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--restore-weight", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--input", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    parser.add_argument("--data-dir", type=str, default='../lip-dataset/LIP',
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--gpu", type=str, default='1,2,3',
                        help="choose gpu device.")
    parser.add_argument("--with_my_bn", type=int, default=False,
                        help="choose the number of recurrence.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']

    model = network(num_classes=num_classes, pretrained=None, with_my_bn=args.with_my_bn, batch_size=args.batch_size).cuda()
    model = nn.DataParallel(model)
    state_dict = torch.load(args.restore_weight)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform, rotation_factor=0, flip_prob=-0.1, scale_factor=0)
    # dataset = SCHPDataset(root=args.input, input_size=input_size, transform=transform)
    dataloader = DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus), drop_last=True)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    palette = get_palette(num_classes)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            image, meta, heatmaps = batch

            img_names = meta['name']
            cs = meta['center'].numpy()
            ss = meta['scale'].numpy()
            ws = meta['width'].numpy()
            hs = meta['height'].numpy()

            print(image.shape)
            output = model(image.cuda(), heatmaps)
            print('output shape', len(output), len(output[0]), len(output[0][1]))
            output = output[0][1]
            print(output.shape)
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_outputs = upsample(output)
            upsample_outputs = upsample_outputs.squeeze()
            # print('upsample_output ', upsample_output.shape)
            for upsample_output, img_name, c, s, w, h in zip(upsample_outputs, img_names, cs, ss, ws, hs):
                upsample_output = upsample_output.permute(1, 2, 0) #CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)

                parsing_result_path = os.path.join(args.output, img_name[:-4]+'.png')
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                output_img.putpalette(palette)
                output_img.save(parsing_result_path)
                if args.logits:
                    logits_result_path = os.path.join(args.output, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()

# python3 evaluate_custom.py --dataset lip --restore-weight snapshots/LIP_epoch_133.pth --input input --output output
# python3 evaluate_custom.py --dataset lip --restore-weight snapshots_simple/LIP_epoch_49.pth --input input --output output --with_my_bn 0

# python3 evaluate_custom.py --dataset lip --restore-weight snapshots1/LIP_epoch_49.pth --input input --output output --with_my_bn 1
