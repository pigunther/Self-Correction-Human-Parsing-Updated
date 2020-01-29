#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   model.py
@Time    :   8/30/19 9:10 PM
@Desc    :   Augmented-CE2P Network Achitecture. Reference: https://github.com/liutinglt/CE2P
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

pretrained_settings = {
    'resnet101': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class InPlaceABNSync(nn.Module):
    """
    Serve same as the InplaceABNSync.
    Reference: https://github.com/mapillary/inplace_abn
    """

    def __init__(self, num_features):
        super(InPlaceABNSync, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features=2048, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class EdgeModule(nn.Module):
    """
    Edge branch.
    """

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super(EdgeModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class DecoderModule(nn.Module):
    """
    Parsing Branch Decoder Module.

    """

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, batch_size, with_my_bn=False):

        self.inplanes = 128

        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        # if with_my_bn:
        #     # self.bn1 = MyBatchNorm2d(64)
        #     self.bn1 = ConvBatchNorm2d(64)
        # else:
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)  # stride 16

        self.context_encoding = PSPModule()
        self.edge = EdgeModule()
        self.decoder = DecoderModule(num_classes)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

        self.with_my_bn = with_my_bn
        self.heatmap_conv1 = conv3x3(3, 64, stride=2)
        self.heatmap_conv2 = conv3x3(64, 64, stride=1)
        self.heatmap_conv3 = conv3x3(64, 64, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=1))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=1))

        return nn.Sequential(*layers)

    def forward(self, x, heatmaps):
        if self.with_my_bn:
            heatmaps = self.heatmap_conv1(heatmaps)
            gamma = self.heatmap_conv2(heatmaps)
            beta = self.heatmap_conv3(heatmaps)
            x = self.conv1(x)
            # x = self.bn1(self.conv1(x))
            x = x * (gamma + 1) + beta
            x = self.relu1(x)
        else:
            x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], [edge_result]]


def initialize_pretrained_model(model, settings, pretrained='./models/resnet101-imagenet.pth'):
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

    if pretrained is not None:
        saved_state_dict = torch.load(pretrained)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        model.load_state_dict(new_params)


def network(num_classes=20, pretrained='./models/resnet101-imagenet.pth', batch_size=12, with_my_bn=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, batch_size,  with_my_bn=with_my_bn)
    settings = pretrained_settings['resnet101']['imagenet']
    initialize_pretrained_model(model, settings, pretrained)
    return model



class MyBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x, heatmaps=None):
        self._check_input_dim(x)
        y = x.transpose(0,1)

        print(y.shape)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        y = y - mu.view(-1, 1)
        y = y / (sigma2.view(-1, 1) ** .5 + self.eps)
        # if self.training is not True:
        #
        #     y = y - self.running_mean.view(-1, 1)
        #     y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
        # else:
        #     if self.track_running_stats is True:
        #         with torch.no_grad():
        #             print('ruunig_mean. momentum. mu', self.running_mean.shape, self.momentum, mu.shape)
        #             print(y.shape, self.running_mean.shape, self.running_var.shape)
        #             self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
        #             self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
        #     y = y - mu.view(-1,1)
        #     y = y / (sigma2.view(-1,1)**.5 + self.eps)
        # print('weight in batchnorm ', self.weight.shape, y.shape)
        # if heatmaps is None:
        #     print('#'*15)
        #     print('#'*15)
        #     print('heatmaps is none in BN')
        #     print('#'*15)
        #     print('#'*15)
        print(heatmaps.shape, self.weight.shape, y.shape, self.bias.shape)
        # print((heatmaps.view(-1, self.weight.shape[0]) @ self.weight.view(-1, 1)).shape)
        print(heatmaps.dtype, self.weight.dtype, self.bias.dtype, y.dtype)
        y = (heatmaps.view(-1, self.weight.shape[0]) @ self.weight.view(-1, 1)) * y + (heatmaps.view(-1, self.weight.shape[0])  @ self.bias.view(-1, 1))
        return y.view(return_shape).transpose(0,1)

class ConvBatchNorm2d(nn.BatchNorm2d):
    # def forward(self, x, gamma=None, beta=None):
    #     self._check_input_dim(x)
    #     y = x.transpose(0, 1)
    #
    #     print(y.shape)
    #     return_shape = y.shape
    #     y = y.contiguous().view(x.size(1), -1)
    #     mu = y.mean(dim=1)
    #     sigma2 = y.var(dim=1)
    #     # y = y - mu.view(-1, 1)
    #     # y = y / (sigma2.view(-1, 1) ** .5 + self.eps)
    #     if self.training is not True:
    #
    #         y = y - self.running_mean.view(-1, 1)
    #         y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
    #     else:
    #         if self.track_running_stats is True:
    #             with torch.no_grad():
    #                 print('ruunig_mean. momentum. mu', self.running_mean.shape, self.momentum, mu.shape)
    #                 print(y.shape, self.running_mean.shape, self.running_var.shape)
    #                 self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
    #                 self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
    #         y = y - mu.view(-1,1)
    #         y = y / (sigma2.view(-1,1)**.5 + self.eps)
    #     y = y * (gamma + 1) + beta
    #     return y.view(return_shape).transpose(0, 1)

    def __init__(self, num_features, momentum=0.9, epsilon=1e-05):
        '''
        input: assume 4D input (mini_batch_size, # channel, w, h)
        momentum: momentum for exponential average
        '''
        super(nn.BatchNorm2d, self).__init__(num_features)
        self.momentum = momentum
        # self.run_mode = 0  # 0: training, 1: testing
        self.insize = num_features
        self.epsilon = epsilon

        # initialize weight(gamma), bias(beta), running mean and variance
        self.register_buffer('running_mean',
                             torch.zeros(self.insize))  # this solves cpu and cuda mismatch location issue
        self.register_buffer('running_var', torch.ones(self.insize))

        # self.running_mean = torch.zeros(self.insize) # torch.zeros(self.insize)
        # self.running_var = torch.ones(self.insize)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, gamma=None, beta=None):
        if self.training is True:
            mean = x.mean([0, 2, 3])  # along channel axis
            var = x.var([0, 2, 3])
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (self.momentum * self.running_mean) + (1.0 - self.momentum) * mean  # .to(input.device)
                    self.running_var = (self.momentum * self.running_var) + (1.0 - self.momentum) * (
                                x.shape[0] / (x.shape[0] - 1) * var)

        else:
            mean = self.running_mean
            var = self.running_var

        current_mean = mean.view([1, self.insize, 1, 1]).expand_as(x)
        current_var = var.view([1, self.insize, 1, 1]).expand_as(x)
        x = x - current_mean
        x = x / ((current_var + self.eps) ** .5)
        # print(x.shape, gamma.shape, beta.shape)

        return x