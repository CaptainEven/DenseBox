# coding=utf-8

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import copy
import re
import torchvision
from torchvision import transforms as T
import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm


# ------------------------------- network

class DenseBox(torch.nn.Module):
    """
    implemention of densebox network with py-torch
    """

    def __init__(self,
                 vgg19):
        """
        init the first 12 layers with pre-trained weights
        :param vgg19: pre-trained net
        """
        super(DenseBox, self).__init__()

        feats = vgg19.features._modules

        # print(feats), print(classifier)

        # ----------------- Conv1
        self.conv1_1_1 = copy.deepcopy(feats['0'])  # (0)
        self.conv1_1_2 = copy.deepcopy(feats['1'])  # (1)
        self.conv1_1 = nn.Sequential(
            self.conv1_1_1,
            self.conv1_1_2
        )  # conv_layer1

        self.conv1_2_1 = copy.deepcopy(feats['2'])  # (2) conv_layer2
        self.conv1_2_2 = copy.deepcopy(feats['3'])  # (3)
        self.conv1_2 = nn.Sequential(
            self.conv1_2_1,
            self.conv1_2_2
        )  # conv_layer2

        self.pool1 = copy.deepcopy(feats['4'])  # (4)

        # ----------------- Conv2
        self.conv2_1_1 = copy.deepcopy(feats['5'])  # (5)
        self.conv2_1_2 = copy.deepcopy(feats['6'])  # (6)
        self.conv2_1 = nn.Sequential(
            self.conv2_1_1,
            self.conv2_1_2
        )  # conv_layer3

        self.conv2_2_1 = copy.deepcopy(feats['7'])  # (7)
        self.conv2_2_2 = copy.deepcopy(feats['8'])  # (8)
        self.conv2_2 = nn.Sequential(
            self.conv2_2_1,
            self.conv2_2_2
        )  # conv_layer4

        self.pool2 = copy.deepcopy(feats['9'])  # (9)

        # ----------------- Conv3
        self.conv3_1_1 = copy.deepcopy(feats['10'])  # (10)
        self.conv3_1_2 = copy.deepcopy(feats['11'])  # (11)
        self.conv3_1 = nn.Sequential(
            self.conv3_1_1,
            self.conv3_1_2
        )  # conv_layer5

        self.conv3_2_1 = copy.deepcopy(feats['12'])  # (12)
        self.conv3_2_2 = copy.deepcopy(feats['13'])  # (13)
        self.conv3_2 = nn.Sequential(
            self.conv3_2_1,
            self.conv3_2_2
        )  # conv_layer6

        self.conv3_3_1 = copy.deepcopy(feats['14'])  # (14)
        self.conv3_3_2 = copy.deepcopy(feats['15'])  # (15)
        self.conv3_3 = nn.Sequential(
            self.conv3_3_1,
            self.conv3_3_2
        )  # conv_layer7

        self.conv3_4_1 = copy.deepcopy(feats['16'])  # (16)
        self.conv3_4_2 = copy.deepcopy(feats['17'])  # (17)
        self.conv3_4 = nn.Sequential(
            self.conv3_4_1,
            self.conv3_4_2
        )  # conv_layer8

        self.pool3 = copy.deepcopy(feats['18'])  # (18)

        # ----------------- Conv4
        self.conv4_1_1 = copy.deepcopy(feats['19'])  # (19)
        self.conv4_1_2 = copy.deepcopy(feats['20'])  # (20)
        self.conv4_1 = nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2
        )  # conv_layer9

        self.conv4_2_1 = copy.deepcopy(feats['21'])  # (21)
        self.conv4_2_2 = copy.deepcopy(feats['22'])  # (22)
        self.conv4_2 = nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2
        )  # conv_layer10

        self.conv4_3_1 = copy.deepcopy(feats['23'])  # (23)
        self.conv4_3_2 = copy.deepcopy(feats['24'])  # (24)
        self.conv4_3 = nn.Sequential(
            self.conv4_3_1,
            self.conv4_3_2
        )  # conv_layer11

        self.conv4_4_1 = copy.deepcopy(feats['25'])  # (25)
        self.conv4_4_2 = copy.deepcopy(feats['26'])  # (26)
        self.conv4_4 = nn.Sequential(
            self.conv4_4_1,
            self.conv4_4_2
        )

        # route: up-sample and concatenate
        # self.up_sampling = nn.Upsample(size=(60, 60),
        #                                mode='bilinear',
        #                                align_corners=True)

        # -------------------------------------- ouput layers
        # scores output
        self.conv5_1_det = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_det = nn.Conv2d(in_channels=512,
                                     out_channels=1,
                                     kernel_size=(1, 1))
        torch.nn.init.xavier_normal_(self.conv5_1_det.weight.data)
        torch.nn.init.xavier_normal_(self.conv5_2_det.weight.data)

        self.output_score = nn.Sequential(
            self.conv5_1_det,
            nn.Dropout(),
            self.conv5_2_det
        )

        # locs output
        self.conv5_1_loc = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_loc = nn.Conv2d(in_channels=512,
                                     out_channels=4,
                                     kernel_size=(1, 1))
        torch.nn.init.xavier_normal_(self.conv5_1_loc.weight.data)
        torch.nn.init.xavier_normal_(self.conv5_2_loc.weight.data)

        self.output_loc = nn.Sequential(
            self.conv5_1_loc,
            nn.Dropout(),
            self.conv5_2_loc
        )

    def forward(self, X):
        """
        :param X:
        :return:
        """
        X = self.conv1_1(X)
        X = self.conv1_2(X)
        X = self.pool1(X)

        X = self.conv2_1(X)
        X = self.conv2_2(X)
        X = self.pool2(X)

        X = self.conv3_1(X)
        X = self.conv3_2(X)
        X = self.conv3_4(X)

        # conv3_4 result
        conv3_4_X = X.clone()
        # conv3_4_X = torch.Tensor.new_tensor(data=X,
        #                                     dtype=torch.float32,
        #                                     device=device,
        #                                     requires_grad=True)

        X = self.pool3(X)

        X = self.conv4_1(X)
        X = self.conv4_2(X)
        X = self.conv4_3(X)
        conv4_4_X = self.conv4_4(X)

        # upsample_X = self.up_sampling
        #  upsample of conv4_4
        conv4_4_X_us = nn.Upsample(size=(conv3_4_X.size(2),
                                         conv3_4_X.size(3)),
                                   mode='bilinear',
                                   align_corners=True)(conv4_4_X)

        # feature fusion: concatenate along channel axis
        fusion = torch.cat((conv4_4_X_us, conv3_4_X), dim=1)
        # print('=> fusion shape', fusion.shape)

        # output layer
        scores = self.output_score(fusion)
        locs = self.output_loc(fusion)
        # print('=> scores shape: ', scores.shape)
        # print('=> locs shape: ', locs.shape)

        return scores, locs


# ------------------------------- DenseBox with landmarks
class DenseBoxLM(torch.nn.Module):
    """
    implemention of densebox network with py-torch
    """

    def __init__(self, vgg19):
        """
        init the first 12 layers with pre-trained weights
        :param vgg19: pre-trained net
        """
        super(DenseBoxLM, self).__init__()

        feats = vgg19.features._modules

        # print(feats), print(classifier)

        # ----------------- Conv1
        self.conv1_1_1 = copy.deepcopy(feats['0'])  # (0)
        self.conv1_1_2 = copy.deepcopy(feats['1'])  # (1)
        self.conv1_1 = nn.Sequential(
            self.conv1_1_1,
            self.conv1_1_2
        )  # conv_layer1

        self.conv1_2_1 = copy.deepcopy(feats['2'])  # (2) conv_layer2
        self.conv1_2_2 = copy.deepcopy(feats['3'])  # (3)
        self.conv1_2 = nn.Sequential(
            self.conv1_2_1,
            self.conv1_2_2
        )  # conv_layer2

        self.pool1 = copy.deepcopy(feats['4'])  # (4)

        # ----------------- Conv2
        self.conv2_1_1 = copy.deepcopy(feats['5'])  # (5)
        self.conv2_1_2 = copy.deepcopy(feats['6'])  # (6)
        self.conv2_1 = nn.Sequential(
            self.conv2_1_1,
            self.conv2_1_2
        )  # conv_layer3

        self.conv2_2_1 = copy.deepcopy(feats['7'])  # (7)
        self.conv2_2_2 = copy.deepcopy(feats['8'])  # (8)
        self.conv2_2 = nn.Sequential(
            self.conv2_2_1,
            self.conv2_2_2
        )  # conv_layer4

        self.pool2 = copy.deepcopy(feats['9'])  # (9)

        # ----------------- Conv3
        self.conv3_1_1 = copy.deepcopy(feats['10'])  # (10)
        self.conv3_1_2 = copy.deepcopy(feats['11'])  # (11)
        self.conv3_1 = nn.Sequential(
            self.conv3_1_1,
            self.conv3_1_2
        )  # conv_layer5

        self.conv3_2_1 = copy.deepcopy(feats['12'])  # (12)
        self.conv3_2_2 = copy.deepcopy(feats['13'])  # (13)
        self.conv3_2 = nn.Sequential(
            self.conv3_2_1,
            self.conv3_2_2
        )  # conv_layer6

        self.conv3_3_1 = copy.deepcopy(feats['14'])  # (14)
        self.conv3_3_2 = copy.deepcopy(feats['15'])  # (15)
        self.conv3_3 = nn.Sequential(
            self.conv3_3_1,
            self.conv3_3_2
        )  # conv_layer7

        self.conv3_4_1 = copy.deepcopy(feats['16'])  # (16)
        self.conv3_4_2 = copy.deepcopy(feats['17'])  # (17)
        self.conv3_4 = nn.Sequential(
            self.conv3_4_1,
            self.conv3_4_2
        )  # conv_layer8

        self.pool3 = copy.deepcopy(feats['18'])  # (18)

        # ----------------- Conv4
        self.conv4_1_1 = copy.deepcopy(feats['19'])  # (19)
        self.conv4_1_2 = copy.deepcopy(feats['20'])  # (20)
        self.conv4_1 = nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2
        )  # conv_layer9

        self.conv4_2_1 = copy.deepcopy(feats['21'])  # (21)
        self.conv4_2_2 = copy.deepcopy(feats['22'])  # (22)
        self.conv4_2 = nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2
        )  # conv_layer10

        self.conv4_3_1 = copy.deepcopy(feats['23'])  # (23)
        self.conv4_3_2 = copy.deepcopy(feats['24'])  # (24)
        self.conv4_3 = nn.Sequential(
            self.conv4_3_1,
            self.conv4_3_2
        )  # conv_layer11

        self.conv4_4_1 = copy.deepcopy(feats['25'])  # (25)
        self.conv4_4_2 = copy.deepcopy(feats['26'])  # (26)
        self.conv4_4 = nn.Sequential(
            self.conv4_4_1,
            self.conv4_4_2
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  dilation=1,
                                  ceil_mode=False)

        # -------------------------------------- ouput layers
        # scores output
        self.conv5_1_det = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_det = nn.Conv2d(in_channels=512,
                                     out_channels=1,
                                     kernel_size=(1, 1))
        nn.init.xavier_normal_(self.conv5_1_det.weight.data)
        nn.init.xavier_normal_(self.conv5_2_det.weight.data)

        self.output_score = nn.Sequential(
            self.conv5_1_det,
            nn.Dropout(),
            self.conv5_2_det
        )

        # locs output
        self.conv5_1_loc = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_loc = nn.Conv2d(in_channels=512,
                                     out_channels=4,
                                     kernel_size=(1, 1))

        nn.init.xavier_normal_(self.conv5_1_loc.weight.data)
        nn.init.xavier_normal_(self.conv5_2_loc.weight.data)

        self.output_loc = nn.Sequential(
            self.conv5_1_loc,
            nn.Dropout(),
            self.conv5_2_loc
        )

        # landmark output
        self.conv5_1_landmark = nn.Conv2d(in_channels=768,
                                          out_channels=512,
                                          kernel_size=(1, 1))
        self.conv5_2_landmark = nn.Conv2d(in_channels=512,
                                          out_channels=4,  # 4 landmarks here
                                          kernel_size=(1, 1))
        nn.init.xavier_normal_(self.conv5_1_landmark.weight.data)
        nn.init.xavier_normal_(self.conv5_2_landmark.weight.data)

        self.output_landmark = nn.Sequential(
            self.conv5_1_landmark,
            nn.Dropout(),
            self.conv5_2_landmark
        )

        # refine branch
        self.conv6_1_det = nn.Conv2d(in_channels=5,  # 1 cls and 4 landmark
                                     out_channels=64,
                                     kernel_size=(3, 3))
        self.conv6_2_det = nn.Conv2d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=(5, 5))
        self.conv6_3_det = nn.Conv2d(in_channels=64,
                                     out_channels=1,
                                     kernel_size=(1, 1))
        nn.init.xavier_normal_(self.conv6_1_det.weight.data)
        nn.init.xavier_normal_(self.conv6_2_det.weight.data)
        nn.init.xavier_normal_(self.conv6_3_det.weight.data)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        X = self.conv1_1(X)
        X = self.conv1_2(X)
        X = self.pool1(X)

        X = self.conv2_1(X)
        X = self.conv2_2(X)
        X = self.pool2(X)

        X = self.conv3_1(X)
        X = self.conv3_2(X)
        X = self.conv3_4(X)

        # conv3_4 result
        conv3_4_X = X.clone()
        # conv3_4_X = torch.Tensor.new_tensor(data=X,
        #                                     dtype=torch.float32,
        #                                     device=device,
        #                                     requires_grad=True)

        X = self.pool3(X)

        X = self.conv4_1(X)
        X = self.conv4_2(X)
        X = self.conv4_3(X)
        conv4_4_X = self.conv4_4(X)

        # upsample_X = self.up_sampling
        #  upsample of conv4_4
        conv4_4_X_ups = nn.Upsample(size=(conv3_4_X.size(2),
                                          conv3_4_X.size(3)),
                                    mode='bilinear',
                                    align_corners=True)(conv4_4_X)

        # feature fusion: concatenate conv_34 and conv4_4 along channel axis
        fusion_1 = torch.cat((conv4_4_X_ups, conv3_4_X), dim=1)
        # print('=> fusion shape', fusion.shape)

        # output scores
        scores = self.output_score(fusion_1)

        # output locs of bbox
        locs = self.output_loc(fusion_1)

        # output landmarks
        landmarks = self.output_landmark(fusion_1)

        # output refined det score
        fusion_2 = torch.cat((landmarks, scores), dim=1)
        X = self.pool4(fusion_2)
        X = self.conv6_1_det(X)
        X = self.conv6_2_det(X)
        X = nn.Upsample(size=(scores.size(2), scores.size(3)),  # default 60×60?
                        mode='bilinear',
                        align_corners=True)(X)
        refine_scores = self.conv6_3_det(X)

        return scores, locs, landmarks, refine_scores


# ------------------------------- dataset
class LPPatchLM_Online(data.Dataset):
    """
    for online label generating
    """

    def __init__(self,
                 root,
                 transform,
                 size=(240, 240)):
        """
        :param root:
        :param transform:
        :param size:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        self.size = size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(self.size),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # init images' path
        self.imgs_path = [root + '/' + x for x in os.listdir(root)]

        self.bboxes = []
        self.vertices = []
        for img_path in self.imgs_path:
            # leftup_x, leftup_y, rightdown_x, right_down_y
            img_name = os.path.split(img_path)[1]
            pattern = '.*_label_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)' \
                      + '_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)' \
                      + '_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).*'
            match = re.match(pattern, img_name)

            # 2 bbox corners
            # turn coordinate to 60×60 coordinate space, float

            bbox_leftup_x = float(match.group(1)) / 4.0
            bbox_leftup_y = float(match.group(2)) / 4.0
            bbox_rightdown_x = float(match.group(3)) / 4.0
            bbox_rightdown_y = float(match.group(4)) / 4.0

            self.bboxes.append(torch.FloatTensor(np.array([
                bbox_leftup_x,
                bbox_leftup_y,
                bbox_rightdown_x,
                bbox_rightdown_y
            ])))

            # 4 vertices: 240×240 coordinate space, float
            # turn coordinate to 60×60 coordinate space, float

            leftup_x = float(match.group(5)) / 4.0
            leftup_y = float(match.group(6)) / 4.0

            rightup_x = float(match.group(7)) / 4.0
            rightup_y = float(match.group(8)) / 4.0

            rightdown_x = float(match.group(9)) / 4.0
            rightdown_y = float(match.group(10)) / 4.0

            leftdown_x = float(match.group(11)) / 4.0
            leftdown_y = float(match.group(12)) / 4.0

            self.vertices.append(torch.FloatTensor(np.array([
                leftup_x,
                leftup_y,  # leftup corner
                rightup_x,
                rightup_y,  # rightup corner
                rightdown_x,
                rightdown_y,  # rightdown corner
                leftdown_x,
                leftdown_y])))  # leftdown corner

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # convert gray to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit or 32bit gray-scale
            img = img.convert('RGB')

        # image data transform
        if self.transform is not None:
            img = self.transform(img)

        return img, self.bboxes[idx], self.vertices[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


class LPPatch_Online(data.Dataset):
    """
    for online label generating
    """

    def __init__(self,
                 root,
                 transform,
                 size=(240, 240)):
        """
        :param root:
        :param transform:
        :param size:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        self.size = size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(self.size),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # init images' path
        self.imgs_path = [root + '/' + x for x in os.listdir(root)]

        self.labels = []
        for img_path in self.imgs_path:
            # leftup_x, leftup_y, rightdown_x, right_down_y
            img_name = os.path.split(img_path)[1]
            match = re.match('.*_label_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                             img_name)

            # 240×240 coordinate space, float, map to output coordinate space
            leftup_x = float(match.group(1)) / 4.0
            leftup_y = float(match.group(2)) / 4.0
            rightdown_x = float(match.group(3)) / 4.0
            rightdown_y = float(match.group(4)) / 4.0

            self.labels.append(torch.FloatTensor(np.array([leftup_x,
                                                           leftup_y,
                                                           rightdown_x,
                                                           rightdown_y])))

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # convert gray to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit or 32bit gray-scale
            img = img.convert('RGB')

        # image data transform
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


class LPPatch_Offline(data.Dataset):
    """
    License plate patch dataset
    """

    def __init__(self,
                 root,
                 transform=None,
                 size=(240, 240)):
        """
        License plate patch
        :param root:
        :param transform:
        :param size:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        # image size
        self.size = size

        # load image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(self.size),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # init images' path
        self.imgs_path = [root + '/' + x for x in os.listdir(root)]

        # ---------------init iamge vertices...
        # read bbox coordinate
        self.label_maps = []
        self.mask_maps = []  # only init part of loss mask map with gray zone
        for img_path in tqdm(self.imgs_path):
            # init each image's label map with zero
            label_map = torch.zeros([5, 60, 60], dtype=torch.float32)

            # init lossmask map with 0
            mask_map = torch.zeros([1, 60, 60], dtype=torch.float32)

            # leftup_x, leftup_y, rightdown_x, right_down_y
            img_name = os.path.split(img_path)[1]
            match = re.match('.*_label_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)',
                             img_name)

            # 240×240 coordinate space, float
            leftup_x = float(match.group(1))
            leftup_y = float(match.group(2))
            rightdown_x = float(match.group(3))
            rightdown_y = float(match.group(4))

            # turn coordinate to 60×60 coordinate space, float
            leftup_x = float(leftup_x) / 4.0  # 240.0 * 60.0
            leftup_y = float(leftup_y) / 4.0  # 240.0 * 60.0
            rightdown_x = float(rightdown_x) / 4.0  # 240.0 * 60.0
            rightdown_y = float(rightdown_y) / 4.0  # 240.0 * 60.0

            # ------------------------- fill label map
            score_map = label_map[0]
            self.init_score_map(score_map=score_map,
                                leftup=(leftup_x, leftup_y),
                                rightdown=(rightdown_x, rightdown_y),
                                ratio=0.3)
            # print(torch.nonzero(score_map))

            dxt_map, dyt_map = label_map[1], label_map[2]
            dxb_map, dyb_map = label_map[3], label_map[4]
            self.init_dist_map(dxt_map=dxt_map,
                               dyt_map=dyt_map,
                               dxb_map=dxb_map,
                               dyb_map=dyb_map,
                               leftup=(leftup_x, leftup_y),
                               rightdown=(rightdown_x, rightdown_y))

            self.label_maps.append(label_map)

            # ------------------------- init loss mask map
            self.init_mask_map(mask_map=mask_map,
                               leftup=(leftup_x, leftup_y),
                               rightdown=(rightdown_x, rightdown_y))
            self.mask_maps.append(mask_map)

    def init_score_map(self,
                       score_map,
                       leftup,
                       rightdown,
                       ratio=0.3):
        """
        init score_map, default 60×60 resolution
        :param score_map:
        :param leftup:
        :param rightdown:
        :param ratio:
        :return:
        """
        # assert score_map.size == torch.Size([60, 60])

        bbox_center_x = float(leftup[0] + rightdown[0]) * 0.5
        bbox_center_y = float(leftup[1] + rightdown[1]) * 0.5
        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        score_map[org_y: end_y + 1, org_x: end_x + 1] = 1.0

        # verify...
        # print(torch.nonzero(score_map))

    def init_mask_map(self,
                      mask_map,
                      leftup,
                      rightdown,
                      ratio=0.3):
        """
        :param mask_map:
        :param leftup:
        :param rightdown:
        :param ratio:
        :return:
        """
        # assert mask_map.size == torch.Size([1, 60, 60])

        bbox_center_x = float(leftup[0] + rightdown[0]) * 0.5
        bbox_center_y = float(leftup[1] + rightdown[1]) * 0.5
        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        mask_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0

    def init_dist_map(self,
                      dxt_map,
                      dyt_map,
                      dxb_map,
                      dyb_map,
                      leftup,
                      rightdown):
        """
        :param dxt_map:
        :param dyt_map:
        :param dxb_map:
        :param dyb_map:
        :param leftup:
        :param rightdown:
        :return:
        """
        # assert dxt_map.size == torch.Size([60, 60])

        bbox_w = rightdown[0] - leftup[0]
        bbox_h = rightdown[1] - leftup[1]

        for y in range(dxt_map.size(0)):  # dim H
            for x in range(dxt_map.size(1)):  # dim W
                dist_xt = (float(x) - leftup[0]) / bbox_w
                dist_yt = (float(y) - leftup[1]) / bbox_h
                dist_xb = (float(x) - rightdown[0]) / bbox_w
                dist_yb = (float(y) - rightdown[1]) / bbox_h

                dxt_map[y, x] = dist_xt
                dyt_map[y, x] = dist_yt
                dxb_map[y, x] = dist_xb
                dyb_map[y, x] = dist_yb

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # convert gray to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # image data transformation
        if self.transform is not None:
            img = self.transform(img)

        return img, self.label_maps[idx], self.mask_maps[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


# ------------------------------- data processing

def pad_img(img):
    """
    padding the image
    :param img: numpy.array
    :return:
    """
    if type(img) != np.ndarray:
        print('=> input image is not numpy array.')
        return

    if len(img.shape) == 2:  # gray-scale
        H, W = img.shape
        dim_diff = np.abs(H - W)

        # left-up padding and right-down padding
        pad_lu = dim_diff // 2
        pad_rd = dim_diff - pad_lu

        padding = ((pad_lu, pad_rd), (0, 0)) if H <= W \
            else ((0, 0), (pad_lu, pad_rd))
    elif len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        H, W, channels = img.shape
        dim_diff = np.abs(H - W)

        # left-up padding and right-down padding
        pad_lu = dim_diff // 2
        pad_rd = dim_diff - pad_lu

        padding = ((pad_lu, pad_rd), (0, 0), (0, 0)) if H <= W \
            else ((0, 0), (pad_lu, pad_rd), (0, 0))

    padded_img = np.pad(img, padding, 'constant', constant_values=128)
    return padded_img


def batch_pad_resize(src_root, dst_root, size=720):
    """
    batch processing of padding and resizing
    """
    if not (os.path.isdir(src_root) and os.path.isdir(dst_root)):
        print('=> [Err]: invalid dir.')
        return

    # clear dst root
    for x in os.listdir(dst_root):
        x_path = dst_root + '/' + x
        if os.path.isfile(x_path):
            os.remove(x_path)

    for img_name in os.listdir(src_root):
        src_path = src_root + '/' + img_name
        if os.path.isfile(src_path):
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            img = pad_img(img)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

            dst_path = dst_root + '/' + img_name
            cv2.imwrite(dst_path, img)
            print('=> %s processed.' % img_name)


# ------------------------------- learning rate

def adjust_LR(optimizer,
              epoch):
    """

    :param optimizer:
    :param epoch:
    :return:
    """
    lr = 1e-8
    if epoch < 5:
        lr = 1e-8
    elif epoch >= 5 and epoch < 10:
        lr = 3e-8
    elif epoch >= 10 and epoch < 15:
        lr = 5e-8
    else:
        lr = 1e-8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def mask_by_sel(loss_mask,
                pos_indices,
                neg_indices):
    """
    cpu side calculation
    :param loss_mask:
    :param pos_indices: N×4dim
    :param neg_indices:
    :return:
    """

    assert loss_mask.size() == torch.Size([loss_mask.size(0), 1, 60, 60])

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))

    for pos_idx in pos_indices:
        loss_mask[pos_idx[0], pos_idx[1], pos_idx[2], pos_idx[3]] = 1.0

    for row in range(neg_indices.size(0)):
        for col in range(neg_indices.size(1)):
            idx = int(neg_indices[row][col])

            if idx < 0 or idx >= 3600:
                # print('=> idx: ', idx)
                continue

            y = idx // 60
            x = idx % 60

            try:
                loss_mask[row, 0, y, x] = 1.0
            except Exception as e:
                print(row, y, x)

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))


# -------------- gray zone processing

def mask_gray_zone_lm_2(loss_mask,
                        vertices,
                        lm_id,
                        gray_border=2.0):
    """
    mask gray zone for landmarks: order may wrong
    :param loss_mask: 
    :param vertices: 
    :param gray_border: 
    :return: 
    """
    batch_size = loss_mask.size(0)
    assert loss_mask.size() == torch.Size([batch_size, 1, 60, 60])
    assert vertices.size() == torch.Size([batch_size, 8])
    assert lm_id == 0 or lm_id == 1 or lm_id == 2 or lm_id == 3

    for item_i, (coords) in enumerate(vertices.numpy()):
        # process each item in the batch
        lm_x = int(vertices[item_i][lm_id * 2] + 0.5)
        lm_y = int(vertices[item_i][lm_id * 2 + 1] + 0.5)

        org_x, org_y = lm_x - 2, lm_y - 2
        end_x, end_y = lm_x + 2, lm_y + 2

        loss_mask[item_i, :, org_y: end_y + 1, org_x: end_x + 1] = 0.0
        loss_mask[item_i, :, lm_y, lm_x] = 1.0


def mask_gray_zone_lm(loss_mask,
                      pos_indices,
                      lm_id,
                      gray_border=2.0):
    """
    mask gray zone for landmarks
    :param loss_mask:
    :param vertices:
    :param gray_border:
    :return:
    """
    batch_size = loss_mask.size(0)
    assert loss_mask.size() == torch.Size([batch_size, 1, 60, 60])
    assert pos_indices.size() == torch.Size([batch_size, 4])
    assert lm_id == 0 or lm_id == 1 or lm_id == 2 or lm_id == 3

    for item_i, pos_idx in enumerate(pos_indices):
        lm_x, lm_y = int(pos_idx[3]), int(pos_idx[2])

        org_x, org_y = lm_x - 2, lm_y - 2
        end_x, end_y = lm_x + 2, lm_y + 2

        loss_mask[item_i, :, org_y: end_y + 1, org_x: end_x + 1] = 0.0
        loss_mask[item_i, :, lm_y, lm_x] = 1.0


def mask_gray_zone_score(loss_mask,
                         label,
                         ratio=0.3,
                         gray_border=2.0):
    """
    only used in online training mode for now
    :param loss_mask:
    process a batch                 0         1          2            3
    :param label:  batch_size×4: leftup_x, leftup_y, rightdown_x, rightdown_y
    :param ratio:
    :param gray_border: gray zone border width
    :return:
    """
    assert loss_mask.size(1) == 1 \
           and loss_mask.size(2) == 60 \
           and loss_mask.size(3) == 60

    # assert label.size() == torch.Size([label.size(0), 4])

    for item_i, coord in enumerate(label.numpy()):
        # process each item in the batch
        bbox_center_x = float(coord[0] + coord[2]) * 0.5
        bbox_center_y = float(coord[1] + coord[3]) * 0.5

        bbox_w = coord[2] - coord[0]
        bbox_h = coord[3] - coord[1]

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5)
                    - gray_border + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5)
                    - gray_border + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + gray_border + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + gray_border + 0.5)

        # fill gray zone with 0
        loss_mask[item_i, 0, org_y: end_y, org_x: end_x] = 0.0


def init_score_map(label,
                   batch_size,
                   ratio=0.3):
    """
    process a batch                 0         1          2            3
    :param label:  batch_size×4: leftup_x, leftup_y, rightdown_x, rightdown_y
    :param batch_size:
    :param ratio:
    :return:
    """
    # assert label.size() == torch.Size([label.size(0), 4])

    score_map = torch.zeros([batch_size, 1, 60, 60], dtype=torch.float32)

    for item_i, coord in enumerate(label.numpy()):
        # process each item in the batch
        bbox_center_x = float(coord[0] + coord[2]) * 0.5
        bbox_center_y = float(coord[1] + coord[3]) * 0.5

        bbox_w = coord[2] - coord[0]
        bbox_h = coord[3] - coord[1]

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        score_map[item_i, :, org_y: end_y + 1, org_x: end_x + 1] = 1.0

    return score_map


def init_loc_map(label,
                 batch_size):
    """
                                    0         1          2            3
    :param label:  batch_size×4: leftup_x, leftup_y, rightdown_x, rightdown_y
    :param label:
    :param batch_size:
    :return:
    """
    # assert label.size() == torch.Size([batch_size, 4])

    loc_map = torch.zeros([batch_size, 4, 60, 60], dtype=torch.float32)

    # norm = 50.0 / 4.0

    for item_i, coord in enumerate(label.numpy()):
        # process each item in the batch
        bbox_w = coord[2] - coord[0]
        bbox_h = coord[3] - coord[1]

        for y in range(60):  # dim H
            for x in range(60):  # dim W
                dist_xt = (float(x) - coord[0])  # / norm  # bbox_w
                dist_yt = (float(y) - coord[1])  # / norm  # bbox_h
                dist_xb = (float(x) - coord[2])  # / norm  # bbox_w
                dist_yb = (float(y) - coord[3])  # / norm  # bbox_h

                loc_map[item_i, 0, y, x] = dist_xt
                loc_map[item_i, 1, y, x] = dist_yt
                loc_map[item_i, 2, y, x] = dist_xb
                loc_map[item_i, 3, y, x] = dist_yb

    return loc_map


def init_landmark_map(vertices,
                      batch_size):
    """
    init landmark gt maps: N×4×H×W
    :param vertices:
    :param batch_size:
    :return:
    """
    # assert vertices.size() == torch.Size([vertices.size(0), 4])

    landmark_map = torch.zeros([batch_size, 4, 60, 60], dtype=torch.float32)

    for item_i, coord in enumerate(vertices.numpy()):
        leftup_x, leftup_y = int(coord[0] + 0.5), int(coord[1] + 0.5)
        rightup_x, rightup_y = int(coord[2] + 0.5), int(coord[3] + 0.5)
        rightdown_x, rightdown_y = int(coord[4] + 0.5), int(coord[5] + 0.5)
        leftdown_X, leftdown_y = int(coord[6] + 0.5), int(coord[7] + 0.5)

        landmark_map[item_i, 0, leftup_y, leftup_x] = 1.0
        landmark_map[item_i, 1, rightup_y, rightup_x] = 1.0
        landmark_map[item_i, 2, rightdown_y, rightdown_x] = 1.0
        landmark_map[item_i, 3, leftdown_y, leftdown_X] = 1.0

    return landmark_map


def gen_neg_loss(loss_orig, map_gt):
    """
    gen negative sample loss...
    :param loss_orig:
    :param map_gt:
    :return:
    """

    assert loss_orig.size() == map_gt.size() \
           == torch.Size([loss_orig.size(0), 1, 60, 60])

    ones_mask = torch.ones([loss_orig.size(0), 1, 60, 60],
                           dtype=torch.float32).to(device)
    neg_mask = ones_mask - map_gt  # element-wise substraction
    neg_loss = loss_orig * neg_mask

    return neg_loss


# ------------------------------- model weights save path

tmp_model_path = './checkpoints/DenseBox_tmp.pth'
final_model_path = './checkpoints/DenseBox_final.pth'
tmp_lm_model_path = './checkpoints/DenseBox_lm_tmp.pth'
final_lm_model_path = './checkpoints/DenseBox_lm_final.pth'


# ------------------------------- training
def train_LM_online(src_root,
                    dst_root,
                    num_epoch=30,
                    lambda_loc=3.0,
                    lambda_det=1.0,
                    lambda_lm=0.5,
                    base_lr=1e-8,
                    batch_size=5,
                    resume=None,
                    is_test=False):
    """
    online label generating and training
    :param src_root:
    :param dst_root:
    :param num_epoch:
    :param lambda_det:
    :param lambda_lm:
    :param base_lr:
    :param batch_size:
    :param resume:
    :param is_test:
    :return:
    """
    train_set = LPPatchLM_Online(root=src_root,
                                 transform=None,
                                 size=(240, 240))

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    # network
    vgg19_pretrain = torchvision.models.vgg19(pretrained=True)
    net = DenseBoxLM(vgg19=vgg19_pretrain).to(device)
    print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- loss functions
    # element-wise L2 loss function
    loss_func = nn.MSELoss(reduce=False).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4 or 5e-8

    # net start to train in train mode
    print('\nTraining...')
    net.train()

    print('=> base learning rate: ', base_lr)

    for epoch_i in range(num_epoch):
        # train each epoch
        epoch_loss = []

        for batch_i, (data, bbox, vertices) in enumerate(train_loader):
            # label in the output coordinate space(60×60)

            data = data.to(device)

            # ----------------- init label maps
            # init score map: N×1×60×60
            cls_map_gt = init_score_map(label=bbox,
                                        batch_size=data.size(0),
                                        ratio=0.3)

            # init score loss mask
            loss_mask_cls = cls_map_gt.clone()

            # init loc map: N×4×60×60
            loc_map_gt = init_loc_map(label=bbox,
                                      batch_size=data.size(0))

            # init landmark map: N×4×60×60
            landmark_map_gt = init_landmark_map(vertices=vertices,
                                                batch_size=data.size(0))

            # init landmark mask
            loss_mask_lm = landmark_map_gt.clone()

            cls_map_gt = cls_map_gt.to(device)
            loc_map_gt = loc_map_gt.to(device)
            landmark_map_gt = landmark_map_gt.to(device)

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass
            score_out, loc_out, lm_out, rf_score_out = net.forward(data)

            # ------------- loss calculation
            # classification(score) loss
            cls_loss = loss_func(score_out, cls_map_gt)
            rf_loss = loss_func(rf_score_out, cls_map_gt)

            # loc loss should be masked by label scores and to be summed
            loc_loss = loss_func(loc_out, loc_map_gt)

            # landmark loss
            landmark_loss = loss_func(lm_out, landmark_map_gt)

            # ------------- negative mining and mask for scores
            # positive and negative sample number
            pos_indices = torch.nonzero(cls_map_gt)
            positive_num = pos_indices.size(0)

            # to keep the ratio of positive and negative sample to 1
            neg_num = int(float(positive_num) / float(data.size(0)) + 0.5)
            ones_mask = torch.ones([data.size(0), 1, 60, 60],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - cls_map_gt

            neg_cls_loss = cls_loss * neg_mask

            half_neg_num = int(neg_num * 0.5 + 0.5)
            neg_cls_loss = neg_cls_loss.view(data.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=neg_cls_loss,
                                                     k=half_neg_num,
                                                     dim=1)

            rand_neg_indices = torch.zeros([data.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            for i in range(data.size(0)):
                indices = np.random.choice(3600,  # 60 * 60,
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices

            # concatenate negative sample ids
            neg_indices = torch.cat((hard_neg_indices,
                                     rand_neg_indices),
                                    dim=1)

            neg_indices = neg_indices.cpu()
            pos_indices = pos_indices.cpu()

            # fill the loss mask
            mask_by_sel(loss_mask=loss_mask_cls,
                        pos_indices=pos_indices,
                        neg_indices=neg_indices)
            mask_gray_zone_score(loss_mask=loss_mask_cls,
                                 label=bbox,
                                 ratio=0.3,
                                 gray_border=2.0)

            # ------------- negative mining and mask for each landmark
            for lm_i in range(4):
                # fetch each landmark and do processing
                lm_i_map_gt = landmark_map_gt[:, lm_i, :, :].unsqueeze(1)
                lm_i_loss = landmark_loss[:, lm_i, :, :].unsqueeze(1)
                lm_i_loss_mask = loss_mask_lm[:, lm_i, :, :].unsqueeze(1)

                lm_i_neg_loss = gen_neg_loss(loss_orig=lm_i_loss,
                                             map_gt=lm_i_map_gt)

                lm_i_pos_indices = torch.nonzero(lm_i_map_gt)
                # lm_i_pos_num = lm_i_pos_indices.size(0)

                lm_i_neg_loss = lm_i_neg_loss.view(data.size(0), -1)
                lm_i_hard_negs, lm_i_hard_neg_indices = torch.topk(input=lm_i_neg_loss,
                                                                   k=1,
                                                                   dim=1)
                lm_i_rand_neg_indices = torch.zeros([data.size(0), 1],
                                                    dtype=torch.long).to(device)
                for item_i in range(data.size(0)):
                    indices = np.random.choice(3600,  # 60 * 60,
                                               1,
                                               replace=False)
                    indices = torch.Tensor(indices)
                    lm_i_rand_neg_indices[item_i] = indices

                # concatenate negative sample ids
                lm_i_neg_indices = torch.cat((lm_i_hard_neg_indices,
                                              lm_i_rand_neg_indices),
                                             dim=1)

                lm_i_neg_indices = lm_i_neg_indices.cpu()
                lm_i_pos_indices = lm_i_pos_indices.cpu()

                # fill loss mask of landmark
                mask_by_sel(loss_mask=lm_i_loss_mask,
                            pos_indices=lm_i_pos_indices,
                            neg_indices=lm_i_neg_indices)  # fill mask by select

                mask_gray_zone_lm(loss_mask=lm_i_loss_mask,
                                  pos_indices=lm_i_pos_indices,
                                  lm_id=lm_i,
                                  gray_border=2.0)
            # ---------------------------------------

            # ------------- calculate final loss
            loss_mask_cls = loss_mask_cls.to(device)
            loss_mask_lm = loss_mask_lm.to(device)

            # detection loss
            mask_cls_loss = loss_mask_cls * cls_loss
            mask_loc_loss = loss_mask_cls * cls_map_gt * loc_loss
            det_loss = lambda_det * (torch.sum(mask_cls_loss)
                                     + lambda_loc * torch.sum(mask_loc_loss))

            # landmark loss
            mask_lm_loss = loss_mask_lm * landmark_loss
            lm_loss = lambda_lm * torch.sum(mask_lm_loss)

            # refine loss
            rf_mask_cls_loss = loss_mask_cls * rf_loss
            rf_loss = torch.sum(rf_mask_cls_loss)

            # full loss calculation
            full_loss = det_loss + lm_loss + rf_loss

            # ------------- collect batch loss for the epoch
            epoch_loss.append(full_loss.item())

            # ------------- back propagation
            full_loss.backward()
            optimizer.step()

            # ------------- loss log
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              full_loss.item()))

        # print average loss of the epoch
        print('=> epoch %d average loss: %.3f'
              % (epoch_i + 1, sum(epoch_loss) / len(epoch_loss)))

        # ------------ save checkpoint
        torch.save(net.state_dict(), tmp_lm_model_path)
        print('<= {} saved.'.format(tmp_lm_model_path))

        # ------------- test this epoch
        if is_test:
            test_lm(src_root=src_root,
                    dst_root=dst_root,
                    resume=resume,
                    valid_ratio=0.005)
            print('=> epoch %d tested.\n' % (epoch_i + 1))

        # ------------- adjust learning for next epoch
        lr = adjust_LR(optimizer=optimizer,
                       epoch=epoch_i)
        print('=> applying learning rate: ', lr)

    torch.save(net.state_dict(), final_lm_model_path)
    print('<= {} saved.\n'.format(final_lm_model_path))


def train_online(src_root,
                 dst_root,
                 num_epoch=30,
                 lambda_loc=3.0,
                 base_lr=1e-8,
                 batch_size=5,
                 resume=None,
                 is_test=False):
    """
    online label generating...
    :param src_root:
    :param dst_root:
    :param num_epoch:
    :param lambda_loc:
    :param base_lr:
    :param batch_size:
    :param resume:
    :param is_test:
    :return:
    """
    train_set = LPPatch_Online(root=src_root,
                               transform=None,
                               size=(240, 240))
    # train_set = LPPatchLM_Online(root=src_root,
    #                              transform=None,
    #                              size=(240, 240))

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    # network
    vgg19_pretrain = torchvision.models.vgg19(pretrained=True)
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)
    else:
        print('=> [Note]: none resume, train from scratch.')

    # ---------------- loss functions
    # element-wise L2 loss function
    loss_func = nn.MSELoss(reduce=False).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4

    # net start to train in train mode
    print('\nTraining...')
    net.train()

    print('=> base learning rate: ', base_lr)

    for epoch_i in range(num_epoch):
        # train each epoch
        epoch_loss = []

        for batch_i, (data, label) in enumerate(train_loader):
            # label in the output coordinate space(60×60)

            data = data.to(device)

            # ----------------- init label maps
            # init score map: N×1×60×60
            score_map_gt = init_score_map(label=label,
                                          batch_size=data.size(0),
                                          ratio=0.3)

            # init loss mask
            loss_mask = score_map_gt.clone()

            # init loc map: N×4×60×60
            loc_map_gt = init_loc_map(label=label,
                                      batch_size=data.size(0))

            score_map_gt, loc_map_gt = score_map_gt.to(
                device), loc_map_gt.to(device)

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass
            score_out, loc_out = net.forward(data)

            # ------------- loss calculation with hard negative mining
            positive_indices = torch.nonzero(score_map_gt)
            positive_num = positive_indices.size(0)

            # to keep the ratio of positive and negative sample to 1
            negative_num = int(float(positive_num) / float(data.size(0)) + 0.5)
            score_loss = loss_func(score_out, score_map_gt)

            # loc loss should be masked by label scores and to be summed
            loc_loss = loss_func(loc_out, loc_map_gt)

            # negative sampling
            ones_mask = torch.ones([data.size(0), 1, 60, 60],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - score_map_gt
            negative_score_loss = score_loss * neg_mask

            half_neg_num = int(negative_num * 0.5 + 0.5)
            negative_score_loss = negative_score_loss.view(data.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=negative_score_loss,
                                                     k=half_neg_num,
                                                     dim=1)

            rand_neg_indices = torch.zeros([data.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            for i in range(data.size(0)):
                indices = np.random.choice(3600,  # 60 * 60,
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices

            # concatenate negative sample ids
            neg_indices = torch.cat(
                (hard_neg_indices, rand_neg_indices), dim=1)

            neg_indices = neg_indices.cpu()
            positive_indices = positive_indices.cpu()

            # fill the loss mask
            mask_by_sel(loss_mask=loss_mask,
                        pos_indices=positive_indices,
                        neg_indices=neg_indices)
            mask_gray_zone_score(loss_mask=loss_mask,
                                 label=label,
                                 ratio=0.3,
                                 gray_border=2.0)

            # ------------- calculate final loss
            loss_mask = loss_mask.to(device)

            mask_score_loss = loss_mask * score_loss
            mask_loc_loss = loss_mask * score_map_gt * loc_loss

            loss = torch.sum(mask_score_loss) \
                   + torch.sum(lambda_loc * mask_loc_loss)  # / 3600.0
            # loss = torch.mean(mask_score_loss) \
            #        + torch.mean(lambda_loc * mask_loc_loss)

            epoch_loss.append(loss.item())

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            # ------------- print loss
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item()))

        # print average loss of the epoch
        print('=> epoch %d average loss: %.3f'
              % (epoch_i + 1, sum(epoch_loss) / len(epoch_loss)))

        # ------------ save checkpoint
        torch.save(net.state_dict(), tmp_model_path)
        print('<= {} saved.'.format(tmp_model_path))

        # ------------- test this epoch
        if is_test:
            test(src_root=src_root,
                 dst_root=dst_root,
                 resume=resume,
                 valid_ratio=0.003)
            print('=> epoch %d tested.\n' % (epoch_i + 1))

        # ------------- adjust learning after this epoch
        lr = adjust_LR(optimizer=optimizer,
                       epoch=epoch_i)
        print('=> applying learning rate: ', lr)

    torch.save(net.state_dict(), final_model_path)
    print('<= {} saved.\n'.format(final_model_path))


def train_offline(num_epoch=30,
                  lambda_loc=3.0,
                  base_lr=1e-5,
                  resume=None):
    """
    generating vertices offline(pre-load to main memory)
    :param num_epoch:
    :param lambda_loc:
    :param resume:
    :return:
    """
    train_set = LPPatch_Offline(root='/mnt/diskc/even/patch',
                                transform=None,
                                size=(240, 240))

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    # network
    vgg19_pretrain = torchvision.models.vgg19(pretrained=True)
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- loss functions
    # element-wise L2 loss
    loss_func = nn.MSELoss(reduce=False).to(device)

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4

    # net start to train in train mode
    print('\nTraining...')
    net.train()

    for epoch_i in range(num_epoch):
        # adjust learning before this epoch
        lr = adjust_LR(optimizer=optimizer,
                       epoch=epoch_i)
        print('=> learning rate: ', lr)

        for batch_i, (data, label_map, loss_mask) in enumerate(train_loader):
            # ------------- put data to device
            data, label_map = data.to(device), label_map.to(device)

            # loss_mask = loss_mask.unsqueeze(1)

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass
            score_out, loc_out = net.forward(data)

            # ------------- loss calculation with hard negative mining
            score_map_gt = label_map[:, 0]
            score_map_gt = score_map_gt.unsqueeze(1)
            loc_map_gt = label_map[:, 1:]

            positive_indices = torch.nonzero(score_map_gt)
            positive_num = positive_indices.size(0)

            # to keep the ratio of positive and negative sample to 1
            negative_num = int(float(positive_num) / float(data.size(0)) + 0.5)
            score_loss = loss_func(score_out, score_map_gt)

            # loc loss should be masked by label scores and to be summed
            loc_loss = loss_func(loc_out, loc_map_gt)

            # negative smapling... debug
            ones_mask = torch.ones([data.size(0), 1, 60, 60],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - score_map_gt
            negative_score_loss = score_loss * neg_mask
            # print('=> neg pix numner: ', torch.nonzero(negative_score_loss).size(0))

            half_neg_num = int(negative_num * 0.5 + 0.5)
            negative_score_loss = negative_score_loss.view(data.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=negative_score_loss,
                                                     k=half_neg_num,
                                                     dim=1)

            rand_neg_indices = torch.zeros([data.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            for i in range(data.size(0)):
                indices = np.random.choice(3600,  # 60 * 60
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices

            # concatenate negative sample ids
            neg_indices = torch.cat((hard_neg_indices, rand_neg_indices), dim=1)

            neg_indices = neg_indices.cpu()
            positive_indices = positive_indices.cpu()

            # fill the loss mask
            mask_by_sel(loss_mask=loss_mask,
                        pos_indices=positive_indices,
                        neg_indices=neg_indices)

            # ------------- calculate final loss
            loss_mask = loss_mask.to(device)

            mask_score_loss = loss_mask * score_loss
            mask_loc_loss = loss_mask * score_map_gt * loc_loss

            loss = torch.sum(mask_score_loss) \
                   + torch.sum(lambda_loc * mask_loc_loss)

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            # ------------- print loss
            iter_count = epoch_i * len(train_loader) + batch_i
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>3d}/{:>3d}'
                      ', total_iter {:>5d} '
                      '| loss {:>5.3f}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item()))

        # ------------ save checkpoint
        torch.save(net.state_dict(), tmp_model_path)
        print('<= {} saved.\n'.format(tmp_model_path))

    torch.save(net.state_dict(), final_model_path)
    print('<= {} saved.\n'.format(final_model_path))


# ------------------------------- validating and testing

def parse_DetLM(score_map,
                loc_map,
                lm_map,
                M,
                N,
                K=10):
    """
    parse output(including landmarks)
    from arbitrary input image size M×N
    M: image height, M rows
    N: image width, N cols
    :param score_map:
    :param loc_map:
    :param lm_map:
    :param M:
    :param N:
    :param K:
    :return:
    """
    assert score_map.size() == torch.Size([1, 1, M // 4, N // 4])  # N×C×H×W
    assert loc_map.size() == torch.Size([1, 4, M // 4, N // 4])  # 4 coordinates
    assert lm_map.size() == torch.Size([1, 4, M // 4, N // 4])  # 4 landmarks

    # squeeze output
    score_map = score_map.squeeze()
    loc_map = loc_map.squeeze()
    lm_map = lm_map.squeeze()

    # reshape output, score_map: 1×(M×N), loc_map:4×(M×N), lm_map:4×(M×N)
    score_map = score_map.view(1, -1)
    loc_map = loc_map.view(4, -1)
    lm_map = lm_map.view(4, -1)

    # filter out top k bbox with highest score
    scores, indices = torch.topk(input=score_map,
                                 k=K,
                                 dim=1)

    indices = indices.squeeze()
    score_map = score_map.squeeze().data

    dets = []
    cols_out = N // 4  # cols in output coordinate space
    for idx in indices:
        idx = int(idx)

        # point location on the output coordinate space
        xi = idx % cols_out
        yi = idx // cols_out

        # parse bbox 2 corners: (xt, yt) and (xb, yb)
        xt = xi - loc_map[0, idx]
        yt = yi - loc_map[1, idx]
        xb = xi - loc_map[2, idx]
        yb = yi - loc_map[3, idx]

        # map back to input coordinate space
        xt = float(xt.data) * 4.0
        yt = float(yt.data) * 4.0
        xb = float(xb.data) * 4.0
        yb = float(yb.data) * 4.0

        # process each landmark
        landmarks = []
        for lm_i in range(4):  # 4 landmarks
            landmark_map = lm_map[lm_i]
            lm_scores, lm_idx = torch.topk(input=landmark_map, k=1)
            lm_idx = int(lm_idx)

            x_lm = float(lm_idx % cols_out) * 4.0
            y_lm = float(lm_idx // cols_out) * 4.0

            landmarks.extend([x_lm, y_lm])

        det = [xt, yt, xb, yb, float(score_map[idx])]
        det += landmarks
        dets.append(det)

    return np.array(dets)


def parse_out_MN(score_map,
                 loc_map,
                 M,
                 N,
                 K=10):
    """
    parse output from arbitrary input image size M×N
    M: image height, M rows
    N: image width, N cols
    """
    assert score_map.size() == torch.Size([1, 1, M // 4, N // 4])  # N×C×H×W
    assert loc_map.size() == torch.Size([1, 4, M // 4, N // 4])

    # squeeze output
    score_map, loc_map = score_map.squeeze(), loc_map.squeeze()

    # reshape output, score_map: 1×(M×N), loc_map:4×(M×N)
    score_map, loc_map = score_map.view(1, -1), loc_map.view(4, -1)

    # filter out top k bbox with highest score
    scores, indices = torch.topk(input=score_map,
                                 k=K,
                                 dim=1)

    indices = indices.squeeze()
    score_map = score_map.squeeze().data

    dets = []
    cols_out = N // 4  # cols in output coordinate space
    for idx in indices:
        idx = int(idx)
        xi, yi = idx % cols_out, idx // cols_out

        xt = xi - loc_map[0, idx]
        yt = yi - loc_map[1, idx]
        xb = xi - loc_map[2, idx]
        yb = yi - loc_map[3, idx]

        # map back to input coordinate space
        xt = float(xt.data) * 4.0
        yt = float(yt.data) * 4.0
        xb = float(xb.data) * 4.0
        yb = float(yb.data) * 4.0

        det = [xt, yt, xb, yb, float(score_map[idx])]
        dets.append(det)

    return np.array(dets)


def parse_output(score_map,
                 loc_map,
                 K=10):
    """
    process batch_size == 1
    parse bbox with score for each pixel
    in the output coordinate space(default: 60×60)
    :param score_map:
    :param loc_map:
    :param K:
    :return: dets: all bounding boxes: x1, y1, x2, y2, score
    """
    assert score_map.size() == torch.Size([1, 1, 60, 60])  # batch_size == 1
    assert loc_map.size() == torch.Size([1, 4, 60, 60])

    # top k bbox with highest score
    # norm = 50.0 / 4.0
    dets = []

    score_map, loc_map = score_map.squeeze(), loc_map.squeeze()
    score_map, loc_map = score_map.view(1, -1), loc_map.view(4, -1)
    scores, indices = torch.topk(input=score_map,
                                 k=K,
                                 dim=1)

    indices = indices.squeeze()
    score_map = score_map.squeeze().data
    for idx in indices:
        idx = int(idx)
        xi, yi = idx % 60, idx // 60

        xt = xi - loc_map[0, idx]
        yt = yi - loc_map[1, idx]
        xb = xi - loc_map[2, idx]
        yb = yi - loc_map[3, idx]

        xt = float(xt.data) * 4.0  # * norm
        yt = float(yt.data) * 4.0  # * norm
        xb = float(xb.data) * 4.0  # * norm
        yb = float(yb.data) * 4.0  # * norm

        det = [xt, yt, xb, yb, float(score_map[idx])]
        dets.append(det)

    return np.array(dets)


def NMS(dets,
        nms_thresh=0.4):
    """
    Pure Python NMS baseline
    :param dets:
    :param nms_thresh:
    :return:
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= nms_thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def viz_result(img_path,
               dets,
               dst_root):
    """
    :param dets: final bounding boxes: x1, y1, x2, y2, score
    :param img_path:
    :param dets:
    :param dst_root:
    :return:
    """
    if not os.path.isfile(img_path):
        print('=> [Err]: invalid img file.')
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # draw each bbox
    color = (0, 215, 255)
    for det in dets:
        # bbox corner and score
        pt_1 = (int(det[0] + 0.5), int(det[1] + 0.5))
        pt_2 = (int(det[2] + 0.5), int(det[3] + 0.5))
        score = str('%.3f' % (det[4]))

        # draw bbox
        cv2.rectangle(img=img, pt1=pt_1, pt2=pt_2, color=color, thickness=2)

        # compute score txt size
        txt_size = cv2.getTextSize(text=score,
                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                   fontScale=2,
                                   thickness=2)[0]

        # draw text background rect
        pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] - txt_size[1] - 5
        cv2.rectangle(img=img,
                      pt1=pt_1,
                      pt2=pt_2,
                      color=color,
                      thickness=-1)  # fill rectangle

        # draw text
        cv2.putText(img=img,
                    text=score,
                    org=(pt_1[0], pt_1[1]),  # pt_1[1] + txt_size[1] + 4
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=[225, 255, 255],
                    thickness=2)

        # draw landmarks
        if len(det) == 13:
            # format output landmark order?
            for pt_idx in range(4):
                cv2.circle(img=img,
                           center=(int(det[5 + pt_idx * 2]), int(det[5 + pt_idx * 2 + 1])),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)

    # write img to dst_root
    img_name = os.path.split(img_path)[1]
    dst_path = dst_root + '/' + img_name
    cv2.imwrite(dst_path, img)
    # print('=> %s processed done.' % img_name)


# ------------------------------- testing
def test_lm(src_root,
            dst_root,
            resume=None,
            valid_ratio=0.05):
    """
    :param root:
    :return:
    """
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid root.')
        return

    # ---------------- network
    vgg19_pretrain = torchvision.models.vgg19(pretrained=True)

    net = DenseBoxLM(vgg19=vgg19_pretrain)
    # print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- image
    imgs_path = [src_root + '/' + x for x in os.listdir(src_root)]

    # ---------------- clear dst root
    for img_name in os.listdir(dst_root):
        img_path = dst_root + '/' + img_name
        os.remove(img_path)

    # ---------------- inference mode
    net.eval()
    net.to(device)

    # ---------------- inference
    for img_path in imgs_path:
        if random.random() < valid_ratio:
            if os.path.isfile(img_path):
                # load image data and transform image
                img = Image.open(img_path)
                W, H = img.size

                # ---------------- transform
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(H, W)),
                    torchvision.transforms.CenterCrop(size=(H, W)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
                img = transform(img)
                img = img.view(1, 3, H, W)

                # inference on gpu side
                img = img.to(device)
                score_out, loc_out, lm_out, rf_score_out = net.forward(img)

                # parse output on the cpu side...使用score还是rf_score更好?
                dets = parse_DetLM(score_map=rf_score_out.cpu(),
                                   loc_map=loc_out.cpu(),
                                   lm_map=lm_out.cpu(),
                                   M=H,
                                   N=W,
                                   K=10)

                # do non-maximum suppression
                keep = NMS(dets=dets, nms_thresh=0.4)  # 0.4
                dets = dets[keep]

                # visualize final results
                viz_result(img_path=img_path,
                           dets=dets,
                           dst_root=dst_root)


def test(src_root,
         dst_root,
         resume=None,
         valid_ratio=0.05):
    """
    :param root:
    :return:
    """
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid root.')
        return

    # ---------------- network
    vgg19_pretrain = torchvision.models.vgg19(pretrained=True)

    net = DenseBox(vgg19=vgg19_pretrain)
    # print('=> net:\n', net)

    # ---------------- whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)

    # ---------------- image
    imgs_path = [src_root + '/' + x for x in os.listdir(src_root)]

    # ---------------- clear dst root
    for img_name in os.listdir(dst_root):
        img_path = dst_root + '/' + img_name
        os.remove(img_path)

    # ---------------- inference mode
    net.eval()
    net.to(device)

    # ---------------- inference
    for img_path in imgs_path:
        if random.random() < valid_ratio:
            if os.path.isfile(img_path):
                # load image data and transform image
                img = Image.open(img_path)
                W, H = img.size

                # ---------------- transform
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(H, W)),
                    torchvision.transforms.CenterCrop(size=(H, W)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
                img = transform(img)
                img = img.view(1, 3, H, W)

                # inference on gpu side
                img = img.to(device)
                score_out, loc_out = net.forward(img)

                # parse output on cpu side
                dets = parse_out_MN(score_map=score_out.cpu(),
                                    loc_map=loc_out.cpu(),
                                    M=H,
                                    N=W,
                                    K=10)

                # do non-maximum suppression
                keep = NMS(dets=dets, nms_thresh=0.4)
                dets = dets[keep]

                # visualize final results
                viz_result(img_path=img_path,
                           dets=dets,
                           dst_root=dst_root)


def train_test(src_root,
               dst_root,
               is_online=True):
    """
    :param src_root:

    :param dst_root:

    :param is_online:
    :return:
    """
    if is_online:
        train_LM_online(src_root=src_root,
                        dst_root=dst_root,
                        num_epoch=100,
                        lambda_loc=3.0,
                        lambda_det=1.0,
                        lambda_lm=0.5,
                        base_lr=1e-8,
                        batch_size=10,
                        resume=tmp_lm_model_path,
                        is_test=True)

        # train_online(src_root=src_root,
        #              dst_root=dst_root,
        #              num_epoch=30,
        #              lambda_loc=3.0,
        #              base_lr=1e-8,
        #              batch_size=10,
        #              resume=tmp_model_path,
        #              is_test=True)

    else:
        train_offline(num_epoch=100,
                      lambda_loc=3.0,
                      resume=tmp_model_path)


if __name__ == '__main__':
    # ---------------- network testing
    # test_net()

    # --------------- training
    src_root = '/mnt/diskc/even/patchLM'
    # src_root = '/mnt/diskc/even/patch'
    dst_root = '/mnt/diskc/even/test_result'

    train_test(src_root=src_root,
               dst_root=dst_root,
               is_online=True)

    # --------------- validating
    # test(src_root=src_root,
    #      dst_root=dst_root,
    #      resume=final_model_path,
    #      valid_ratio=0.003)

    # test_lm(src_root=src_root,
    #         dst_root=dst_root,
    #         resume=tmp_lm_model_path,
    #         valid_ratio=5e-3)

    # --------------- data pre-processing
    # batch_pad_resize(src_root='/mnt/diskc/even/plate_data_pro/JPEGImages',
    #                  dst_root='/mnt/diskc/even/test_set',
    #                  size=768)
