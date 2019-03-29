# coding: utf-8

import os
import random
import shutil
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
import math

from PIL import Image, ImageDraw
from tqdm import tqdm

from torchvision import transforms as T
from models import LP_Net


# 类别名称
classes = ['Plate']
# classes = ['car', 'Car']
# plate_models = [u'单', u'双']
# plate_colors = [u'蓝', u'黄', u'白', u'黑', u'绿']

# 输出文件名格式
# count_id = 0
type_name = 'det'
date_name = time.strftime('_%Y_%m_%d_', time.localtime(time.time()))
print('date_name: ', date_name)
# id_name = '{:0>6d}'.format(count_id)

print(type_name, date_name, '{:0>6d}'.format(0))


# 将原始数据目录分割为3个子目录
def move_rename(origin_dir, img_path, xml_path):
    """
    :param origin_dir:  图像原始路径
    :param img_path: 图像新路径
    :param xml_path: xml文件新路径
    :return: None
    """
    global count_id
    for img_name in os.listdir(origin_dir):
        if img_name.endswith('.jpg'):
            xml_src_path = os.path.join(origin_dir, img_name[:-4] + '.xml')

            if os.path.exists(xml_src_path):  # 验证一个img对应一个xml
                id_name = '{:0>6d}'.format(count_id)
                format_name = type_name + date_name + id_name

                img_src_path = os.path.join(origin_dir, img_name)

                img_dst_path = os.path.join(img_path, format_name + '.jpg')
                xml_dst_path = os.path.join(xml_path, format_name + '.xml')

                shutil.copyfile(img_src_path, img_dst_path)
                shutil.copyfile(xml_src_path, xml_dst_path)

                count_id += 1
                if count_id % 100 == 0:
                    print('=> {} files_path moved and renamed.'.format(count_id))
    print('=> copy and rename total {} files_path done.'.format(count_id))


# 生成img绝对路径和列表
def get_img_path_list(img_path):
    """
    :param img_path: 图像重命名后的路径
    :return: None, 将图像文件绝对路径列表写入磁盘
    """
    list_path = os.path.join(dst_root + os.path.sep, 'all.txt')
    if os.path.exists(list_path):
        os.remove(list_path)

    with open(list_path, 'w') as f_list:
        img_files = os.listdir(img_path)
        for f_name in img_files:
            the_path = os.path.join(img_path + os.path.sep, f_name)
            f_list.write(the_path + '\n')
    print('=> generating all.txt done, total {} images.'.format(len(img_files)))


# 计算label, 转换成dark-net要求的形式,对原图归一化
def convert(size, bbox):
    """
    :param size: 原始图像宽高
    :param bbox: 车牌最小外接矩形
    :return: 转换后的bbox中心坐标和宽高
    """
    dw = 1.0 / size[0]  # 整张原图宽度
    dh = 1.0 / size[1]  # 整张原图高度
    x = (bbox[0] + bbox[1]) / 2.0 - 1.0
    y = (bbox[2] + bbox[3]) / 2.0 - 1.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x *= dw
    w *= dw
    y *= dh
    h *= dh
    return x, y, w, h


global bbox_max_width, bbox_max_height
bbox_max_width, bbox_max_height = 0, 0


def format_4_vertices(vertices):
    """
    vertices: 2D list or numpy array
    """
    assert len(vertices) == 4

    left_pts = sorted(vertices, key=lambda x: int(x[0]))[:2]
    right_pts = [pt for pt in vertices if pt not in left_pts]

    left_up = sorted(left_pts, key=lambda x: int(x[1]))[0]
    left_down = [x for x in left_pts if x != left_up][0]

    right_up = sorted(right_pts, key=lambda x: int(x[1]))[0]
    right_down = [x for x in right_pts if x != right_up][0]

    return [left_up, right_up, right_down, left_down]


def format_lp_vertices(vertices):
    """
    按照left-up编号为0, 顺时针依次编号的循序排列:
    leftup -> 0
    rightup -> 1
    rightdown -> 2
    leftdown -> 3
    """
    assert len(vertices) == 4
    vertices = [pt.split(',') for pt in vertices]
    print('=> vertices: ', vertices)

    left_pts = sorted(vertices, key=lambda x: int(x[0]))[:2]
    right_pts = [pt for pt in vertices if pt not in left_pts]

    left_up = sorted(left_pts, key=lambda x: int(x[1]))[0]
    left_down = [x for x in left_pts if x != left_up][0]
    right_up = sorted(right_pts, key=lambda x: int(x[1]))[0]
    right_down = [x for x in right_pts if x != right_up][0]

    # do statistics...
    # global bbox_max_width, bbox_max_height
    # bbox_width = max(right_up[0], right_down[0]) - min(left_up[0], left_down[0])
    # bbox_height = max(left_down[1], right_down[1]) - min(left_up[1], right_up[1])
    # if bbox_width > bbox_max_width:
    #     bbox_max_width = bbox_width
    # if bbox_height > bbox_max_height:
    #     bbox_max_height = bbox_height

    return [left_up, right_up, right_down, left_down]


# ---------------------- generate patches
interp_methods = [cv2.INTER_LINEAR,
                  cv2.INTER_CUBIC,
                  cv2.INTER_AREA,
                  cv2.INTER_NEAREST,
                  cv2.INTER_LANCZOS4]


def gen_pos_patch(img,
                  vertices,
                  offset_radius=15,
                  size=(240, 240)):
    """
    生成positive patch: 大多数bbox都满足条件, 做非等比缩放
    """
    H, W, _ = img.shape  # H×W×channels

    # 格式化四个顶点
    vertices = [[int(x) for x in pt.split(',')]
                for pt in vertices]
    vertices = format_4_vertices(vertices)

    #                       0      1      2      3
    # 将四个顶点转换成bbox: x_min, x_max, y_min, y_max
    bbox = bbox_from_vertices(vertices)

    # bbox中心点坐标
    center_x = int(float(bbox[0] + bbox[1]) * 0.5 + 0.5)
    center_y = int(float(bbox[2] + bbox[3]) * 0.5 + 0.5)

    # bbox长宽
    bbox_w, bbox_h = float(bbox[1] - bbox[0]), float(bbox[3] - bbox[2])

    # 计算patch中心offset
    offset_x = random.uniform(-1.0, 1.0) * offset_radius
    offset_y = random.uniform(-1.0, 1.0) * offset_radius

    # 计算patch的org坐标和end坐标: 这个4倍有改进的空间...
    patch_w, patch_h = 4.0 * bbox_w, 4.0 * bbox_h
    patch_org = [center_x - patch_w * 0.5 - offset_x,
                 center_y - patch_h * 0.5 - offset_y]
    patch_end = [patch_org[0] + patch_w,
                 patch_org[1] + patch_h]

    # 改进: 使得位于图片边界的bbox也能用来训练
    # if patch_org[0] < 0 or patch_org[1] < 0 \
    #         or patch_end[0] >= W or patch_end[1] >= H:
    #     return None, None, None, None

    if patch_org[0] < 0 or patch_org[1] < 0:
        patch_org[0] = patch_org[0] if patch_org[0] >= 0 else 0
        patch_org[1] = patch_org[1] if patch_org[1] >= 0 else 0

        # 更新patch w, h
        patch_w = patch_end[0] - patch_org[0]
        patch_h = patch_end[1] - patch_org[1]

    if patch_end[0] >= W or patch_end[1] >= H:
        patch_end[0] = patch_end[0] if patch_end[0] < W else W - 1
        patch_end[1] = patch_end[1] if patch_end[1] < H else H - 1

        # 更新patch w, h
        patch_w = patch_end[0] - patch_org[0]
        patch_h = patch_end[1] - patch_org[1]

    # 计算相对坐标, 并归一化到[0, 1]
    bbox_leftup = [(bbox[0] - patch_org[0]) / patch_w,
                   (bbox[2] - patch_org[1]) / patch_h]
    bbox_rightdown = [(bbox[1] - patch_org[0]) / patch_w,
                      (bbox[3] - patch_org[1]) / patch_h]

    # 四个顶点(keypoint)
    leftup = [(vertices[0][0] - patch_org[0]) / patch_w,
              (vertices[0][1] - patch_org[1]) / patch_h]

    rightup = [(vertices[1][0] - patch_org[0]) / patch_w,
               (vertices[1][1] - patch_org[1]) / patch_h]

    rightdown = [(vertices[2][0] - patch_org[0]) / patch_w,
                 (vertices[2][1] - patch_org[1]) / patch_h]

    leftdown = [(vertices[3][0] - patch_org[0]) / patch_w,
                (vertices[3][1] - patch_org[1]) / patch_h]

    # 截取patch: H,W,channels
    patch = img[int(patch_org[1] + 0.5): int(patch_end[1] + 0.5),
                int(patch_org[0] + 0.5): int(patch_end[0] + 0.5), :]

    if patch is None:
        return None, None, None, None

    # 非等比缩放patch到指定size
    try:
        patch = cv2.resize(patch, dsize=size, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(e)

    # 计算在新坐标系下的绝对坐标
    patch_w_new, patch_h_new = float(size[0]), float(size[1])

    bbox_leftup = [int(bbox_leftup[0] * patch_w_new + 0.5),
                   int(bbox_leftup[1] * patch_h_new + 0.5)]
    bbox_rightdown = [int(bbox_rightdown[0] * patch_w_new + 0.5),
                      int(bbox_rightdown[1] * patch_h_new + 0.5)]

    leftup = [int(leftup[0] * patch_w_new + 0.5),
              int(leftup[1] * patch_h_new + 0.5)]
    rightup = [int(rightup[0] * patch_w_new + 0.5),
               int(rightup[1] * patch_h_new + 0.5)]
    rightdown = [int(rightdown[0] * patch_w_new + 0.5),
                 int(rightdown[1] * patch_h_new + 0.5)]
    leftdown = [int(leftdown[0] * patch_w_new + 0.5),
                int(leftdown[1] * patch_h_new + 0.5)]

    # landmark计算gray zone的约束
    if leftup[0] < 8 or leftup[1] < 8 or \
            rightup[0] > W - 8 or rightup[1] < 8 or \
            rightdown[0] > W - 8 or rightdown[1] > H - 8 or \
            leftdown[0] < 8 or leftdown[1] > H - 8:
        return None, None, None, None

    # 可视化中间结果

    """
    # leftup
    cv2.circle(patch, tuple(leftup), 5, (0, 255, 255), -1)
    txt = 'LU'
    txt_size = cv2.getTextSize(text=txt,
                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                               fontScale=1,
                               thickness=1)[0]
    cv2.putText(img=patch,
                text=txt,
                org=(leftup[0] - txt_size[0], leftup[1]),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=[225, 255, 255],
                thickness=1)

    # rightup
    cv2.circle(patch, tuple(rightup), 5, (0, 255, 255), -1)
    txt = 'RU'
    txt_size = cv2.getTextSize(text=txt,
                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                               fontScale=1,
                               thickness=1)[0]
    cv2.putText(img=patch,
                text=txt,
                org=(rightup[0], rightup[1]),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=[225, 255, 255],
                thickness=1)

    # rightdown
    cv2.circle(patch, tuple(rightdown), 5, (0, 255, 255), -1)
    txt = 'RD'
    txt_size = cv2.getTextSize(text=txt,
                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                               fontScale=1,
                               thickness=1)[0]
    cv2.putText(img=patch,
                text=txt,
                org=(rightdown[0],
                     rightdown[1] + txt_size[1]),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=[225, 255, 255],
                thickness=1)

    # leftdown
    txt = 'LD'
    cv2.circle(patch, tuple(leftdown), 5, (0, 255, 255), -1)
    txt_size = cv2.getTextSize(text=txt,
                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                               fontScale=1,
                               thickness=1)[0]
    cv2.putText(img=patch,
                text=txt,
                org=(leftdown[0] - txt_size[0],
                     leftdown[1] + txt_size[1]),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=[225, 255, 255],
                thickness=1)

    cv2.imshow('patch', patch)
    cv2.waitKey()
    """

    return patch, bbox_leftup, bbox_rightdown, [leftup,
                                                rightup,
                                                rightdown,
                                                leftdown]


def gen_patch_lm(img,
                 vertices,
                 offset_radius=15,
                 min_scale=0.22,
                 max_scale=2.8):
    """
    生成patch的一种方法
    """
    W, H, _ = img.shape
    vertices = [[int(x) for x in pt.split(',')]
                for pt in vertices]
    vertices = format_4_vertices(vertices)

    # 将四个顶点转换成bbox(x_min, x_max, y_min, y_max)
    bbox = bbox_from_vertices(vertices)

    # 计算bbox是否满足尺度条件
    center_x = int(float(bbox[0] + bbox[1]) * 0.5 + 0.5)
    center_y = int(float(bbox[2] + bbox[3]) * 0.5 + 0.5)

    area = float(bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    if area > 2500.0 * max_scale or area < 2500.0 * min_scale:
        # print('=> incompatible scale.')
        return None, None, None, None
    else:
        offset_x = random.uniform(-1.0, 1.0) * offset_radius
        offset_y = random.uniform(-1.0, 1.0) * offset_radius
        offset_x = int(
            offset_x + 0.5) if offset_x > 0.0 else int(offset_x - 0.5)
        offset_y = int(
            offset_y + 0.5) if offset_y > 0.0 else int(offset_y - 0.5)
        # print(offset_x, ' ', offset_y)

        org = [center_x - offset_x - 120, center_y - offset_y - 120]
        if org[0] < 0 or org[0] > W or \
                org[1] < 0 or org[1] > H:
            return None, None, None, None
        else:
            bbox_leftup = [bbox[0] - org[0], bbox[2] - org[1]]
            bbox_rightdown = [bbox[1] - org[0], bbox[3] - org[1]]

            leftup = vertices[0]
            rightup = vertices[1]
            rightdown = vertices[2]
            leftdown = vertices[3]

            leftup = [leftup[0] - org[0], leftup[1] - org[1]]
            rightup = [rightup[0] - org[0], rightup[1] - org[1]]
            rightdown = [rightdown[0] - org[0], rightdown[1] - org[1]]
            leftdown = [leftdown[0] - org[0], leftdown[1] - org[1]]

            patch = img[org[1]: org[1] + 240, org[0]: org[0] + 240, :]

            return patch, bbox_leftup, bbox_rightdown, [leftup,
                                                        rightup,
                                                        rightdown,
                                                        leftdown]


def gen_patch_1(img,
                vertices,
                offset_radius=15,
                min_scale=0.22,
                max_scale=2.8):
    """
    生成patch的一种方法
    """
    W, H, _ = img.shape
    vertices = [[int(x) for x in pt.split(',')]
                for pt in vertices]

    # 将四个顶点转换成bbox(leftup)
    bbox = bbox_from_vertices(vertices)

    # 计算bbox是否满足尺度条件
    center_x = int(float(bbox[0] + bbox[1]) * 0.5 + 0.5)
    center_y = int(float(bbox[2] + bbox[3]) * 0.5 + 0.5)
    area = float(bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    if area > 2500.0 * max_scale or area < 2500.0 * min_scale:
        # print('=> incompatible scale.')
        return None, None, None
    else:
        offset_x = random.uniform(-1.0, 1.0) * offset_radius
        offset_y = random.uniform(-1.0, 1.0) * offset_radius
        offset_x = int(
            offset_x + 0.5) if offset_x > 0.0 else int(offset_x - 0.5)
        offset_y = int(
            offset_y + 0.5) if offset_y > 0.0 else int(offset_y - 0.5)
        # print(offset_x, ' ', offset_y)

        org = [center_x - offset_x - 120, center_y - offset_y - 120]
        if org[0] < 0 or org[0] > W or \
                org[1] < 0 or org[1] > H:
            return None, None, None
        else:
            leftup = [bbox[0] - org[0], bbox[2] - org[1]]
            rightdown = [bbox[1] - org[0], bbox[3] - org[1]]

            patch = img[org[1]: org[1] + 240, org[0]: org[0] + 240, :]
            return patch, leftup, rightdown


def pad_resize_img(img, size=(240, 240)):
    """
    :param img: RGB image
    :return:
    """
    img = np.array(img)  # H x W x channels
    H, W, channels = img.shape
    dim_diff = np.abs(H - W)

    # upper(left) and lower(right) padding
    pad_lu = dim_diff // 2  # integer division
    pad_rd = dim_diff - pad_lu

    # determine padding for each axis: H, W, channels
    pad = ((pad_lu, pad_rd), (0, 0), (0, 0)) if H <= W else \
        ((0, 0), (pad_lu, pad_rd), (0, 0))

    # do padding(0.5) and normalize
    img = np.pad(img,
                 pad,
                 'constant',
                 constant_values=128.0)  # / 255.0
    img = cv2.resize(img,
                     size,
                     cv2.INTER_CUBIC)
    return img


def gen_patch_2(img,
                vertices,
                offset_radius=8):
    """
    从任意一张车牌生成patch
    """
    W, H, _ = img.shape
    vertices = [[int(x) for x in pt.split(',')]
                for pt in vertices]
    bbox = bbox_from_vertices(vertices)
    center_x = int(float(bbox[0] + bbox[1]) * 0.5 + 0.5)
    center_y = int(float(bbox[2] + bbox[3]) * 0.5 + 0.5)
    bbox_w, bbox_h = int(bbox[1] - bbox[0]), int(bbox[3] - bbox[2])
    # area = float(bbox_w) * (bbox_h)

    # offset_x = random.uniform(-1.0, 1.0) * offset_radius
    # offset_y = random.uniform(-1.0, 1.0) * offset_radius
    # offset_x = int(offset_x + 0.5) if offset_x > 0.0 else int(offset_x - 0.5)
    # offset_y = int(offset_y + 0.5) if offset_y > 0.0 else int(offset_y - 0.5)
    offset_x, offset_y = 0, 0
    print(offset_x, ' ', offset_y)

    patch_w, patch_h = int(float(bbox_w) * 4.8 +
                           0.5), int(float(bbox_h) * 4.8 + 0.5)
    if patch_w > 0 and patch_w < W \
            and patch_h > 0 and patch_h < H:
        org_x = center_x - offset_x - int(patch_w * 0.5 + 0.5)
        org_y = center_y - offset_y - int(patch_h * 0.5 + 0.5)
        if org_x > 0 and org_x < W \
                and org_y > 0 and org_y < H:

            # org = [center_x - offset_x - int(patch_w * 0.5 + 0.5),
            #        center_y - offset_y - int(patch_h * 0.5 + 0.5)]

            end_x, end_y = org_x + patch_w, org_y + patch_h
            if end_x > 0 and end_x < W \
                    and end_y > 0 and end_y < H:
                patch = img[org_x: end_y,
                            org_y: end_x, :]

                # patch = cv2.resize(patch, (240, 240), cv2.INTER_CUBIC)
                patch = pad_resize_img(img)
                return patch
    else:
        return None


def gen_lp_patches(root,
                   offset_radius=10,
                   min_scale=0.25,
                   max_scale=2.8,
                   is_viz=False):
    """
    处理成DenseBox训练数据
    """
    if not os.path.isdir(root):
        print('=> [Err]: invalid root.')
        return

    jpg_dir = root + '/JPEGImages'
    txt_dir = root + '/labels'
    if not (os.path.isdir(jpg_dir) and os.path.isdir(txt_dir)):
        print('=> [Err]: invalid dir.')
        return

    patch_dir = root + '/patches'
    if not os.path.isdir(patch_dir):
        os.makedirs(patch_dir)

    # process each img
    cnt = 0
    for img_name in tqdm(os.listdir(jpg_dir)):
        img_path = jpg_dir + '/' + img_name
        txt_path = txt_dir + '/' + img_name.replace('.jpg', '.txt')
        if os.path.isfile(img_path) and os.path.isfile(txt_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            with open(txt_path, 'r', encoding='utf-8') as txt_h:
                # process each plate
                for line in txt_h.readlines():
                    # 4个顶点
                    vertices = line.strip().split(' ')

                    assert len(vertices) == 4

                    patch, leftup, rightdown = gen_patch_1(img,
                                                           vertices,
                                                           offset_radius,
                                                           min_scale=min_scale,
                                                           max_scale=max_scale)
                    # patch = gen_patch_2(img, vertices, offset_radius=8)

                    if patch is None:
                        continue

                    # -------------- 可视化
                    if is_viz:
                        cv2.circle(patch, tuple(leftup), 3, (0, 255, 255), -1)
                        txt = 'LU'
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(leftup[0] - txt_size[0], leftup[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        cv2.circle(patch, tuple(rightdown),
                                   3, (0, 255, 255), -1)
                        txt = 'RD'
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(rightdown[0] -
                                         txt_size[0], rightdown[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        window_name = 'test'
                        cv2.moveWindow(window_name, 200, 200)
                        cv2.imshow(window_name, patch)
                        cv2.waitKey()
                    # --------------

                    # -------------- 生成label: leftup_x, leftup_y, rightdown_x, right_down_y
                    coord_str_arr = [str(leftup[0]),
                                     str(leftup[1]),
                                     str(rightdown[0]),
                                     str(rightdown[1])]
                    label = '_'.join(coord_str_arr)

                    patch_path = patch_dir + '/' \
                        + img_name[:-4] + '_label_' + label + '.jpg'

                    # 保存patch
                    cv2.imwrite(patch_path, patch)

                    cnt += 1
                    print('=> %d patche generated.' % cnt)

                # print('\n')
    print('=> total %d patches' % cnt)


# -------------------------------- generating patch with label containing formatted 4 corners

def intersect_small(bboxes, patch_center, TH):
    """
    计算patch与bbox的intersection
    """
    patch_x_min = patch_center[0] - 120
    patch_x_max = patch_center[0] + 120
    patch_y_min = patch_center[1] - 120
    patch_y_max = patch_center[1] + 120

    for bbox in bboxes:
        # bbox: leftup, rightup, rightdown, leftdown
        bbox_x_min, bbox_x_max = bbox[0][0], bbox[2][0]
        bbox_y_min, bbox_y_max = bbox[0][1], bbox[2][1]

        inter_w = min(bbox_x_max, patch_x_max) \
            - max(bbox_x_min, patch_x_min)
        inter_h = min(bbox_y_max, patch_y_max) \
            - max(bbox_y_min, patch_y_min)

        if inter_w < 0 or inter_h < 0:
            return True
        intersection = inter_w * inter_h
        print('=> intersection: ', intersection)

        if intersection > TH:
            return False

    return True


def gen_pure_neg_patches(src_root,
                         dst_root,
                         num=1000,
                         TH=0):
    """
    生成纯负样本的patch数据
    """
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid dir.')
        return

    src_jpg_dir = src_root + '/JPEGImages'
    src_txt_dir = src_root + '/labels'
    if not (os.path.isdir(src_jpg_dir) and os.path.isdir(src_txt_dir)):
        print('=> [Err]: invalid jpg or txt dir.')
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
    # else:
    #     for x in os.listdir(dst_root):
    #         x_path = dst_root + '/' + x
    #         os.remove(x_path)

    imgs_path = [src_jpg_dir + '/' +
                 img_name for img_name in os.listdir(src_jpg_dir)]
    pic_num = len(imgs_path)
    print('=> total %d src jpgs.' % (pic_num))

    cnt = 0
    for item_i in range(num):
        # 随机选择一张图片
        rand_pic_id = int(np.random.choice(pic_num, 1, replace=True)[0])
        # print('=> rand pic id: ', rand_pic_id)

        img_path = imgs_path[int(rand_pic_id)]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        H, W, num_chans = img.shape

        # 生成patch中心点
        center_x = np.random.choice(
            np.arange(120, W - 120), 1, replace=True)[0]
        center_y = np.random.choice(
            np.arange(120, H - 120), 1, replace=True)[0]
        print('=> patch center:(%d, %d)' % (center_x, center_y))

        # 生成一个patch
        patch = img[center_y-120: center_y + 120,  # H
                    center_x-120: center_x + 120,  # W
                    :]                             # num_channels
        if patch is None:
            continue

        # 读取bboxes
        txt_path = str(src_txt_dir + '/' + os.path.split(img_path)
                       [1]).replace('.jpg', '.txt')
        if os.path.isfile(txt_path):
            bboxes = []

            # 读取每一个bbox
            with open(txt_path, 'r', encoding='utf-8') as f_h:
                for line in f_h.readlines():
                    line = line.strip().split(' ')
                    vertices = [[int(x) for x in pt.split(',')]
                                for pt in line]

                    vertices = format_4_vertices(vertices=vertices)
                    bboxes.append(vertices)

            # 计算intersection是否超过阈值
            if intersect_small(bboxes, [center_x, center_y], TH=TH):

                # 将满足条件的patch写入目标目录
                dst_path = dst_root + '/' + \
                    os.path.split(img_path)[1][:-4] + \
                    '_label_0_0_0_0_0_0_0_0_0_0_0_0' + '.jpg'
                if not os.path.isfile(dst_path):
                    cv2.imwrite(dst_path, patch)
                    cnt += 1
        print('=> item %d done\n' % (item_i))
    print('=> total %d negative patch sample generated.' % (cnt))


def gen_lplm_pos_patches(src_root,
                         dst_root,
                         offset_radius=10,
                         is_viz=False):
    """
    处理成DenseBox训练数据, 生成带标签的positive patch
    非等比缩放到指定尺寸
    """
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid root.')
        return

    jpg_dir = src_root + '/JPEGImages'
    txt_dir = src_root + '/labels'
    if not (os.path.isdir(jpg_dir) and os.path.isdir(txt_dir)):
        print('=> [Err]: invalid dir.')
        return

    # dst_root = src_root + '/patchesLM'
    # if not os.path.isdir(dst_root):
    #     os.makedirs(dst_root)

    # process each img
    cnt = 0  # patch计数
    for item_i, img_name in enumerate(os.listdir(jpg_dir)):
        img_path = jpg_dir + '/' + img_name
        txt_path = txt_dir + '/' + img_name.replace('.jpg', '.txt')
        if os.path.isfile(img_path) and os.path.isfile(txt_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            with open(txt_path, 'r', encoding='utf-8') as txt_h:
                # process each plate
                for line in txt_h.readlines():
                    # 4个顶点
                    vertices = line.strip().split(' ')

                    assert len(vertices) == 4

                    # 生成patch和坐标
                    patch, bbox_leftup, bbox_rightdown, vertices = gen_pos_patch(img=img,
                                                                                 vertices=vertices,
                                                                                 offset_radius=15,
                                                                                 size=(240, 240))
                    if patch is None:
                        continue

                    # -------------- 生成label:
                    leftup = vertices[0]     # 0
                    rightup = vertices[1]    # 1
                    rightdown = vertices[2]  # 2
                    leftdown = vertices[3]   # 3

                    coords_str = [str(bbox_leftup[0]),
                                  str(bbox_leftup[1]),  # bbox_leftup
                                  str(bbox_rightdown[0]),
                                  str(bbox_rightdown[1]),  # bbox_rightdown

                                  str(leftup[0]),
                                  str(leftup[1]),  # leftup

                                  str(rightup[0]),
                                  str(rightup[1]),  # rightup

                                  str(rightdown[0]),
                                  str(rightdown[1]),  # rightdown

                                  str(leftdown[0]),
                                  str(leftdown[1])]  # leftdown

                    label = '_'.join(coords_str)
                    patch_path = dst_root + '/' \
                        + img_name[:-4] + '_label_' + label + '.jpg'

                    # 保存patch
                    cv2.imwrite(patch_path, patch)

                    cnt += 1
                    print('=> %d patch generated\n' % (cnt))

        print('=> %s proceesed, %.3f%% completed.'
              % (img_name,
                 float(item_i + 1) / float(len(os.listdir(jpg_dir))) * 100.0))

    print('=> total %d patches' % cnt)


def gen_lplm_patches(root,
                     offset_radius=10,
                     min_scale=0.25,
                     max_scale=2.8,
                     is_viz=False):
    """
    处理成DenseBox训练数据, 生成带标签的patch
    """
    if not os.path.isdir(root):
        print('=> [Err]: invalid root.')
        return

    jpg_dir = root + '/JPEGImages'
    txt_dir = root + '/labels'
    if not (os.path.isdir(jpg_dir) and os.path.isdir(txt_dir)):
        print('=> [Err]: invalid dir.')
        return

    patch_dir = root + '/patchesLM'
    if not os.path.isdir(patch_dir):
        os.makedirs(patch_dir)

    # process each img
    cnt = 0
    for item_i, img_name in enumerate(os.listdir(jpg_dir)):
        img_path = jpg_dir + '/' + img_name
        txt_path = txt_dir + '/' + img_name.replace('.jpg', '.txt')
        if os.path.isfile(img_path) and os.path.isfile(txt_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            with open(txt_path, 'r', encoding='utf-8') as txt_h:
                # process each plate
                for line in txt_h.readlines():
                    # 4个顶点
                    vertices = line.strip().split(' ')

                    assert len(vertices) == 4

                    patch, bbox_leftup, bbox_rightdown, vertices = gen_patch_lm(img=img,
                                                                                vertices=vertices,
                                                                                offset_radius=offset_radius,
                                                                                min_scale=min_scale,
                                                                                max_scale=max_scale)

                    if patch is None:
                        continue

                    # -------------- 可视化
                    # 绘制每个顶点
                    if is_viz:
                        # left-up
                        pt = tuple(vertices[0])
                        txt = 'LU'
                        cv2.circle(img, pt, 5, (0, 255, 255), -1)
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(
                                        pt[0] - txt_size[0], pt[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        # right-up
                        pt = tuple(vertices[1])
                        txt = 'RU'
                        cv2.circle(img, pt, 5, (0, 255, 255), -1)
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(pt[0], pt[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        # right-down
                        pt = tuple(vertices[2])
                        txt = 'RD'
                        cv2.circle(img, pt, 5, (0, 255, 255), -1)
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(pt[0],
                                         pt[1] + txt_size[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        # left-down
                        pt = tuple(vertices[3])
                        txt = 'LD'
                        cv2.circle(img, pt, 5, (0, 255, 255), -1)
                        txt_size = cv2.getTextSize(text=txt,
                                                   fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                   fontScale=1,
                                                   thickness=1)[0]
                        cv2.putText(img=patch,
                                    text=txt,
                                    org=(pt[0] - txt_size[0],
                                         pt[1] + txt_size[1]),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    color=[225, 255, 255],
                                    thickness=1)

                        window_name = 'test'
                        cv2.moveWindow(window_name, 200, 200)
                        cv2.imshow(window_name, patch)
                        cv2.waitKey()
                    # --------------

                    # -------------- 生成label:
                    leftup = vertices[0]     # 0
                    rightup = vertices[1]    # 1
                    rightdown = vertices[2]  # 2
                    leftdown = vertices[3]   # 3

                    coords_str = [str(bbox_leftup[0]),
                                  str(bbox_leftup[1]),  # bbox_leftup
                                  str(bbox_rightdown[0]),
                                  str(bbox_rightdown[1]),  # bbox_rightdown

                                  str(leftup[0]),
                                  str(leftup[1]),  # leftup

                                  str(rightup[0]),
                                  str(rightup[1]),  # rightup

                                  str(rightdown[0]),
                                  str(rightdown[1]),  # rightdown

                                  str(leftdown[0]),
                                  str(leftdown[1])]  # leftdown

                    label = '_'.join(coords_str)
                    patch_path = patch_dir + '/' \
                        + img_name[:-4] + '_label_' + label + '.jpg'

                    # 保存patch
                    cv2.imwrite(patch_path, patch)

                    cnt += 1
                    print('=> %d patch generated\n' % (cnt))

                # print('\n')

        print('=> %s proceesed, %.3f%% completed.'
              % (img_name,
                 float(item_i + 1) / float(len(os.listdir(jpg_dir))) * 100.0))

    print('=> total %d patches' % cnt)


def viz_patch(root):
    """
    可视化license plate patch
    """


def viz_vertices(root):
    """
    可视化车牌的四个顶点
    """
    viz_res_dir = root + '/' + 'viz_result'
    if not os.path.exists(viz_res_dir):
        os.makedirs(viz_res_dir)
    else:
        for x in os.listdir(viz_res_dir):
            x_path = viz_res_dir + '/' + x
            os.remove(x_path)

    JPEG_dir = root + '/' + 'JPEGImages'
    label_dir = root + '/' + 'labels'

    viz_vertex(JPEG_dir, label_dir, viz_res_dir)


def viz_vertex(JPEG_dir, label_dir, viz_res_dir):
    for x in tqdm(os.listdir(JPEG_dir)):
        if x.endswith('.jpg'):
            img_path = JPEG_dir + '/' + x
            if os.path.isfile(img_path):
                label_path = label_dir + '/' + x[:-4] + '.txt'
                if os.path.isfile(label_path):
                    # 读取原图
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                    # 读取label
                    with open(label_path, 'r', encoding='utf-8') as f_label:
                        # 绘制每一个车牌
                        for line in f_label.readlines():
                            vertices = line.split(' ')

                            assert len(vertices) == 4  # 4个顶点

                            # 绘制每个顶点
                            for pt in vertices:
                                if vertices.index(pt) == 0:  # test left-up
                                    pt = tuple([int(x) for x in pt.split(',')])
                                    cv2.circle(img, pt, 5, (0, 255, 255), -1)
                                    txt = 'LU'
                                    txt_size = cv2.getTextSize(text=txt,
                                                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                               fontScale=1,
                                                               thickness=1)[0]
                                    cv2.putText(img=img,
                                                text=txt,
                                                org=(
                                                    pt[0] - txt_size[0], pt[1]),
                                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                color=[225, 255, 255],
                                                thickness=1)
                                elif vertices.index(pt) == 1:  # test right-up
                                    pt = tuple([int(x) for x in pt.split(',')])
                                    cv2.circle(img, pt, 5, (0, 255, 255), -1)
                                    txt = 'RU'
                                    txt_size = cv2.getTextSize(text=txt,
                                                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                               fontScale=1,
                                                               thickness=1)[0]
                                    cv2.putText(img=img,
                                                text=txt,
                                                org=(pt[0], pt[1]),
                                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                color=[225, 255, 255],
                                                thickness=1)
                                elif vertices.index(pt) == 2:  # test right-down
                                    pt = tuple([int(x) for x in pt.split(',')])
                                    cv2.circle(img, pt, 5, (0, 255, 255), -1)

                                    txt = 'RD'
                                    txt_size = cv2.getTextSize(text=txt,
                                                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                               fontScale=1,
                                                               thickness=1)[0]

                                    cv2.putText(img=img,
                                                text=txt,
                                                org=(pt[0],
                                                     pt[1] + txt_size[1]),
                                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                color=[225, 255, 255],
                                                thickness=1)
                                else:  # test left-down
                                    pt = tuple([int(x) for x in pt.split(',')])
                                    cv2.circle(img, pt, 5, (0, 255, 255), -1)

                                    txt = 'LD'
                                    txt_size = cv2.getTextSize(text=txt,
                                                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                               fontScale=1,
                                                               thickness=1)[0]

                                    cv2.putText(img=img,
                                                text=txt,
                                                org=(pt[0] - txt_size[0],
                                                     pt[1] + txt_size[1]),
                                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                color=[225, 255, 255],
                                                thickness=1)
                        # cv2.imshow('test', img)
                        # cv2.waitKey()

                        res_path = viz_res_dir + '/' + x
                        cv2.imwrite(res_path, img)


# 解析xml并转换成labels
def xml2label(img_path, xml_path, label_path, f_name):
    """
    :param img_path:
    :param xml_path:
    :param label_path:
    :param file_name:
    :return:
    """
    xml_f_name = xml_path + '/' + f_name + '.xml'
    label_f_name = label_path + '/' + f_name + '.txt'
    in_file = open(xml_f_name, 'r', encoding='utf-8')  # 指定打开xml的编码方式
    out_file = open(label_f_name, 'w', encoding='utf-8')

    # 解析xml字段
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    W = int(size.find('width').text)  # 整张图宽度
    H = int(size.find('height').text)  # 整张图高度
    # print('w: %d, h: %d' % (W, H))

    # 处理一张图
    for obj in root.iter('object'):  # object字段
        cls = obj.find('name').text  # 类别: 车牌...
        if cls not in classes:
            continue

        # 处理图中的每一个车牌
        for plate in obj.iter('plate'):  # plate字段
            vertices = plate.find('vertexs')
            if None == vertices or 4 != len(vertices):
                continue

            vertices = [(int(v.find('x').text), int(v.find('y').text))
                        for v in vertices]
            vertices = format_lp_vertices(vertices)

            out_file.write(' '.join([str(x).replace('(', '').replace(
                ')', '').replace(' ', '') for x in vertices]) + '\n')

    # 释放资源
    in_file.close()
    out_file.close()


def parse_xml(in_file):
    """
    @param in_file: file handle of read xml
    """
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    W = int(size.find('width').text)  # 整张图宽度
    H = int(size.find('height').text)  # 整张图高度
    # print('w: %d, h: %d' % (W, H))

    # 处理一张图
    label = []
    for obj in root.iter('object'):  # object字段
        cls = obj.find('name').text  # 类别: 车牌...
        if cls not in classes:
            continue

        # 处理图中的每一个车牌
        for plate in obj.iter('plate'):  # plate字段
            vertices = plate.find('vertexs')
            if None == vertices or 4 != len(vertices):
                continue

            vertices = [(int(v.find('x').text), int(v.find('y').text))
                        for v in vertices]
            vertices = format_lp_vertices(vertices)
        label.append(vertices)
    return label


# 分割预处理结束后的数据集
def split_data(path_dir, thresh):
    """
    :param path_dir: 包含all.txt(所有文件目录的)
    :param thresh:
    :return:
    """
    f = open(os.path.join(path_dir, 'all.txt'), 'r')
    train_list = open(os.path.join(path_dir, 'train.txt'), 'w')
    valid_list = open(os.path.join(path_dir, 'valid.txt'), 'w')

    for line in f.readlines():
        if random.random() < thresh:
            valid_list.write(line)
        else:
            train_list.write(line)

    # 释放资源
    f.close()
    train_list.close()
    valid_list.close()
    print('=> splitting done.')


def generate_labels(root_dir, img_path, xml_path, label_path):
    """
    :param root_dir:
    :param img_path:
    :param xml_path:
    :param label_path:
    :return: None
    """
    all_files = open(root_dir + '/' + 'all.txt', 'r')
    for i, line in enumerate(all_files.readlines()):
        # print('line: %s, len(line): %s\k' % (line, len(line)))
        f_name = os.path.split(line)[1].strip('\n')[:-4]

        print('\n=> processing number %d: %s' % (i + 1, f_name))
        xml2label(img_path, xml_path, label_path, f_name)
        if i % 100 == 0:
            print('=> {} labels generated'.format(i + 1))

    # 释放资源
    all_files.close()
    # print('=> bbox max width: %d, max height: %d' %(bbox_max_width, bbox_max_height))
    print('=> generating all labels done.')


def bbox_from_vertices(vertices):
    """
    get bounding box from 4 vertices:
      0      1      2      3
    x_min, x_max, y_min, y_max
    """
    return min(vertices[0][0], vertices[3][0]), \
        max(vertices[1][0], vertices[2][0]), \
        min(vertices[0][1], vertices[1][1]), \
        max(vertices[3][1], vertices[2][1])


global count_crop_id
count_crop_id = 0


def rand_crop_around_center(root,
                            fix_size=256,
                            border=15):
    """
    围绕中心随机裁剪固定大小的图像块
    生成新的训练数据和标签
    """
    if fix_size <= (border + border):
        print('[Err]: incompatible border and fix_size.')

    # 清空目标目录
    crop_img_dir = root + '/CroppedImages/'
    crop_label_dir = root + '/CroppedLabels/'
    img_list = os.listdir(crop_img_dir)
    label_list = os.listdir(crop_label_dir)
    if len(img_list) != 0:
        for x in img_list:
            x_path = crop_img_dir + x
            if os.path.exists(x_path):
                os.remove(x_path)
    if len(label_list) != 0:
        for x in label_list:
            x_path = crop_label_dir + x
            if os.path.exists(x_path):
                os.remove(x_path)

    global count_crop_id

    all_files = open(root + '/' + 'all.txt', 'r')
    for i, line in enumerate(all_files.readlines()):
        # print('line: %s, len(line): %s\k' % (line, len(line)))
        f_name = os.path.split(line)[1].strip('\n')[: -4]
        img_f_path = root + '/JPEGImages/' + f_name + '.jpg'
        xml_f_path = root + '/Annotations/' + f_name + '.xml'

        if os.path.isfile(img_f_path) and os.path.isfile(xml_f_path):
            with open(xml_f_path, 'r', encoding='utf-8') as f_xml:
                label = parse_xml(f_xml)

                # 处理每一个车牌
                for vertices in label:
                    # 确定leftup坐标范围
                    if None != vertices and len(vertices) == 4:
                        bbox = bbox_from_vertices(vertices)
                        x_min = bbox[0] - \
                            (fix_size - (bbox[1] - bbox[0])) + border
                        x_max = bbox[0] - border
                        y_min = bbox[2] - \
                            (fix_size - (bbox[3] - bbox[2])) + border
                        y_max = bbox[2] - border
                        if x_max > x_min and y_max > y_min:
                            # 随机leftup坐标
                            left_up = x_min + int(random.random() * (x_max - x_min)), \
                                y_min + int(random.random() * (y_max - y_min))
                            # print('=> leftup: ', left_up)

                            # 格式化
                            img = cv2.imread(img_f_path, cv2.IMREAD_UNCHANGED)
                            if img.shape[2] == 3:  # turn all 3 channels to RGB format
                                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            elif img.shape[2] == 1:  # turn 1 channel to RGB
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                            # 固定尺寸裁剪
                            ROI = img[left_up[1]: left_up[1] + fix_size,
                                      left_up[0]: left_up[0] + fix_size]
                            if ROI.shape == (fix_size, fix_size, 3):
                                # 生成ROI
                                id_name = '{:0>6d}'.format(count_crop_id)
                                format_name = type_name + date_name + id_name
                                dst_f_path = crop_img_dir + format_name + '.jpg'
                                cv2.imwrite(dst_f_path, ROI)

                                # 对vertices转换坐标系
                                vertices = [
                                    (coord[0] - left_up[0], coord[1] - left_up[1]) for coord in vertices]

                                # 生成label
                                label_f_path = crop_label_dir + format_name + '.txt'
                                f_label = open(
                                    label_f_path, 'w', encoding='utf-8')
                                f_label.write(' '.join([str(x).replace('(', '').replace(
                                    ')', '').replace(' ', '') for x in vertices]) + '\n')
                                f_label.close()

                                count_crop_id += 1
                                # cv2.imshow('test', ROI)
                                # cv2.waitKey()
        print('\n=> processing number %d: %s' % (i + 1, f_name))


def process_batch(src_root, dst_root):
    """
    :param src_root:
    :param dst_root:
    :return: None
    """
    # step-1: 创建子目录
    img_path = dst_root + os.path.sep + 'JPEGImages'
    xml_path = dst_root + os.path.sep + 'Annotations'
    label_path = dst_root + os.path.sep + 'labels'

    # 第一次会创建子目录, 然后复用
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
        print('=> %s made.' % dst_root)
    else:
        print('=> %s already exists.' % dst_root)

    if not os.path.exists(img_path):
        os.makedirs(img_path)
        print('=> %s made.' % img_path)
    else:
        print('=> %s already exists.' % img_path)

    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
        print('=> %s made.' % xml_path)
    else:
        print('=> %s already exists.' % xml_path)

    if not os.path.exists(label_path):
        os.makedirs(label_path)
        print('=> %s made.' % label_path)
    else:
        print('=> %s already exists.' % label_path)
    # ---------------------------三个子目录创建完成

    # step-2: 切分原始数据集并重命名(图像数据和xml标注文件)
    # move_rename(src_root, img_path, xml_path)

    # step-3: 生成图像文件绝对路径列表(all.txt)
    get_img_path_list(img_path)

    # step-4: 转换annotations(解析xml, 生成labels)
    generate_labels(dst_root, img_path, xml_path, label_path)

    # step-5: 将数据集分割为训练数据集和验证数据集
    split_data(dst_root, 0.0)

    print('=> Test done.')


def get_outfile_name(init_counter=0):
    outfile_counter = init_counter
    type_label = "ocr"
    # date_label = time.strftime('%4Y%2m%2d', time.localtime(time.time()))
    # id_label = "%06d"%(count_id)

    while True:
        id_label = '{:0>6d}'.format(outfile_counter)
        yield type_label + date_name + id_label
        outfile_counter += 1


# 递归搜索包含jpg的目录
def get_src_path(root, dirs):
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        if os.path.isdir(file_path):
            get_src_path(file_path, dirs)
        else:
            if os.path.isfile(file_path) and file_path.endswith('.jpg'):
                dirs.append(root)
                break


def test_8_neighbor_CA():
    """
    测试8邻域连通区域检测
    """
    img = cv2.imread('e:/gauss.jpg', cv2.IMREAD_UNCHANGED)
    print('=> img.shape: ', img.shape)
    print('=> img:\n', img)
    cv2.imshow('test', img)
    cv2.waitKey()

    # ret, img_bin = cv2.threshold(src=img,
    #                              thresh=127,
    #                              maxval=255,
    #                              type=cv2.THRESH_BINARY)
    # cv2.imshow('test', img_bin)
    # cv2.waitKey()

    img = img / 255.0

    ret, img_bin = cv2.threshold(src=img,
                                 thresh=0.6,
                                 maxval=1.0,
                                 type=cv2.THRESH_BINARY)
    img_bin = img_bin.astype(np.uint8)

    print(type(img_bin[0][0]))
    print('=> img_bin:\n', img_bin[:10])
    # cv2.imshow('test', img_bin)
    # cv2.waitKey()

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
    print('=> labels[:10]:\n', labels[:10])
    print('=> labels[10:20]:\n', labels[10:20])
    print('=> labels[20:]:\n', labels[20:])
    print('\n=> stats:\n', stats)
    print('\n=> centroids:\n',  centroids)

    # np.where(img)


def test_forward():
    """
    测试前向运算
    """
    net = LP_Net()
    net.load_state_dict(torch.load(
        'e:/plate_data_pro/checkpoint/epoch_780.pth'))
    net.eval()

    img = Image.open(
        'e:/plate_data_pro/CroppedImages/det_2018_09_19_000013.jpg')
    if img.mode == 'L' or img.mode == 'I':
        img = img.convert('RGB')

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.view(1, 3, 256, 256)

    output = net.forward(img)
    print(output.shape)

    # 转换成numpy.ndarray并可视化
    leftup_map = output[0][0].detach().numpy()
    rightdown_map = output[0][1].detach().numpy()
    leftup_map = leftup_map.astype(np.uint8)
    rightdown_map = rightdown_map.astype(np.uint8)

    left_non_0_ids = np.where(leftup_map != 0)
    print('=> leftup non zero ids:', left_non_0_ids)

    cv2.imwrite('e:/leftup_map.jpg', leftup_map)
    cv2.imwrite('e:/rightdown_map.jpg', rightdown_map)

    cv2.imshow('leftup_map', leftup_map)
    cv2.waitKey()
    cv2.imshow('rightdown_map', rightdown_map)
    cv2.waitKey()

    print(output[0][0][:10])
    print('\n', output[0][0][10:20])
    print('\n', output[0][0][20:])


def test_random_choice():
    """
    测试numpy.random.choice
    """
    a = np.array([19, 71, 36, 81, 69, 55])
    # for i in range(10):
    #     select = np.random.choice(a, 4, replace=False)
    #     print('=> select: ', select)

    for i in range(10):
        select = np.random.choice(len(a), 4, replace=False)
        print('=> select_ids: ', select)


def vertex2bbox_corner(vertices):
    """
    input 4 vertices(8 coordinates) and output 2 corners of bbox
    """
    assert len(vertices) == 4
    x1 = min(int(vertices[0][0]), int(vertices[3][0]))
    y1 = min(int(vertices[0][1]), int(vertices[1][1]))
    x2 = max(int(vertices[1][0]), int(vertices[2][0]))
    y2 = max(int(vertices[3][1]), int(vertices[2][1]))
    assert x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2
    return str(x1), str(y1), str(x2), str(y2)


def vertex2bbox(vertices):
    """
    input 4 vertices(8 coordinates) and output left, right, top, bottom
    """
    assert len(vertices) == 4
    x1 = min(int(vertices[0][0]), int(vertices[3][0]))
    y1 = min(int(vertices[0][1]), int(vertices[1][1]))
    x2 = max(int(vertices[1][0]), int(vertices[2][0]))
    y2 = max(int(vertices[3][1]), int(vertices[2][1]))
    assert x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2
    return str(x1), str(x2), str(y1), str(y2)


def label2anno(label_dir, anno_dir):
    """
    从label格式转换成MTCNN格式
    """
    if not os.path.isdir(label_dir):
        print('[Err]: invalid src dir.')
        return

    # JPEG目录
    jpg_dir = label_dir.replace('labels', 'JPEGImages')
    if not os.path.isdir(jpg_dir):
        print('[Err]: invalid jpg dir.')

    if not os.path.isdir(anno_dir):
        os.makedirs(anno_dir)
    anno_file = anno_dir + '/' + 'bbox_annos.txt'

    # 格式转换: 将所有label写到一个txt文件
    i = 0
    with open(anno_file, 'w') as anno_h:
        for x in os.listdir(label_dir):
            x_path = label_dir + '/' + x
            jpg_path = x_path.replace(
                'labels', 'JPEGImages').replace('.txt', '.jpg')
            if os.path.isfile(x_path) and os.path.isfile(jpg_path):
                jpg_name = os.path.split(jpg_path)[-1]
                with open(x_path, 'r') as f_h:
                    # 写文件相对路径
                    anno_h.write(jpg_name)

                    # 写bbox_corner坐标
                    for line in f_h.readlines():
                        vertices = [x.split(',')
                                    for x in line.strip().split(' ')]
                        coorners = vertex2bbox_corner(vertices)
                        coordinates = ' ' + ' '.join(coorners)
                        anno_h.write(coordinates)
                    anno_h.write('\n')
                    i += 1
                if i % 1000 == 0:
                    print('=> %d labels processed.' % i)


def label2bbox_landmark(label_dir, anno_dir):
    """
    将车牌标注信息转换成: 路径 + bbox_corners + landmarks
    """
    if not os.path.isdir(label_dir):
        print('[Err]: invalid src dir.')
        return

    # JPEG目录
    jpg_dir = label_dir.replace('labels', 'JPEGImages')
    if not os.path.isdir(jpg_dir):
        print('[Err]: invalid jpg dir.')

    if not os.path.isdir(anno_dir):
        os.makedirs(anno_dir)
    anno_file = anno_dir + '/' + 'bbox_landmark.txt'

    # 格式转换: 将所有label写到一个txt文件
    i = 0
    with open(anno_file, 'w') as anno_h:
        for x in os.listdir(label_dir):
            x_path = label_dir + '/' + x
            jpg_path = x_path.replace(
                'labels', 'JPEGImages').replace('.txt', '.jpg')
            if os.path.isfile(x_path) and os.path.isfile(jpg_path):
                jpg_name = os.path.split(jpg_path)[-1]
                with open(x_path, 'r') as f_h:
                    # 过滤出只包含一张车牌的数据
                    # 并且写bbox_corner坐标与landmark坐标
                    lines = f_h.readlines()
                    if len(lines) == 1:
                        label = jpg_name
                        vertices = [x.split(',')
                                    for x in lines[0].strip().split(' ')]

                        # 写bbox_corner: left, right, top, bottom(x1, x2, y1, y2)
                        coorners = vertex2bbox(vertices)
                        coordinates = ' ' + ' '.join(coorners)
                        label += coordinates

                        # 写landmarks
                        landmarks = [c for pt in vertices for c in pt]
                        landmarks = ' ' + ' '.join(landmarks)
                        label += landmarks

                        # 写换行符
                        label += '\n'
                        anno_h.write(label)
                        i += 1

                if i % 1000 == 0:
                    print('=> %d labels processed.' % i)


def verify_annos(anno_path, root):
    """
    验证label转换成annotations是正确的:
    0: image path
    1-4: left, right, top, bottom of bbox => left top right down
    5-14: (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)
    """
    if not os.path.isfile(anno_path):
        print('=> [Err]: invalid anno file.')
        return

    with open(anno_path, 'r') as f_h:
        for line in f_h.readlines():
            line = line.strip().split(' ')

            # 确保数据维度正确
            assert len(line) == 13

            # 读取数据预处理
            img_name = line[0]
            corners = [int(x) for x in line[1:5]]
            landmarks = [int(float(x)) for x in line[5:]]

            img_path = root + '/' + img_name
            assert os.path.isfile(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # 画矩形
            cv2.rectangle(img,
                          (corners[0], corners[2]),  # left-up corner
                          (corners[1], corners[3]),  # right-down corner
                          (0, 0, 255),
                          thickness=2)

            # 画关键点
            cv2.circle(img, (landmarks[0], landmarks[1]), 3, (0, 255, 255), -1)
            cv2.circle(img, (landmarks[2], landmarks[3]), 3, (0, 255, 255), -1)
            cv2.circle(img, (landmarks[4], landmarks[5]), 3, (0, 255, 255), -1)
            cv2.circle(img, (landmarks[6], landmarks[7]), 3, (0, 255, 255), -1)
            # cv2.circle(img, (landmarks[8], landmarks[9]), 3, (0, 255, 255), -1)

            cv2.imshow(img_path, img)
            cv2.waitKey()


def merge2one_dir(root):
    """
    """
    if not os.path.isdir(root):
        print('=> [Err]: invalid root.')
        return

    # 创建JPEGImages目录
    jpg_dir = root + '/' + 'JPEGImages'
    if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir)

    # 合并到一个目录
    for x in tqdm(os.listdir(root)):
        x_path = root + '/' + x
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                if y.endswith('.jpg'):
                    y_path = x_path + '/' + y
                    if os.path.isfile(y_path):
                        dst_path = jpg_dir + '/' + y
                        if os.path.isfile(dst_path):
                            continue
                        shutil.copy(y_path, jpg_dir)

    # 删除其余目录
    for x in os.listdir(root):
        x_path = root + '/' + x
        if os.path.isdir(x_path):
            if x != 'JPEGImages':
                shutil.rmtree(x_path)


def gen_data_set():
    temp_dirs = []
    src_roots = ['e:/plate_data']
    dst_root = 'e:/plate_data_pro'

    for rd in src_roots:
        get_src_path(rd, temp_dirs)
    print('=> src dirs: ', temp_dirs)

    # 处理每一个src根目录
    for temp_dir in temp_dirs:
        process_batch(temp_dir, dst_root)


if __name__ == '__main__':
    # -------------------------处理一堆车牌目录
    # gen_data_set()

    # -------------中间结果测试
    # vertices = [('1686', '540'), ('1665', '625'), ('1855', '716'), ('1876', '624')]
    # vertices = format_lp_vertices(vertices)
    # print(vertices)

    # viz_vertices('e:/plate_data_pro')

    # rand_crop_around_center('e:/plate_data_pro')

    # viz_vertex(JPEG_dir='e:/plate_data_pro/CroppedImages',
    #            label_dir='e:/plate_data_pro/CroppedLabels',
    #            viz_res_dir='e:/plate_data_pro/viz_crop_result')

    # test_8_neighbor_CA()
    # test_forward()
    # test_random_choice()

    # label2anno('e:/plate_data_pro/labels', 'e:/plate_data_pro/annos')
    # merge2one_dir('f:/car_1009')
    # verify_annos(anno_path='g:/Car_DR/mtcnn_pytorch/annotations/landmark_imagelist.txt',
    #              root='f:/bbox_landmark_train')

    # label2bbox_landmark(label_dir='e:/plate_data_pro/labels',
    #                     anno_dir='e:/plate_data_pro/annos')

    # verify_annos(anno_path='e:/plate_data_pro/annos/bbox_landmark.txt',
    #              root='e:/plate_data_pro/JPEGImages')

    # ------------------ generating lp patches
    # gen_lp_patches(root='e:/plate_data_pro',
    #                offset_radius=15,
    #                min_scale=0.22,
    #                max_scale=3.0)

    # gen_lplm_patches(root='e:/plate_data_pro',
    #                  offset_radius=15,
    #                  min_scale=0.2,
    #                  max_scale=4.0,
    #                  is_viz=False)

    gen_lplm_pos_patches(src_root='e:/plate_data_pro',
                         dst_root='e:/plate_data_pro/patchesLM',
                         offset_radius=15,
                         is_viz=False)

    gen_pure_neg_patches(src_root='e:/plate_data_pro',
                         dst_root='e:/neg_patch',
                         num=5000)

    # for iter in range(5):
    #     gen_lplm_patches(root='e:/plate_data_pro',
    #                      offset_radius=12,
    #                      min_scale=0.25,
    #                      max_scale=2.5,
    #                      is_viz=False)

    #     print('=> iter %d done' % (iter + 1))
