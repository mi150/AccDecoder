import torch
from torch.distributions import Categorical, Distribution
from typing import List
class MultiCategorical(Distribution):

    def __init__(self, dists: List[Categorical]):
        super().__init__(validate_args=False)
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

class multi_categorical_maker:
    def __init__(self, nvec):
        self.nvec = nvec

    def __call__(self, logits):
        start = 0
        ans = []
        for n in self.nvec:
            ans.append(Categorical(logits=logits[:, start: start + n]))
            start += n
        return MultiCategorical(ans)

# def multi_categorical_maker(nvec):
#     def get_multi_categorical(logits):
#         start = 0
#         ans = []
#         for n in nvec:
#             ans.append(Categorical(logits=logits[:, start: start + n]))
#             start += n
#         return MultiCategorical(ans)
#     return get_multi_categorical
import numpy as np
# -*- coding: utf-8 -*-
"""
@File    : 20200119_图像下采样测试.py
@Time    : 2020/1/19 9:35
@Author  : Dontla
@Email   : sxana@qq.com
@Software: PyCharm
"""
import cv2
img = cv2.imread('a.png')
img1= cv2.imread('b.png')
# # print(img.shape)    # (1280, 1920, 3)
# # for i in range(100):
# img = cv2.pyrUp(img)
#     #img = cv2.pyrDown(img)
# # print(img.shape)    # (320, 480, 3)
# cv2.imshow('win', img)
# cv2.imwrite('b.png',img)
# cv2.waitKey(0)
def get_frame_feature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5),
                            0)
    # 边缘检测
    # gray_lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    # dst = cv2.convertScaleAbs(gray_lap)
    edge = cv2.Canny(gray, 101, 255)
    return edge


# 对边缘进行差值计算
def cal_frame_diff( edge, prev_edge):
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff,21, 255,
                               cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed
# f1=get_frame_feature(img)
# f2=get_frame_feature(img1)
# a=cal_frame_diff(f1,f2)
# print(a)