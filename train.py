# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import json
import logging
import os, sys, math
import argparse
from collections import deque
import datetime
import traceback

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet
from dataset import get_image_id
from tool.torch_utils import do_detect

from usertool.userprint import debugPrint
from usertool.usercoco import cocoEvaluate


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):  # 一个pytorch模块，计算loss，独立于主网络
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device  # "cuda"
        self.strides = [8, 16, 32]  # 3种步长
        image_size = 608  # 图片进入网络时的大小
        self.n_classes = n_classes
        self.n_anchors = n_anchors  # 3

        # 每个都对应一个anchor，先宽后长
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            # anchor的长和宽除以步长，得到的all_anchors_grid应该是输出维度上的对应长宽，注意得到的数据类型是浮点
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]  
            # 根据mask，选出对应组的anchors，masked_anchors的shape:(3, 2)
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            # shape:(9, 4)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            # 所有anchor的长和宽赋值到ref_anchors每行的后两个位置，shape:(9, 2)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            # ref_anchors由ndarray转化为tensor
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            # 608 // 步长（3种），得到3种输出图的大小fsize
            fsize = image_size // self.strides[i]  
            # 创建0~fsize-1的x索引，并扩展复制为（B, 3, fsize, fsize）
            # 补充，tensor的repeat方法会在最后的维度上开始复制，要理解repeat的效果，最好自己亲自验证一下
            # 补充，tensor的repeat方法相较于expand方法的区别在于，repeat()会开辟新的内存
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            # 创建0~fsize-1的y索引，并扩展复制为（B, 3, fize, fsize），再交换维度变为（B, 3, fsize, fsize）
            # 补充，tensor的permute方法用于维度换位
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            # 对于3个anchor的宽度，扩展复制为（B, fsize, fsize, 3），再交换维度变为（B, 3, fsize, fize）
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            # 对于3个anchor的高度，扩展复制为（B, fsize, fsize, 3），再交换维度变为（B, 3, fsize, fize）
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            # 3组anchor的参数存放到list中，记录下来
            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        """
        创建target，核心是根据IOU选择出与标签对应的预测结果，把label的坐标转化为tx,ty,tw,th等
        """
        # pred:(B, 3, fsize, fsize, 4) 最后一个维度的四个元素代表预测框的x,y,w,h 
        # output_id表示输出序号，yolo共有3路张量输出，这个函数是处理当前路

        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels shape: (B, N, 5)，得到的nlabel shape: (B,)
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # 算出在输出维度上的x,y,w,h
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)  # (xmax+xmin)/2/s
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)  # (ymax+ymin)/2/s
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]  # (xmax-xmin)/s
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]  # (ymax-ymin)/s
        # 四舍五入取整，找到标签对应单元(i,j)
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):  # batch里的每张图片分别进行
            n = int(nlabel[b])  # 当前图片上的方框标签数
            if n == 0:
                continue  # 如果当前图片没有标签，直接跳过？
            # truth_box shape:(N, 4) x,y,w,h
            # 这里x,y保持为0，所以truth_box还在原点
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            # truth_box shape:(N, 4) x,y,w,h
            # ref_anchors shape:(9, 4) x,y,w,h
            # 这里truth_box和ref_anchors的前两列元素都是0，相当于是移到原点进行IOU的计算
            # 因为这里是计算anchor和标签的IOU，所以移到原点计算的结果是相同的，可以使计算更加简便
            # 这里的CIOU是一种最新提出的计算IOU的方式，方法内部的具体算法先暂不深究
            # 返回的anchor_ious_all shape:(N, 9)
            # 初看可能会问：这里xyxy不应该是False吗？ 画图看看就会发现，xyxy=True是左上角对齐，反之是中心对齐，都是正确的，但结果是否就完全相同可能取决于IOU是哪种
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # 找到IOU每行最大值，返回的best_n_all shape为(N,)，每个元素的值的范围为0到8
            best_n_all = anchor_ious_all.argmax(dim=1)
            # 确定与该truth_box具有最大iou的是哪一组anchor
            best_n = best_n_all % 3
            # 返回的是一个逻辑tensor，形状为(N,)，表示与该truth_box具有最大iou的anchor是否在当前组中
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue  # 如果没有符合条件的truth_box，直接跳过

            # truth_box shape:(N, 4) x,y,w,h
            # truth_box的中心移到真正的位置
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # 网络预测的box和truth_box进行iou计算
            # FIXME: 前面anchor和truth_box计算用的是CIoU=True，这里为什么改为了普通IOU?
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            # pred_iou shape:(3*fsize*fsize, N)，找到IOU每行的最大值，返回的pred_best_iou shape为(3*fsize*fsize,)
            pred_best_iou, _ = pred_ious.max(dim=1)
            # 最大IOU小于ignore_thre（0.5)的预测框过滤掉，返回的是一个形状不变的逻辑张量，还是pred_best_iou shape为(3*fsize*fsize,)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            # 将形状变换为(3,fsize,fsize)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            # 设置预测框IOU小于阈值的obj mask为1，这些是作为计算obj loss的负例
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):  # 0~N-1，N为所有标签框的数量
                if best_n_mask[ti] == 1:  # 如果当前anchor组负责该标签
                    i, j = truth_i[ti], truth_j[ti]  # 得到当前标签框的中心坐标所在单元，在输出维度上
                    a = best_n[ti]  # 0或1或2，得到组内对应的anchor序号，这其实就是通道序号
                    obj_mask[b, a, j, i] = 1  # 对应位置的obj_mask设为1
                    tgt_mask[b, a, j, i, :] = 1 # 对应位置的tgt_mask设为1
                    # 将标签的x,y,w,h转化为target
                    # 在特征维度上，x和y相当于只留下小数部分
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    # w和h先除以对应anchor的尺寸，然后取对数
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    # target的confidence设为1
                    target[b, a, j, i, 4] = 1
                    # 类别部分只把对应类的位置设为1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    # FIXME:tgt_scale为(2-w*h/(fsize*fsize))再开方，这是干什么用的？
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        # xin是yolov4网络3路输出组成的列表，每一路都是一个张量，形如（B, C, H, W），H和W相等即fsize
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):  # 提取每一路输出
            batchsize = output.shape[0]  # 第0维的大小为batchsize
            fsize = output.shape[2]  # 第2维的大小为fsize
            n_ch = 5 + self.n_classes  # 根据类别数计算通道数

            # self.n_anchors为3，将输出形状调整为（B， 3， n_classes + 5, fize, fize）
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            # 交换维度变为（B， 3, fize, fize, n_classes + 5）
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # begin of 网络输出转化为预测框的坐标和长宽
            
            # logistic activation for xy, obj, cls
            # ...是一种切片的写法
            # np.r_的用法：np.r_[:2,4:8]的结果是array([0, 1, 4, 5, 6, 7])
            # 对tx,ty,confidence,cls过一个sigmoid激活函数，数值压到（0,1）
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            # 单独把tx,ty,tw,th拿出来，这里clone()是在原有的计算图中添加一个复制的运算，得到的结果有grad_fn
            # 得到的pred结果形状为(B, 3, fsize, fsize, 4)，这里3的意思是每一路anchor的个数，yolov4里都是3
            pred = output[..., :4].clone()
            # self.grid_x[output_id]的形状为(B, 3, fsize, fsize)，所以这里是相同形状的数组相加，没有广播
            # self.grid_x[output_id]的第3个维度是0~fsize-1，对应横向W
            pred[..., 0] += self.grid_x[output_id]
            # self.grid_y[output_id]的第2个维度是0~fsize-1，对应纵向H
            pred[..., 1] += self.grid_y[output_id]
            # self.anchor_w[output_id]第1个维度有三种值，对应3种anchor的宽
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            # self.anchor_h[output_id]第1个维度有三种值，对应3种anchor的高
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            # end of 网络输出转化为预测框的坐标和长宽

            # 得到target，返回了预测框的筛选结果以及转化后的target值
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation

            # loss总体上分为两部分，obj loss, tgt loss，然后tgt loss又包含xy loss, wh loss, cls loss

            # output shape:(B， 3, fize, fize, n_classes + 5)
            # 推断出的confidence乘上一个obj_mask，作为obj的单元是乘1，不作为obj的单元变成0，后者部分相当于不参与obj loss计算（因为输出和目标全置零了）
            # 哪些地方的obj_mask为1呢，首先是最大iou小于阈值的（负例），然后加上所在单元负责某个标签框的（正例），
            # 对于前者，output是网络输出的confidence，target的confidence是0，
            # 对于后者，output是网络输出的confidence，target的confidence是1
            # 这样分析下来，这个obj_mask不仅包含正例，也包含负例
            output[..., 4] *= obj_mask
            # [0, 1, 2, 3, 5, 6, 7]对应tx,ty,tw,th,cls... 乘上tgt_mask，作为tgt的单元是乘1，不作为tgt的单元变成0，后者部分相当于不参与tgt loss计算（因为输出和目标全置零了）
            # 哪些地方的tgt_mask为1呢，只有所在单元负责某个标签框时为1
            # 只有这些tgt_mask为1的地方会计算tgt部分的loss，包括x,y,w,h和cls loss
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            # [2, 3]对应tw,th 乘上tgt_scale
            output[..., 2:4] *= tgt_scale

            # target进行相同的乘法操作
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # binary_cross_entropy计算公式：loss = -weight * ( target * torch.log(input) + (1 - target) * torch.log(1 - input) )
            #                              loss = torch.sum(loss) / torch.numel(loss)
            # 当target为1时，input越靠近1，loss越小；当target为0时，input越远离1，loss越小
            # log函数当自变量为(0,1)时，因变量的结果是(0,无穷)，-log(0.5)=1

            # x和y的loss，输入input和target形状为(B, 3, fsize, fsize, 2)，有效项个数：num_truth_boxes
            # tgt_scale*tgt_scale 的值是(2-w*h/(fsize*fsize))
            # target[..., :2]也是小数
            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            # w和h的loss，输入input和target的形状为(B, 3, fsize, fsize, 2)，有效项个数：num_truth_boxes
            # tw和th求的是mse loss，即L2 loss，因为tw和th可能是负的
            # 它们在求loss之前乘了一个系数tgt_scale，相当于总的loss也乘了一个tgt_scale * tgt_scale=(2-w*h/(fsize*fsize))
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            # confidence的loss，输入input和target的形状为(B, 3, fsize, fsize,)，有效项个数约为 fsize*fsize/2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            # cls的loss，输入input和target的形状为(B, 3, fsize, fsize, num_classes)，有效项个数：num_truth_boxes
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            # mse loss也称L2 loss，每个位置的元素各自相减，然后全部求平方和
            # 注意这个loss不计入总的loss
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


# def train_collate(batch):
#     # type(batch): list
#     # len(batch): batch_size
#     images = []
#     bboxes = []
#     for img, box in batch:
#         images.append([img])
#         bboxes.append([box])
#     images = np.concatenate(images, axis=0)
#     images = images.transpose(0, 3, 1, 2)
#     images = torch.from_numpy(images).div(255.0)
#     bboxes = np.concatenate(bboxes, axis=0)
#     bboxes = torch.from_numpy(bboxes)
#     return images, bboxes


def val_collate(batch):
    return tuple(zip(*batch))


def makeTgtJson(val_loader, categories):
    os.makedirs("./tmp", exist_ok=True)
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 
                'id': cid, 
                'name': cate
                }
        json_dict['categories'].append(cat)
    tgtFile = "./tmp/tgt.json"

    for images, targets in val_loader:
        for img, target in zip(images, targets):
            # targets是一个列表，里面的每个成员是字典
            # "boxes": tensor shape(N, 4) xmin, ymin, xmax, ymax
            # "labels": tensor shape(N,)
            # "image_id": tensor shape (1,)
            # "area": tesor shape (N,) 表示框的面积
            # "iscrowd": tensor shape(N,) 全零
            image_id = int(target["image_id"][0])
            height, width = img.shape[:2]
            image = {'file_name': "0", 
                    'height': height, 
                    'width': width,
                    'id':image_id
                    }
            json_dict['images'].append(image)

            for i, box in enumerate(target["boxes"]):
                # box shape (4,)  xmin,ymin,xmax,ymax
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2])
                ymax = int(box[3])
                assert(xmax > xmin)
                assert(ymax > ymin)
                o_width = xmax - xmin
                o_height = ymax - ymin
                ann = {'area': o_width*o_height, 
                    'iscrowd': 0, 
                    'image_id': image_id,
                    'bbox':[xmin, ymin, o_width, o_height],
                    'category_id': int(target["labels"][i]),
                    'id': bnd_id,
                    'ignore': 0,
                    'segmentation': []
                    }
                json_dict['annotations'].append(ann)
                bnd_id += 1

    with open(tgtFile, 'w') as json_fp:
        # 将字典转化为json
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)

    return tgtFile


def evaluate(model, val_label, val_dataset_dir, use_cuda, net_width, net_height):
    os.makedirs("./tmp", exist_ok=True)
    resFile = "./tmp/res.json"

    f = open(val_label, 'r', encoding='utf-8')
    truth = {}
    for line in f.readlines():
        data = line.split(" ")  # 以空格分开不同目标
        truth[data[0]] = []  # data[0]是图片名称（xxx.jpg）
        for i in data[1:]:
            # 每一项是一个列表，[x1,y1,x2,y2,cls_id],列表中的元素全部为int类型
            truth[data[0]].append([int(float(j)) for j in i.split(',')])
    
    imgs = list(truth.keys())  # 列表
    # net_width = model.width
    # net_height = model.height
    # if use_cuda:
    #     model.cuda()
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    boxes_json = []
    for i, image_file_name in enumerate(imgs):
        image_id = get_image_id(image_file_name)
        img = cv2.imread(os.path.join(val_dataset_dir, image_file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width = img.shape[:-1]
        sized = cv2.resize(img, (net_width, net_height), cv2.INTER_LINEAR)
        start = time.time()
        boxes = do_detect(model, sized, 0.0, 0.6, use_cuda)
        finish = time.time()

        assert type(boxes[0]) == list
        for box in boxes[0]:
            box_json = {}
            # xmin,ymin,xmax,ymax -> xmin,ymin,w,h
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            category_id = box[-1]
            score = box[-2]
            bbox_normalized = box[:4]
            box_json["category_id"] = int(category_id)
            box_json["image_id"] = int(image_id)
            bbox = []
            for i, bbox_coord in enumerate(bbox_normalized):
                modified_bbox_coord = float(bbox_coord)
                if i % 2:
                    modified_bbox_coord *= image_height
                else:
                    modified_bbox_coord *= image_width
                modified_bbox_coord = round(modified_bbox_coord, 2)
                bbox.append(modified_bbox_coord)
            box_json["bbox_normalized"] = list(map(lambda x: round(float(x), 2), bbox_normalized))
            box_json["bbox"] = bbox
            box_json["score"] = round(float(score), 2)
            box_json["timing"] = float(finish - start)
            boxes_json.append(box_json)
    if len(boxes_json) == 0:
        return None
    with open(resFile, 'w') as outfile:
        json.dump(boxes_json, outfile)

    return resFile   


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    # TODO:加上resume功能，resume需要什么信息？
    # config的所有信息、yolov4-custom.cfg的所有信息，权重，epoch序号，学习率到哪了
    
    
    # 创建dataset
    # config.train_label为data/coins.txt标签文本的路径
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    # 获得dataset的长度
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 创建dataloader
    # 当pin_memory=False,num_workers=0（子进程数量为0，即只有主进程）时，正常
    # 当pin_memory=True,num_workers=8时，卡住
    # 当pin_memory=False,num_workers=8时，卡住
    # 当pin_memory=True,num_workers=0时，正常
    # 综上，原因在于num_workers大于0开启多线程导致
    # 经查，dataset加载图片中使用OpenCV，OpenCV某些函数默认也会开多线程，
    # 多线程套多线程，容易导致线程卡住（是否会卡住可能与不同操作系统有关）
    # 解决方法：法一，在dataset的前面import cv2时加上cv2.setNumThreads(0)禁用OpenCV多进程（推荐）
    #          法二，使用PIL加载和预处理图片（不推荐，PIL速度不如OpenCV）
    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=False, collate_fn=val_collate)
                            
    if config.only_evaluate or config.evaluate_when_train:
        tgtFile = makeTgtJson(val_loader, config.categories)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    
    # 计算迭代次数的最大值
    max_itr = config.TRAIN_EPOCHS * n_train
    
    # 迭代次数的全局计数器
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:      {config.pretrainedWeight is not None or config.Pretrained is not None}
    ''')
    if config.only_evaluate:
        if config.use_darknet_cfg:
            eval_model = Darknet(config.cfgfile)
        else:
            raise NotImplementedError
        if torch.cuda.device_count() > 1:
            eval_model.load_state_dict(model.module.state_dict())
        else:
            eval_model.load_state_dict(model.state_dict())
        eval_model.to(device)
        eval_model.eval()
        resFile = evaluate(eval_model, config.val_label, config.dataset_dir, device==torch.device("cuda"))
        if resFile is None:
            debugPrint("detect 0 boxes in the val set")
            return
        cocoEvaluate(tgtFile, resFile)
        return

    # learning rate setup
    # 自定义的学习率调整函数，先递增，然后阶梯性降低
    def burnin_schedule(i):
        # i表示iter，而不是epoch
        if i < config.burn_in:  # 按4次方递增阶段
            # factor表示乘在学习率上的倍数
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:  # 第一阶段
            factor = 1.0
        elif i < config.steps[1]:  # 第二阶段
            factor = 0.1
        else:  # 第三阶段
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':  # 默认是adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,  # 学习率的实际值是设置值/batch_size
            betas=(0.9, 0.999),  # adam的特殊参数，一般用默认即可
            eps=1e-08,  # adam的特殊参数，一般用默认即可
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )

    # pytorch调整学习率的专用接口
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # 计算loss的对象，这个模块是在yolo网络后专门求解loss的（yolo主网络只负责接收图片，然后输出三路张量），这个模块不需要权重等参数
    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_step = 0
        model.train()
        logging.info("===Train===")
        for i, batch in enumerate(train_loader):
            global_step += 1
            epoch_step += 1
            images = batch[0]
            bboxes = batch[1]

            images = images.to(device=device, dtype=torch.float32)
            bboxes = bboxes.to(device=device)

            bboxes_pred = model(images)
            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
            loss.backward()

            epoch_loss += loss.item()

            if global_step % config.subdivisions == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            logging.info("Epoch:[{:3}/{}],step:[{:3}/{}],total loss:{:.2f}|lr:{:.5f}".format(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), scheduler.get_last_lr()[0]))

            if global_step % (log_step * config.subdivisions) == 0:  # log_step默认为20，这里指的是迭代次数
                
                writer.add_scalar('train/Loss', loss.item(), global_step)
                writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                writer.add_scalar('lr', scheduler.get_last_lr()[0] * config.batch, global_step)
                
                logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                            'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                            .format(global_step, loss.item(), loss_xy.item(),
                                    loss_wh.item(), loss_obj.item(),
                                    loss_cls.item(), loss_l2.item(),
                                    scheduler.get_last_lr()[0] * config.batch))
        if save_cp:  # True
            # 创建checkpoints文件夹
            if not os.path.exists(config.checkpoints):
                os.makedirs(config.checkpoints, exist_ok=True)  # exist_ok=True表示可以接受已经存在该文件夹，当exist_ok=False时文件夹存在会抛出错误
                logging.info('Created checkpoint directory')
            save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.weights')                
            # 考虑torch.nn.DataParallel特殊情况
            if torch.cuda.device_count() > 1:
                model.module.save_weights(save_path)
            else:
                model.save_weights(save_path)                
            logging.info(f'Checkpoint {epoch + 1} saved !')
            # 只保留最新keep_checkpoint_max个checkpoint，自动删除较早的checkpoint
            saved_models.append(save_path)
            if len(saved_models) > config.keep_checkpoint_max > 0:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    logging.info(f'failed to remove {model_to_remove}')

        if config.evaluate_when_train:
            try:
                model.eval()
                resFile = evaluate(model, config.val_label, config.dataset_dir, device==torch.device("cuda"), config.width, config.height)
                if resFile is None:
                    continue
                stats = cocoEvaluate(tgtFile, resFile)

                logging.info("===Val===")
                logging.info("Epoch:[{:3}/{}],AP:{:.3f}|AP50:{:.3f}|AP75:{:.3f}|APs:{:.3f}|APm:{:.3f}|APl:{:.3f}".format(
                    epoch + 1, epochs, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]))
                logging.info("Epoch:[{:3}/{}],AR1:{:.3f}|AR10:{:.3f}|AR100:{:.3f}|ARs:{:.3f}|ARm:{:.3f}|ARl:{:.3f}".format(
                    epoch + 1, epochs, stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]))


                writer.add_scalar('train/AP', stats[0], global_step)
                writer.add_scalar('train/AP50', stats[1], global_step)
                writer.add_scalar('train/AP75', stats[2], global_step)
                writer.add_scalar('train/AP_small', stats[3], global_step)
                writer.add_scalar('train/AP_medium', stats[4], global_step)
                writer.add_scalar('train/AP_large', stats[5], global_step)
                writer.add_scalar('train/AR1', stats[6], global_step)
                writer.add_scalar('train/AR10', stats[7], global_step)
                writer.add_scalar('train/AR100', stats[8], global_step)
                writer.add_scalar('train/AR_small', stats[9], global_step)
                writer.add_scalar('train/AR_medium', stats[10], global_step)
                writer.add_scalar('train/AR_large', stats[11], global_step)
            except Exception as e:
                debugPrint("evaluate meets an exception, here is the exception info:")
                traceback.print_exc()
                debugPrint("ignore error in evaluate and continue training")

    writer.close()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help="GPU, if multi, use ',' to separate", dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137.pth')
    parser.add_argument('-pretrainedWeight', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")

    args = vars(parser.parse_args())
    cfg.update(args)

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    # 创建log目录和log文本文件
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    debugPrint('log file path:' + log_file)

    # logging是一个库，相当于高级版的print，这里是一些logging初始化工作
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    # 初始化logger，用于记录训练过程
    logging = init_logger(log_dir='log')

    # 超参设置总结：当cfg.py中的cfg.use_darknet_cfg为1时，超参来自三部分：
    # 命令行、cfg.py和cfg/yolov4-custom.cfg，
    # 其中cfg.py是控制训练和验证的超参（例如batchsize等)，某些参数可以被命令行覆盖（详见cfg.py里的注释）
    # 而cfg/yolov4-custom.cfg中有用的只是网络结构本身的超参设置（例如每一层的卷积核大小等）

    # 当cfg.py中的cfg.use_darknet_cfg为0时，超参来自命令行和cfg.py，此时网络结构直接Yolov4使用这个模块
    # 之所以这样，是为了兼容darknet中形如cfg/yolov4-custom.cfg的网络设置

    # Cfg来自cfg.py,这个函数可以用命令行参数去更新Cfg中的某几项
    cfg = get_args(**Cfg)  

    # 设置GPU用哪几个，cfg.gpu这个参数来自命令行
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        # 当use_darknet_cfg时，cfg.pretrained就没用了
        logging.info("Using darknet cfg")
        model = Darknet(cfg.cfgfile)
        if cfg.pretrained is not None and cfg.pretrainedWeight is None:
            raise ValueError("Darknet can't load Pytorch weights")
        elif cfg.pretrainedWeight is None:
            raise ValueError("Please specify pretrainedWeight file")
        model.load_weights(cfg.pretrainedWeight)
    else:
        # 当不是use_darknet_cfg时，cfg/yolov4.cfg等就没用了
        logging.info("Not using darknet cfg")
        if cfg.pretrainedWeight is not None and cfg.pretrained is None:
            raise ValueError("Pytorch can't load Darknet weights")
        elif cfg.pretrained is None:
            raise ValueError("Please specify pretrained file")
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        # 如果GPU数量大于1，将模型转为并行
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,  # 300，在cfg.py中改
              device=device, 
              save_cp=True,
              )
    except KeyboardInterrupt:
        # 在训练过程中，捕捉ctrl+c中断，保存模型到INTERRUPTED.pth
        if torch.cuda.device_count() > 1:
            model.module.save_weights("INTERRUPTED.weights")
        else:
            model.save_weights("INTERRUPTED.weights")    
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
