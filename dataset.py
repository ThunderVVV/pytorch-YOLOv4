# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    # random.random()返回随机生成的一个实数，它在[0,1)范围内
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    # bboxes(ndarray): shpae is (N, 5)
    # num_boxes(int): 60
    # classes(int): 3
    # flip(int): 1
    # dx(int): 假设为 100,代表左偏移量
    # dy(int): 假设为 100,代表上偏移量
    # sx(int): 假设原图大小为1216，则sx为1216-左偏移量-右偏移量
    # sy(int): 假设原图大小为1216，则sy为1216-上偏移量-下偏移量
    # net_w(int): 608
    # net_h(int): 608


    if bboxes.shape[0] == 0:
        return bboxes, 10000
    # 打乱bboxes顺序
    np.random.shuffle(bboxes)

    # begin of bboxes随图片裁剪
    # bbox的坐标减去左偏移量和上偏移量
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    # np.clip是一个截取函数，用于截取数组中小于0或者大于sx的部分，得到的结果一定处于[0,sx]范围
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    # 删除经过裁剪后超出范围的框    
    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    # end of bboxes随图片裁剪

    # 如果没有任何框，直接返回
    if bboxes.shape[0] == 0:
        return bboxes, 10000

    # 选择cls_id处于0~num_classes-1的框
    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    # num_boxes为60，一张图最多支持60个bbox标签
    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    # 得到bbox的长和宽的最小值
    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    # begin of bboxes随图片缩放
    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)
    # end of bboxes随图片缩放

    # begin of bboxes随图片翻转
    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp
    # end of bboxes随图片翻转

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    try:
        # begin of 按照已生成的随机参数裁剪
        img = mat  # 原始尺寸的img
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop

        # (0,0)表示在源坐标系下的原点
        # pleft>0, ptop>0  补充：pright<0,pbot<0
        # ------------------------
        # |(0,0)img_rect         |
        # |    ------------------+--  ---
        # |    |                 | |   ^
        # |    |   new_src_rect  | |   |
        # |    |                 | |   |
        # |    |                 | |  sheight
        # -----+------------------ |   |
        #      |     src_rect      |   v
        #      ---------------------  ---
        #
        #      |<-----swidth------>|

        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        # 将new_src_rect的坐标转移到目标坐标系下
        # 当pleft>0, ptop>0时，dst_rect为[0, 0, new_src_rect[2] - new_src_rect[0], new_src_rect[3] - new_src_rect[1]]
        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            cropped = np.zeros([sheight, swidth, 3])
            # 在每个通道的平面图上，求出全图的平均值
            cropped[:, :, ] = np.mean(img, axis=(0, 1))

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # end of 按照已生成的随机参数裁剪

            # begin of 缩放
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)
            # end of 缩放

        # begin of 随机左右翻转
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
        # end of 随机左右翻转

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        # begin of HSV颜色增强
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:  # True
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
                # end of HSV颜色增强
            else:
                sized *= dexp

        if blur:  # False
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:  # False
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except KeyboardInterrupt:
        pass
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    """
    用于在mosaic拼接时过滤越界的bboxes
    """
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    # i_mixup是序号，这里是0~3
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        # cfg.mixup为3
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")  # 以空格分开不同目标
            truth[data[0]] = []  # data[0]是图片名称（xxx.jpg）
            for i in data[1:]:
                # 每一项是一个列表，[x1,y1,x2,y2,cls_id],列表中的元素全部为int类型
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth  # 字典
        self.imgs = list(self.truth.keys())  # 列表

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        """
        根据索引，获得图片和标注框

        Args:
            index(int): 索引
        Returns:
            out_img(ndarray): 形状为(self.cfg.h, self.cfg.w, 3)
                self.cfg.h and self.cfg.w 是网络输入的高和宽
            out_bboxes1(ndarray): 形状为(self.cfg.boxes, 5)
                self.cfg.boxes是设定的标注框最大数量

        """
        # getitem时间越长，从dataloader取数据的时间越长，这个时间位于每个iter开始部分
        if not self.train:
            # 因为把train和val的dataset类写到一起，但train和val预处理肯定是不同的，所以在内部分成两个函数
            return self._get_val_item(index)
        img_path = self.imgs[index]  # xxx.jpg
        # ndarray[x1,y1,x2,y2],这个地方注意np.float和np.double的结果都是float64
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)  
        img_path = os.path.join(self.cfg.dataset_dir, img_path)  # 把cfg.dataset_dir和img_path拼接起来
        use_mixup = self.cfg.mixup  # 3
        if random.randint(0, 1):  # 返回[0,1]范围的随机整数，0.5的概率不用mixup
            use_mixup = 0

        if use_mixup == 3:  # True
            min_offset = 0.2
            # 在0.2w和0.8w之间随机取一个整数点作为cut_x
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])  # 图片数据shape（H,W,3）
        out_bboxes = []

        # begin of 多图顺序处理并组成mosaic
        for i in range(use_mixup + 1):  # 0~3  因为使用mosaic数据增强，这里是4张图进行mosaic
            if i != 0:
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)
                img_path = os.path.join(self.cfg.dataset_dir, img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            oh, ow, oc = img.shape
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)

            # 返回-cfg.hue到cfg.hue的随机实数
            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
            # 一半概率返回1~cfg.saturation的随机实数，一半概率返回1/(1~cfg.saturation的随机实数)
            dsat = rand_scale(self.cfg.saturation)
            # 一半概率返回1~cfg.exposure的随机实数，一半概率返回1/(1~cfg.exposure的随机实数)
            dexp = rand_scale(self.cfg.exposure)

            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            # cfg.flip为1,一半概率启用flip
            flip = random.randint(0, 1) if self.cfg.flip else 0

            if (self.cfg.blur):  # False
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:  # True
                gaussian_noise = 0

            if self.cfg.letter_box:  # False
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            # bbox随图片裁剪、缩放、翻转
            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            # 按照已生成的随机参数裁剪，缩放，随机左右翻转，HSV颜色增强
            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if use_mixup == 0:
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:  # True
                if flip:  # 一半概率为True，一半概率为False
                    # pleft和pright对调
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                # mosaic数据增强讲解：https://www.cnblogs.com/xiamuzi/p/13471396.html

                # 以image_data_augmentation中的示意图为例
                # 当pleft>0 ptop>0 pright<0 pbot<0时
                # left_shift必为0，top_shift必为0，right_shift为min(w-cut_x, pright*w/sw) bot_shift为min(h-cut_y, pbot*h/sh)
                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

                right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
                bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        # end of 多图顺序处理并组成mosaic
        
        if use_mixup == 3:  # True
            out_bboxes = np.concatenate(out_bboxes, axis=0)

        # out_bboxes1的形状固定为(cfg.boxes,5)，不够的补零
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
#     raise NotImplementedError("Create your own 'get_image_id' function")
    lv, no = os.path.splitext(os.path.basename(filename))[0].split("-")
    print(lv + " " + no)
    return int(lv[1:] + no)


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    Cfg.dataset_dir = '/mnt/e/Dataset'
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(a.astype(np.int32))
        plt.show()
