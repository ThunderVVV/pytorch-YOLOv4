# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@Fise          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# EasyDict是一个很好用的库，用它定义的字典，每一项可以以属性的方式访问
Cfg = EasyDict()

Cfg.use_darknet_cfg = True
# 当Cfg.use_darknet_cfg时，启用Cfg.cfgfile里面的参数，[net]里只有部分参数有效，详见yolov4-custom.cfg里的注释
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4-custom.cfg') 

Cfg.only_evaluate = 0
Cfg.evaluate_when_train = 1

Cfg.batch = 2  # 真正的batchsize是Cfg.batch/Cfg.subdivisions
Cfg.subdivisions = 1
Cfg.width = 608  # 如果use_darknet_cfg，必须和对应*.cfg中的width相同
Cfg.height = 608  # 如果use_darknet_cfg，必须和对应*.cfg中的height相同
Cfg.channels = 3  # 只能是3，不可更改
Cfg.momentum = 0.949  # 当Cfg.TRAIN_OPTIMIZER为sgd时，作为sgd优化器的参数
Cfg.decay = 0.0005  # 当Cfg.TRAIN_OPTIMIZER为sgd时，作为sgd优化器的参数
Cfg.angle = 0  # 没用
Cfg.saturation = 1.5  # 用于HSV颜色增强
Cfg.exposure = 1.5  # 用于HSV颜色增强
Cfg.hue = .1  # 用于HSV颜色增强

Cfg.learning_rate = 0.001    # 最终的学习率还要除以batch size                         # 可以被命令行的值覆盖
Cfg.burn_in = 60  # 指学习率逐渐增长到设定值的分界点，iter而不是epoch
Cfg.max_batches = 500500  # 没用
Cfg.steps = [1200, 1500]  # 指学习率阶梯更新分界点，iter而不是epoch
Cfg.policy = Cfg.steps  # 没用
Cfg.scales = .1, .1  # 没用

Cfg.cutmix = 0  # cutmix是一种利用多图混合的数据增强算法
Cfg.mosaic = 1  # mosaic是cutmix的升级版

Cfg.letter_box = 0
Cfg.jitter = 0.2  # 用于随机裁剪数据增强
Cfg.classes = 80  # 类别数量                                                          # 可以被命令行的值覆盖
Cfg.track = 0  # 没用
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1  # 打开随机翻转的数据增强
Cfg.blur = 0  # 一种数据增强，这里关闭
Cfg.gaussian = 0  # 一种数据增强，这里关闭
Cfg.boxes = 60  # 标注框最大数量
Cfg.TRAIN_EPOCHS = 300  # 训练的epoch数
Cfg.TRAIN_OPTIMIZER = 'adam'  # 优化器种类
Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')  # 训练集标注文件的路径
Cfg.val_label = os.path.join(_BASE_DIR, 'data' ,'val.txt')  # 验证集标注文件的路径
Cfg.categories = {
    "1yuan":0,
    "5jiao":1,
    "1jiao":2
}
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:  # 默认用的是mosaic数据增强，Cfg.mixup为3
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')  # 保存checkpoints的文件夹
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10    # 只保留最新keep_checkpoint_max个checkpoint，自动删除较早的checkpoint
