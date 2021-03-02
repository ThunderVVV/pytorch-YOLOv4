import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.region_loss import RegionLoss
from tool.yolo_layer import YoloLayer
from tool.config import *
from tool.torch_utils import *

from usertool.userprint import debugPrint


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        # 这里的意思用人话说就是处理darknet和pytorch计算maxpool时padding的方法不同：
        # pytorch是两边padding相同长度（padding=k//2），然后划窗，但这样可能导致滑窗到最后不对齐
        # darknet是自适应padding，padding总长为k-1，两边padding长度可能相同，也可能相差1（为了使滑窗到最后对齐）
        p = self.size // 2
        # 上下方向自适应padding
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        # 左右方向自适应padding
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        # pad方式用的是replicate，将矩阵的边缘复制并填充到矩阵的外围
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        # x: (B, C, H, W)
        
        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous().\
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x


class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, inference=False):
        super(Darknet, self).__init__()
        # 表示模型是用于训练还是用于推断
        self.inference = inference
        self.training = not self.inference

        # 将cfg文件中的参数解析，self.blocks是一个列表，列表中每个成员是一个字典
        self.blocks = parse_cfg(cfgfile)  
        self.width = int(self.blocks[0]['width'])  # 608
        self.height = int(self.blocks[0]['height'])  # 608

        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0, 0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        for block in self.blocks:
            ind = ind + 1
            # if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':  # yolo中的一个特殊层，作用是路由和拼接
                layers = block['layers'].split(',')
                # 如果layers索引大于0，则代表层数，如果小于0，代表从当前层往前减
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:  # 如果layers只有一个
                    if 'groups' not in block.keys() or int(block['groups']) == 1:  # 对于yolov4.cfg，为True
                        # 当layers只有一个时，根据索引连接
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    # 如果layer有2个，在通道维度拼接
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    # 如果layer有4个，在通道维度拼接
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':  # 残差结构
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                # 将上一层的输出和from层的输出相加
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                # yolov4.cfg中的shortcut全部是linear，没有激活
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'yolo':
                # if self.training:
                #     pass
                # else:
                #     boxes = self.models[ind](x)
                #     out_boxes.append(boxes)
                
                # yolov4.cfg有3个yolo，所以out_boxes包含三个结果，每个结果的形状为(B, C, H, W)
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        if self.training:
            return out_boxes
        else:
            # 返回：[boxes, confs]
            return get_region_boxes(out_boxes)

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()  # 创建一个空的ModuleList

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            if block['type'] == 'net':  # 只有一个net
                prev_filters = int(block['channels'])  # 3
                continue
            elif block['type'] == 'convolutional':  # 卷积层，yolov4.cfg共有110个卷积层
                conv_id = conv_id + 1  # 按顺序将卷积层编号，后面进行模块命名
                # 得到该卷积层的各项参数
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                # darknet只有两种kernel_size，若kernel_size=1,则pad=0 若kernel_size=3,则pad=1
                # 补充：output_size =1+ (input_size+2*padding-kernel_size)/stride
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()  # 创建一个空的Sequential
                # 公式：卷积层=ConvBnAct，有个别卷积层没有BN，有的层没有激活（linear）
                if batch_normalize:
                    # 有BN，不需要偏置
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    # 没有BN，需要偏置（nn.Conv2d中的bias默认为True）
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    # inplace表示原地操作，在原变量不存在分支的情况下是没问题的，可以节省内存
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    # yolov4里没有用relu
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    # yolov4里除了leaky和mish，就是linear，表示没有激活层
                    # 这个打印信息没用,改成空语句pass
                    # print("convolution haven't activate {}".format(activation))
                    pass

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':  # 最大池化层，yolov4.cfg只有3个
                # 池化层只有两个参数，核大小和步长
                pool_size = int(block['size'])  # yolov4三个最大池化层的步长分别为5,9,13
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    # yolov4用的都是MaxPoolDark
                    model = MaxPoolDark(pool_size, stride)
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':  # yolov4.cfg里没有avgpool
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':  # yolov4.cfg里没有softmax
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':  # yolov4.cfg里没有cost
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':  # yolov4.cfg里没有reorg
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':  # yolov4.cfg里有两个upsample
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                models.append(Upsample_expand(stride))  # Upsample_expand上采样
                # models.append(Upsample_interpolate(stride))

            elif block['type'] == 'route':  # yolo中的一个特殊层，作用是路由和拼接
                layers = block['layers'].split(',')
                ind = len(models)
                # 如果layers索引大于0，则代表层数，如果小于0，代表从当前层往前减
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:  # 如果layers只有一个
                    if 'groups' not in block.keys() or int(block['groups']) == 1:  # 对于yolov4.cfg，为True
                        # 获取指定layer的核数量和步长
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:  # 如果layer有两个
                    # 如果layer有两个，肯定有一个是-1
                    assert (layers[0] == ind - 1 or layers[1] == ind - 1)
                    # 核数量为两个层的核数量之和
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:  # 如果layer有4个
                    # 如果layer有四个，肯定有一个是-1
                    assert (layers[0] == ind - 1)
                    # 核数量为四个层的核数量之和
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + \
                                   out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                # 设置虚空的核数量和步长，以及用于占位的空层
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':  # yolov4里有23个shortcut
                ind = len(models)
                # 设置虚空的核数量和步长，以及用于占位的空层
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':  # yolov4.cfg里没有connected
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':  # yolov4.cfg里没有region
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':  # yolov4.cfg里有3个yolo
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                self.num_classes = yolo_layer.num_classes
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                yolo_layer.scale_x_y = float(block['scale_x_y'])
                # yolo_layer.object_scale = float(block['object_scale'])
                # yolo_layer.noobject_scale = float(block['noobject_scale'])
                # yolo_layer.class_scale = float(block['class_scale'])
                # yolo_layer.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')  # r:读模式打开 b:二进制文件
        # 按int32的方式，读取前5个数，这5个数包含了文件的一些版本等信息，没啥用
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':  # 第一层是net，无参数
                continue
            elif block['type'] == 'convolutional':  # 卷积层，yolov4.cfg里只有卷积层有参数
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':  # 表示全连接层，yolov4.cfg里没有connected
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            # 其它层没有参数
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        debugPrint("{} weights loaded".format(weightfile))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
    
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
    
        ind = -2
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                pass
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
