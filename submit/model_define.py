import os
import torch.nn.functional as F
from collections import OrderedDict
import torch
import math
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
from resnest.torch import resnest101, resnest50

import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ALIGN_CORNERS = True

class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

def conv3x3_2(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck2(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

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

        out += residual
        out = self.relu(out)

        return out

class ResNet2(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=True, norm_layer=nn.BatchNorm2d, multi_grid=False,
                 multi_dilation=None):
        self.inplanes = 64
        super(ResNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            if multi_grid:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False,
                    multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if multi_grid == False:
            if dilation == 1 or dilation == 2:
                layers.append(block(self.inplanes, planes, stride, dilation=1,
                                    downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2,
                                    downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div], previous_dilation=dilation,
                                    norm_layer=norm_layer))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet101_2(pretrained=False, **kwargs):
    model = ResNet2(Bottleneck2, [3, 4, 23, 3], **kwargs)
    return model

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, dilated=True, norm_layer=None, pretrained=True):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.pretrained = resnet101_2(pretrained=False, dilated=dilated, norm_layer=norm_layer)

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

class DANet(BaseNet):

    def __init__(self, nclass, backbone, pretrained=True, norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.head = DANetHead(2048, 512, norm_layer)

        self.seg1 = nn.Sequential(nn.Dropout(0.1),
                                  nn.Conv2d(512, nclass, 1))

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, _, c4 = self.base_forward(x)

        x = self.head(c4)

        output = self.seg1(x)
        output = F.interpolate(output, imsize, mode='bilinear', align_corners=True)

        return output

class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        feat_sum = sa_conv + sc_conv

        return feat_sum

# for deeplab below
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride=16, norm_layer=nn.BatchNorm2d):
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        elif output_stride is None:
            dilations = [1, 1, 1, 1]
            strides = [1, 2, 2, 2]
        else:
            raise Warning("output_stride must be 8 or 16 or None!")
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       norm_layer=norm_layer)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _load_pretrained_model(self, pretrain_dict):
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ResNeSt(nn.Module):
    def __init__(self, backbone="resnest101", norm_layer=nn.BatchNorm2d, pretrained=False):
        super(ResNeSt, self).__init__()
        if backbone == 'resnest50':
            self.pretrained = resnest50(pretrained=False, dilated=True, norm_layer=norm_layer)
        elif backbone == 'resnest101':
            self.pretrained = resnest101(pretrained=False, dilated=True, norm_layer=norm_layer)

    def forward(self, x):

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c4

def resnet101(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet101-5d3b4d8f.pth"))
    return model

class ASPPBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=256, os=16, norm_layer=nn.BatchNorm2d):
        '''
        :param in_channel: default 2048 for resnet101
        :param out_channel: default 256 for resnet101
        :param os: 16 or 8
        '''
        super(ASPPBlock, self).__init__()
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.gave_pool = nn.Sequential(
            OrderedDict([('gavg', nn.AdaptiveAvgPool2d(rates[0])),
                         ('conv0_1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_1', norm_layer(out_channel)),
                         ('relu0_1', nn.ReLU(inplace=True))])
        )
        self.conv1_1 = nn.Sequential(
            OrderedDict([('conv0_2', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_2', norm_layer(out_channel)),
                         ('relu0_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra1 = nn.Sequential(
            OrderedDict([('conv1_1', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[1], dilation=rates[1], bias=False)),
                         ('bn1_1', norm_layer(out_channel)),
                         ('relu1_1', nn.ReLU(inplace=True))])
        )
        self.aspp_bra2 = nn.Sequential(
            OrderedDict([('conv1_2', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[2], dilation=rates[2], bias=False)),
                         ('bn1_2', norm_layer(out_channel)),
                         ('relu1_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra3 = nn.Sequential(
            OrderedDict([('conv1_3', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[3], dilation=rates[3], bias=False)),
                         ('bn1_3', norm_layer(out_channel)),
                         ('relu1_3', nn.ReLU(inplace=True))])
        )
        self.aspp_catdown = nn.Sequential(
            OrderedDict([('conv_down', nn.Conv2d(5 * out_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn_down', norm_layer(out_channel)),
                         ('relu_down', nn.ReLU(inplace=True)),
                         ('drop_out', nn.Dropout(.1))])
        )

    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), size[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, self.conv1_1(x),
                       self.aspp_bra1(x),
                       self.aspp_bra2(x),
                       self.aspp_bra3(x)], dim=1)
        x = self.aspp_catdown(x)
        return x

class DeepLabv3_plus(nn.Module):
    def __init__(self, in_channels, num_classes, backend='resnet101', os=16, pretrained=False,
                 norm_layer=nn.BatchNorm2d):
        '''
        :param in_channels:
        :param num_classes:
        :param backend: only support resnet, otherwise need to have low_features
                        and high_features methods for out
        '''
        super(DeepLabv3_plus, self).__init__()
        self.in_channes = in_channels
        self.num_classes = num_classes
        if hasattr(backend, 'low_features') and hasattr(backend, 'high_features') \
                and hasattr(backend, 'lastconv_channel'):
            self.backend = backend
        elif 'resnet' in backend:
            self.backend = ResnetBackend(backend, os, pretrained, norm_layer=norm_layer)
        elif 'resnest' in backend:
            self.backend = ResNeSt(backbone=backend, norm_layer=norm_layer, pretrained=pretrained)
            self.backend.lastconv_channel = 2048
        else:
            raise NotImplementedError

        self.aspp_out_channel = 256

        if hasattr(self.backend, 'interconv_channel'):
            self.aspp_out_channel = self.backend.interconv_channel

        self.aspp_pooling = ASPPBlock(self.backend.lastconv_channel, self.aspp_out_channel, os, norm_layer=norm_layer)

        self.cbr_low = nn.Sequential(nn.Conv2d(256, 48,
                                               kernel_size=1, bias=False),
                                     norm_layer(48),
                                     nn.ReLU(inplace=True))
        self.cbr_last = nn.Sequential(nn.Conv2d(self.aspp_out_channel + 48,
                                                self.aspp_out_channel, kernel_size=3, padding=1, bias=False),
                                      norm_layer(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.aspp_out_channel,
                                                kernel_size=3, padding=1, bias=False),
                                      norm_layer(self.aspp_out_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.aspp_out_channel, self.num_classes, kernel_size=1))

    def forward(self, x):
        h, w = x.size()[2:]
        low_features, x = self.backend(x)
        x = self.aspp_pooling(x)
        x = F.interpolate(x, size=low_features.size()[2:], mode='bilinear', align_corners=True)
        low_features = self.cbr_low(low_features)

        x = torch.cat([x, low_features], dim=1)
        x = self.cbr_last(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

class ResnetBackend(nn.Module):
    def __init__(self, backend='resnet18', os=16, pretrained=False, norm_layer=nn.BatchNorm2d):
        '''
        :param backend: resnet<> or se_resnet
        '''
        super(ResnetBackend, self).__init__()

        _backend_model = resnet101(pretrained=pretrained, output_stride=os, norm_layer=norm_layer)

        self.low_features = nn.Sequential(_backend_model.conv1,
                                          _backend_model.bn1,
                                          _backend_model.relu,
                                          _backend_model.maxpool,
                                          _backend_model.layer1
                                          )

        self.high_features = nn.Sequential(_backend_model.layer2,
                                           _backend_model.layer3,
                                           _backend_model.layer4
                                           )

        self.lastconv_channel = 2048

    def forward(self, x):
        low_features = self.low_features(x)
        x = self.high_features(low_features)
        return low_features, x

##########################################################################################
Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]

# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]

def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])

def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read()  # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset]  # string of '0's and '1's
    return code_str

def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32': bitstr2float, 'int32': bitstr2int}
    idx = 0

    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1':  # Leaf node
            value = converter[dtype](code_str[idx:idx + 32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()

def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory / f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory / f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        if ptr.left is not None and bit == '0':
            ptr = ptr.left
        elif ptr.right is not None:
            ptr = ptr.right

        if ptr.value is not None:  # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=dtype)

def huffman_decode_model(model, directory):
    # for name, param in model.named_parameters():
    new_state = model.state_dict()
    for name, param in new_state.items():
        # print(name)
        if 'weight' in name:
            dev = param.device
            weight = param.data.cpu().numpy()
            tar_shape = weight.shape
            if len(weight.shape) != 2:
                length = len(weight.flatten())
                num = np.arange(int(np.sqrt(length)), length + 1)
                mask = length % num == 0
                num = num[mask]
                weight = weight.reshape(-1, num[0])
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            matrix = csr_matrix if shape[0] < shape[1] else csc_matrix

            # Decode data
            data = huffman_decode(directory, name + f'_{form}_data', dtype='float32')
            indices = huffman_decode(directory, name + f'_{form}_indices', dtype='int32')
            indptr = reconstruct_indptr(huffman_decode(directory, name + f'_{form}_indptr', dtype='int32'))

            # Construct matrix
            mat = matrix((data, indices, indptr), shape)

            # Insert to model
            new_weight = torch.from_numpy(mat.toarray()).to(dev).reshape(tar_shape)
            param.data = new_weight
        else:
            dev = param.device
            bias = np.load(os.path.join(directory, name), allow_pickle=True)
            param.data = torch.from_numpy(bias).to(dev)
    return new_state

BatchNorm2d_class = BatchNorm2d = nn.BatchNorm2d

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

def conv3x3_3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context

class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3, self).__init__()
        self.conv1 = conv3x3_3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, )))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock3,
    'BOTTLENECK': Bottleneck3
}

class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = True

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        ocr_mid_channels = 512
        ocr_key_channels = 256

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(config['DATASET']['NUM_CLASSES'])

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, config['DATASET']['NUM_CLASSES'], kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, config['DATASET']['NUM_CLASSES'],
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        imsize = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1)

        out_aux_seg = []

        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return F.interpolate(out, size=imsize, mode="bilinear", align_corners=ALIGN_CORNERS)

    def init_weights(self, pretrained='', ):
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                               pretrained_dict.items()}
            # print(set(model_dict) - set(pretrained_dict))
            # print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}

            pretrained_dict.pop("cls_head.weight")
            pretrained_dict.pop("cls_head.bias")
            pretrained_dict.pop("aux_head.3.weight")
            pretrained_dict.pop("aux_head.3.bias")
            # print(pretrained_dict)

            # for k, _ in pretrained_dict.items():
            # logger.info(
            #     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

config = {
    "DATASET": {
        "NUM_CLASSES": 15
    },
    "MODEL": {
        "NAME": "seg_hrnet_ocr",
        "NUM_OUTPUTS": 2,
        "PRETRAINED": "./hrnet_ocr_cs_trainval_8227_torch11.pth",
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_RANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [
                    4
                ],
                "NUM_CHANNELS": [
                    64
                ],
                "FUSE_METHOD": "SUM"
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [
                    4,
                    4
                ],
                "NUM_CHANNELS": [
                    48,
                    96
                ],
                "FUSE_METHOD": "SUM"
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [
                    4,
                    4,
                    4
                ],
                "NUM_CHANNELS": [
                    48,
                    96,
                    192
                ],
                "FUSE_METHOD": "SUM"
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [
                    4,
                    4,
                    4,
                    4
                ],
                "NUM_CHANNELS": [
                    48,
                    96,
                    192,
                    384
                ],
                "FUSE_METHOD": "SUM"
            }
        }
    }
}

def get_seg_model(cfg=config, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)

    return model

##########################################################################################

def init_model():
    model = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path1 = os.path.join(os.path.dirname(__file__), 'deeplab_resnet')
    model1 = DeepLabv3_plus(in_channels=3, num_classes=15, os=16, pretrained=False, norm_layer=nn.BatchNorm2d,
                            backend="resnet101")
    model1.load_state_dict(huffman_decode_model(model1, path1))
    model1 = model1.to(device)
    model1.eval()
    model.append(model1)

    path2 = os.path.join(os.path.dirname(__file__), 'danet')
    model2 = DANet(backbone='resnet101', nclass=15, pretrained=False, norm_layer=nn.BatchNorm2d)
    model2.load_state_dict(huffman_decode_model(model2, path2))
    model2 = model2.to(device)
    model2.eval()
    model.append(model2)

    path3 = os.path.join(os.path.dirname(__file__), 'deeplab_resnest')
    model3 = DeepLabv3_plus(in_channels=3, num_classes=15, os=16, pretrained=False, norm_layer=nn.BatchNorm2d,
                            backend="resnest101")
    model3.load_state_dict(huffman_decode_model(model3, path3))
    model3 = model3.to(device)
    model3.eval()
    model.append(model3)

    path4 = os.path.join(os.path.dirname(__file__), 'model_hrnet.pth')
    model4 = get_seg_model()
    model4.load_state_dict(torch.load(path4, map_location='cpu'))
    model4 = model4.to(device)
    model4.eval()
    model.append(model4)

    return model
