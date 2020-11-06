import os
import torch.nn.functional as F
from collections import OrderedDict
import torch
import math
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


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
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], norm_layer=norm_layer)
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


def resnet18(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet34-333f7ec4.pth.pth"))
    return model


def resnet34(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet34-333f7ec4.pth.pth"))
    return model


def resnet50(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet50-19c8e357.pth"))
    return model


def resnet101(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet101-5d3b4d8f.pth"))
    return model


def resnet152(pretrained=False, output_stride=None, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], output_stride, **kwargs)
    if pretrained:
        model._load_pretrained_model(torch.load("./resnet152-b121ed2d.pth"))
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
    def __init__(self, in_channels, num_classes, backend='resnet18', os=16, pretrained=False, norm_layer=nn.BatchNorm2d):
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


def init_model():
    path = os.path.join(os.path.dirname(__file__), 'model.pth')
    model = DeepLabv3_plus(in_channels=3, num_classes=15, os=16, pretrained=False, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model
