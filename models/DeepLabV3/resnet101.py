import sys
import torch
from collections import OrderedDict
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, bn_momentum=0.0003):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    
    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
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

        out = out + residual      
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1]!=1 else 2, dilation=dilation[1], bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2]!=1 else 2, dilation=dilation[2], bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3]!=1 else 2, dilation=dilation[3], bn_momentum=bn_momentum)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine = True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x, start_module=1, end_module=5):
        if start_module <= 1:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            start_module = 2
        features = []
        for i in range(start_module, end_module+1):
            x = eval('self.layer%d'%(i-1))(x)
            features.append(x)

        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x


def get_resnet101(dilation=[1,1,1,1], bn_momentum=0.0003, is_fpn=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    return model



if __name__ == '__main__':
    net = get_resnet101()
    x = torch.randn(4,3,128,128)
    print(net(x).shape)
