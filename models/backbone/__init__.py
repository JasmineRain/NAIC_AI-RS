# -*- coding: utf-8 -*-
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .senet import se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d, senet154
from .mobilenet import mobilenet_v2
from .shufflenet import shufflenet_v2
from .xception import AlignedXception
from .resnest import ResNeSt

_all_resnet_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                         'resnet152', 'senet154', 'se_resnet50', 'se_resnet101',
                         'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']

_all_mobilenet_backbones = ['mobilenet_v2']

_all_shufflenet_backbones = ['shufflenet_v2']

_all_xception_backbones = ['AlignedXception']

_all_resnest_backbones = ['resnest50', 'resnest101']