import torch
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from torchsummary import summary
from utils_Deeplab import SyncBN2d
from resnest.torch import resnest101, resnest50


class ResNeSt(nn.Module):
    def __init__(self, backbone="resnest101", norm_layer=nn.BatchNorm2d, pretrained=True):
        super(ResNeSt, self).__init__()
        if backbone == 'resnest50':
            self.pretrained = resnest50(pretrained=False, dilated=True, norm_layer=norm_layer)
            if pretrained:
                self.pretrained.load_state_dict(torch.load("./resnest50-528c19ca.pth"))
        elif backbone == 'resnest101':
            self.pretrained = resnest101(pretrained=False, dilated=True, norm_layer=norm_layer)
            if pretrained:
                self.pretrained.load_state_dict(torch.load("./resnest101-22405ba7.pth"))

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