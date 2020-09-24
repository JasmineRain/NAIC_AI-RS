import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils_Deeplab import SyncBN2d
from torch.nn import Module, Conv2d, Parameter, Softmax

__all__ = ['SEDANet']


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


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):

        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class SegNet(nn.Module):

    def __init__(self, pretrained=False, n_class=8):
        super(SegNet, self).__init__()

        pretrained = models.resnext101_32x8d(pretrained=pretrained, progress=True, replace_stride_with_dilation=[0, 1, 1], norm_layer=SyncBN2d)
        pretrained.load_state_dict(torch.load("./resnext101_32x8d-8ba56ff5.pth"))

        # 64 192
        self.layer0 = nn.Sequential(pretrained.conv1,
                                    pretrained.bn1,
                                    pretrained.relu)

        # 256 192
        self.layer1 = nn.Sequential(pretrained.maxpool,
                                    pretrained.layer1)

        # 512 96
        self.layer2 = pretrained.layer2

        # 1024 48
        self.layer3 = pretrained.layer3

        # 2048 24
        self.layer4 = pretrained.layer4

        self.att1 = ChannelSpatialSELayer(num_channels=2048, reduction_ratio=4)
        self.att2 = DANetHead(2048, 512, SyncBN2d)

        self.seg = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(2048, n_class, 1))

    def forward(self, x):
        # encoder
        x0 = self.layer0(x)
        # print(x0.shape)
        x1 = self.layer1(x0)
        # print(x1.shape)
        x2 = self.layer2(x1)
        # print(x2.shape)
        x3 = self.layer3(x2)
        # print(x3.shape)
        x4 = self.layer4(x3)
        # print(x4.shape)

        coarse = self.seg(self.att2(x4))

        return coarse


class SEDANet(nn.Module):
    def __init__(self):
        super(SEDANet, self).__init__()
        self.seg = SegNet(pretrained=False, n_class=8)

    def forward(self, x):

        coarse = self.seg(x)
        return F.interpolate(coarse, size=x.size()[2:], mode='bilinear', align_corners=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = SEDANet()
    img = torch.rand(1, 3, 256, 256)
    model.train()
    # print('# parameters:', sum(param.numel() for param in model.parameters()))
    res = model(img)
    # for k, v in res.items():
    #     print(k, v.shape)
    # for i in range(len(res)):
    #     print(torch.max((res[i])), torch.min(res[i]))
