import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from models.RendPoint import sampling_points_v2, sampling_features

import dilated as resnet
from utils_Deeplab import SyncBN2d


__all__ = ["RendDANet"]


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


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, dilated=True, norm_layer=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        # copying modules from pretrained HRNet+OCR
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pretrained.load_state_dict(torch.load("./resnet101-5d3b4d8f.pth"))

    def base_forward(self, x):

        x = self.pretrained.conv1(x)

        x = self.pretrained.bn1(x)

        x = self.pretrained.relu(x)

        x = self.pretrained.maxpool(x)
        c0 = x
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)


        return c0, c1, c2, c3, c4


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

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class PointHead(nn.Module):
    def __init__(self, in_c=520, num_classes=1, k=3, beta=0.75):
        super(PointHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.k = k
        self.beta = beta

    def forward(self, x, feature, mask):

        if not self.training:
            return self.inference(x, feature, mask)

        num_points = 2048
        points = sampling_points_v2(torch.softmax(mask, dim=1), num_points, self.k, self.beta)
        coarse = sampling_features(mask, points, align_corners=True)
        fine = sampling_features(feature, points, align_corners=True)
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points, "coarse": mask}

    @torch.no_grad()
    def inference(self, x, feature, mask):

        num_points = 512
        while mask.shape[-1] != x.shape[-1]:
            mask = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points_v2(torch.softmax(mask, dim=1), num_points, training=self.training)

            coarse = sampling_features(mask, points, align_corners=True)
            fine = sampling_features(feature, points, align_corners=True)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            #print(rend.min())

            B, C, H, W = mask.shape

            #print(mask.shape)

            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            mask = (mask.reshape(B, C, -1)
                    .scatter_(2, points_idx, rend)
                    .view(B, C, H, W))

            #print(mask.shape)

        return {"fine": mask}


class RendDANet(BaseNet):

    def __init__(self, nclass, backbone, norm_layer=SyncBN2d):
        super(RendDANet, self).__init__(nclass, backbone, norm_layer=norm_layer)
        self.da_head = DANetHead(2048, nclass, norm_layer)
        self.rend_head = PointHead(in_c=520, num_classes=nclass)

    def forward(self, x):
        c0, c1, c2, c3, c4 = self.base_forward(x)

        mask = self.da_head(c4)
        result = self.rend_head(x, c2, mask)

        return result


if __name__ == "__main__":
    net = RendDANet(backbone='resnet101', nclass=8)
    img = torch.rand(1, 3, 384, 384)
    mask = torch.rand(1, 8, 384, 384)
    net.train()
    output = net(img)
    for k, v in output.items():
        print(k, v.shape)
    test = sampling_features(mask, output['points'], mode='nearest').squeeze().transpose(0, 1)
    print(test.shape)
