import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RendPoint import sampling_points_v2, sampling_features
from torchvision import models
from torch.nn import Module, Conv2d, Parameter, Softmax

__all__ = ['RendUNet']


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

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class RefineUnit(nn.Module):

    def __init__(self, in_channel, out_channel, residual=False):
        super(RefineUnit, self).__init__()
        self.residual = residual
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):

        if self.residual:
            residual = x
            x = self.conv(x)
            return x + residual
        else:
            return self.conv(x)


class MLP(nn.Module):
    def __init__(self, fc_dim_in, fc_dims, num_classes):
        super(MLP, self).__init__()
        self.fc_dim_in = fc_dim_in
        self.fc_dims = fc_dims
        self.fc_nums = len(self.fc_dims)
        self.num_classes = num_classes
        self.mlp = nn.Sequential()
        for i in range(self.fc_nums):
            self.fc_dim = self.fc_dims[i]
            self.mlp.add_module("fc{}".format(i + 1),
                                nn.Conv1d(in_channels=self.fc_dim_in, out_channels=self.fc_dim, kernel_size=1, stride=1,
                                          padding=0, bias=True))
            self.mlp.add_module("relu{}".format(i + 1),
                                nn.ReLU(inplace=True))
            self.fc_dim_in = self.fc_dim
        self.mlp.add_module("final_fc",
                            nn.Conv1d(in_channels=self.fc_dim_in, out_channels=self.num_classes, kernel_size=1,
                                      stride=1, padding=0))

    def forward(self, x):
        out = self.mlp(x)
        return out


class SegNet(nn.Module):

    def __init__(self, pretrained=False, n_class=8):
        super(SegNet, self).__init__()

        pretrained = models.resnet50(pretrained=pretrained, progress=True, replace_stride_with_dilation=[0, 1, 1])
        pretrained.load_state_dict(torch.load("./resnet50-19c8e357.pth"))

        self.refine = RefineUnit(in_channel=3, out_channel=64, residual=False)

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

        self.att = ChannelSpatialSELayer(num_channels=2048, reduction_ratio=4)

        self.seg = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(2048, n_class, 1))

    def forward(self, x):
        # encoder
        refine = self.refine(x)
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

        coarse = self.seg(self.att(x4))

        return refine, x0, x1, x2, x3, coarse


class RendNet(nn.Module):
    def __init__(self, n_class):
        super(RendNet, self).__init__()
        self.mlp3 = MLP(fc_dim_in=1032, fc_dims=[512, 512, 512], num_classes=n_class)
        self.mlp2 = MLP(fc_dim_in=520, fc_dims=[512, 512, 512], num_classes=n_class)
        self.mlp1 = MLP(fc_dim_in=264, fc_dims=[512, 512, 512], num_classes=n_class)
        self.mlp0 = MLP(fc_dim_in=72, fc_dims=[512, 512, 512], num_classes=n_class)
        self.mlp_refine = MLP(fc_dim_in=72, fc_dims=[512, 512, 512], num_classes=n_class)

    def forward(self, refine, x0, x1, x2, x3, coarse):
        if not self.training:
            return self.inference(refine, x0, x1, x2, x3, coarse)

        # coarse size: 48x48
        # rend stage 1 with layer3
        temp1 = coarse
        points1 = sampling_points_v2(torch.softmax(temp1, dim=1), N=512, k=3, beta=0.75)
        coarse_feature = sampling_features(temp1, points1, align_corners=True)
        fine_feature = sampling_features(x3, points1, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend1 = self.mlp3(feature_representation)

        # coarse size: 48x48
        # rend stage 2 with layer2
        temp2 = coarse
        points2 = sampling_points_v2(torch.softmax(temp2, dim=1), N=512, k=3, beta=0.75)
        coarse_feature = sampling_features(temp2, points2, align_corners=True)
        fine_feature = sampling_features(x2, points2, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend2 = self.mlp2(feature_representation)

        # coarse size: 96x96
        # rend stage 3 with layer1
        temp3 = F.interpolate(temp2, scale_factor=2, mode='bilinear', align_corners=True)
        points3 = sampling_points_v2(torch.softmax(temp3, dim=1), N=2048, k=3, beta=0.75)
        coarse_feature = sampling_features(temp3, points3, align_corners=True)
        fine_feature = sampling_features(x1, points3, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend3 = self.mlp1(feature_representation)

        # coarse size: 192x192
        # rend stage 4 with layer0
        temp4 = F.interpolate(temp3, scale_factor=2, mode='bilinear', align_corners=True)
        points4 = sampling_points_v2(torch.softmax(temp4, dim=1), N=2048, k=3, beta=0.75)
        coarse_feature = sampling_features(temp4, points4, align_corners=True)
        fine_feature = sampling_features(x0, points4, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend4 = self.mlp0(feature_representation)

        # coarse size: 384x384
        # rend stage 5 with layer refined
        temp5 = F.interpolate(temp4, scale_factor=2, mode='bilinear', align_corners=True)
        points5 = sampling_points_v2(torch.softmax(temp5, dim=1), N=2048, k=3, beta=0.75)
        coarse_feature = sampling_features(temp5, points5, align_corners=True)
        fine_feature = sampling_features(refine, points5, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend5 = self.mlp_refine(feature_representation)

        return {
            "coarse": coarse,
            "stage1": [points1, rend1],
            "stage2": [points2, rend2],
            "stage3": [points3, rend3],
            "stage4": [points4, rend4],
            "stage5": [points5, rend5],
        }

    @torch.no_grad()
    def inference(self, refine, x0, x1, x2, x3, coarse):
        # stage 1
        # coarse size: 48x48
        temp = coarse
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 512, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=True)
        fine_feature = sampling_features(x3, points, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp3(feature_representation)
        B, C, H, W = coarse.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse1 = (coarse.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # stage 2
        # 48x48
        temp = coarse1
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 512, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=True)
        fine_feature = sampling_features(x2, points, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp2(feature_representation)
        B, C, H, W = coarse1.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse2 = (coarse1.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # stage 3
        # 96x96
        coarse3 = F.interpolate(coarse2, scale_factor=2, mode='bilinear', align_corners=True)
        temp = coarse3
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 512, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=True)
        fine_feature = sampling_features(x1, points, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp1(feature_representation)
        B, C, H, W = coarse3.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse3 = (coarse3.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # stage 4
        # 192x192
        coarse4 = F.interpolate(coarse3, scale_factor=2, mode='bilinear', align_corners=True)
        temp = coarse4
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 512, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=True)
        fine_feature = sampling_features(x0, points, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp0(feature_representation)
        B, C, H, W = coarse4.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse4 = (coarse4.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # stage 5
        # 384x384
        coarse5 = F.interpolate(coarse4, scale_factor=2, mode='bilinear', align_corners=True)
        temp = coarse5
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 512, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=True)
        fine_feature = sampling_features(refine, points, align_corners=True)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp_refine(feature_representation)
        B, C, H, W = coarse5.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse5 = (coarse5.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        return {
            "fine": coarse5
        }


class RendUNet(nn.Module):
    def __init__(self, n_class=8):
        super(RendUNet, self).__init__()
        self.seg = SegNet(pretrained=False, n_class=8)
        self.rend = RendNet(n_class=n_class)

    def forward(self, x):

        refine, x0, x1, x2, x3, coarse = self.seg(x)
        res = self.rend(refine, x0, x1, x2, x3, coarse)
        return res

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = RendUNet()
    img = torch.rand(4, 3, 384, 384)
    model.train()
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    res = model(img)
    for k, v in res.items():
        print(k, v[0].shape)
