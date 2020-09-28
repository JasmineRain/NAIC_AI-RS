import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils_Deeplab import SyncBN2d


__all__ = ["scSEUNet"]


class DoubleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=SyncBN2d):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=SyncBN2d, scale_factor=2):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


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


class scSEUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=8, pretrained=False, norm_layer=SyncBN2d):
        super(scSEUNet, self).__init__()

        self.encoder = models.resnet34(pretrained=False, norm_layer=norm_layer)
        if pretrained:
            self.encoder.load_state_dict(torch.load("./resnet34-333f7ec4.pth"))

        self.GapCov = DoubleConvBlock(ch_in=1024, ch_out=512, norm_layer=norm_layer)
        self.Up4 = UpBlock(ch_in=512, ch_out=256, scale_factor=2, norm_layer=norm_layer)
        self.UpConv4 = DoubleConvBlock(ch_in=512, ch_out=256, norm_layer=norm_layer)

        self.Up3 = UpBlock(ch_in=256, ch_out=128, scale_factor=2, norm_layer=norm_layer)
        self.UpConv3 = DoubleConvBlock(ch_in=256, ch_out=128, norm_layer=norm_layer)

        self.Up2 = UpBlock(ch_in=128, ch_out=64, scale_factor=2, norm_layer=norm_layer)
        self.UpConv2 = DoubleConvBlock(ch_in=128, ch_out=64, norm_layer=norm_layer)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.SCSE4 = ChannelSpatialSELayer(num_channels=512, reduction_ratio=2)
        self.SCSE3 = ChannelSpatialSELayer(num_channels=256, reduction_ratio=2)
        self.SCSE2 = ChannelSpatialSELayer(num_channels=128, reduction_ratio=2)
        self.SCSE1 = ChannelSpatialSELayer(num_channels=64, reduction_ratio=2)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        c1 = self.encoder.layer1(x)
        # print(c1.shape)
        c2 = self.encoder.layer2(c1)
        # print(c2.shape)
        c3 = self.encoder.layer3(c2)
        # print(c3.shape)
        c4 = self.encoder.layer4(c3)
        # print(c4.shape)

        G = F.interpolate(F.adaptive_avg_pool2d(c4, output_size=(1, 1)), mode='bilinear', align_corners=True, scale_factor=8)
        c4 = self.SCSE4(c4)
        x = self.GapCov(torch.cat((G, c4), dim=1))

        x = self.Up4(x)
        c3 = self.SCSE3(c3)
        x = self.UpConv4(torch.cat((x, c3), dim=1))

        x = self.Up3(x)
        c2 = self.SCSE2(c2)
        x = self.UpConv3(torch.cat((x, c2), dim=1))

        x = self.Up2(x)
        c1 = self.SCSE1(c1)
        x = self.UpConv2(torch.cat((x, c1), dim=1))

        x = self.Conv_1x1(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x


if __name__ == "__main__":
    net = scSEUNet(img_ch=3, output_ch=8, pretrained=False)
    img = torch.rand(1, 3, 256, 256)
    res = net(img)
    print(res.shape)