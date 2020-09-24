import torch
import torch.nn as nn


__all__ = ["UNet"]


class DoubleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=8):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = DoubleConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = DoubleConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = DoubleConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpBlock(ch_in=1024, ch_out=512)
        self.Up_conv5 = DoubleConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpBlock(ch_in=512, ch_out=256)
        self.Up_conv4 = DoubleConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpBlock(ch_in=256, ch_out=128)
        self.Up_conv3 = DoubleConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpBlock(ch_in=128, ch_out=64)
        self.Up_conv2 = DoubleConvBlock(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
