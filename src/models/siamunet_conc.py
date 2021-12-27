# Implementation of
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
# with modifications (remove dropout layers, add residual connections, double number of channels, and change decoding blocks).

# Transferred from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/siamunet_conc.py

## Original head information
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._blocks import Conv3x3, MaxPool2x2, ConvTransposed3x3


class SiamUnet_conc(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv11 = Conv3x3(in_ch, 16, bn=True, act=True)
        self.conv12 = Conv3x3(16, 16, bn=True, act=True)
        self.pool1 = MaxPool2x2()

        self.conv21 = Conv3x3(16, 32, bn=True, act=True)
        self.conv22 = Conv3x3(32, 32, bn=True, act=True)
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(32, 64, bn=True, act=True)
        self.conv32 = Conv3x3(64, 64, bn=True, act=True)
        self.conv33 = Conv3x3(64, 64, bn=True, act=True)
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(64, 128, bn=True, act=True)
        self.conv42 = Conv3x3(128, 128, bn=True, act=True)
        self.conv43 = Conv3x3(128, 128, bn=True, act=True)
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(128, 128, output_padding=1)

        self.conv43d = Conv3x3(384, 128, bn=True, act=True)
        self.conv42d = Conv3x3(128, 128, bn=True, act=True)
        self.conv41d = Conv3x3(128, 64, bn=True, act=True)

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(192, 64, bn=True, act=True)
        self.conv32d = Conv3x3(64, 64, bn=True, act=True)
        self.conv31d = Conv3x3(64, 32, bn=True, act=True)

        self.upconv2 = ConvTransposed3x3(32, 32, output_padding=1)

        self.conv22d = Conv3x3(96, 32, bn=True, act=True)
        self.conv21d = Conv3x3(32, 16, bn=True, act=True)

        self.upconv1 = ConvTransposed3x3(16, 16, output_padding=1)

        self.conv12d = Conv3x3(48, 16, bn=True, act=True)
        self.conv11d = Conv3x3(16, out_ch)

    def forward(self, t1, t2):
        # Encode 1
        # Stage 1
        x11 = self.conv11(t1)
        x12_1 = self.conv12(x11)
        x1p = self.pool1(x12_1)

        # Stage 2
        x21 = self.conv21(x1p)
        x22_1 = self.conv22(x21)
        x2p = self.pool2(x22_1)

        # Stage 3
        x31 = self.conv31(x2p)
        x32 = self.conv32(x31)
        x33_1 = self.conv33(x32)
        x3p = self.pool3(x33_1)

        # Stage 4
        x41 = self.conv41(x3p)
        x42 = self.conv42(x41)
        x43_1 = self.conv43(x42)
        x4p = self.pool4(x43_1)

        # Encode 2
        # Stage 1
        x11 = self.conv11(t2)
        x12_2 = self.conv12(x11)
        x1p = self.pool1(x12_2)

        # Stage 2
        x21 = self.conv21(x1p)
        x22_2 = self.conv22(x21)
        x2p = self.pool2(x22_2)

        # Stage 3
        x31 = self.conv31(x2p)
        x32 = self.conv32(x31)
        x33_2 = self.conv33(x32)
        x3p = self.pool3(x33_2)

        # Stage 4
        x41 = self.conv41(x3p)
        x42 = self.conv42(x41)
        x43_2 = self.conv43(x42)
        x4p = self.pool4(x43_2)
        
        # Decode
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, x43_1.shape[3]-x4d.shape[3], 0, x43_1.shape[2]-x4d.shape[2])
        x4d = paddle.concat([F.pad(x4d, pad=pad4, mode='replicate'), x43_1, x43_2], 1)
        x43d = self.conv43d(x4d)
        x42d = self.conv42d(x43d)
        x41d = self.conv41d(x42d)

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, x33_1.shape[3]-x3d.shape[3], 0, x33_1.shape[2]-x3d.shape[2])
        x3d = paddle.concat([F.pad(x3d, pad=pad3, mode='replicate'), x33_1, x33_2], 1)
        x33d = self.conv33d(x3d)
        x32d = self.conv32d(x33d)
        x31d = self.conv31d(x32d)

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, x22_1.shape[3]-x2d.shape[3], 0, x22_1.shape[2]-x2d.shape[2])
        x2d = paddle.concat([F.pad(x2d, pad=pad2, mode='replicate'), x22_1, x22_2], 1)
        x22d = self.conv22d(x2d)
        x21d = self.conv21d(x22d)

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, x12_1.shape[3]-x1d.shape[3], 0, x12_1.shape[2]-x1d.shape[2])
        x1d = paddle.concat([F.pad(x1d, pad=pad1, mode='replicate'), x12_1, x12_2], 1)
        x12d = self.conv12d(x1d)
        x11d = self.conv11d(x12d)

        return x11d