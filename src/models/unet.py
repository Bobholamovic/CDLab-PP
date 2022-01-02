# Implementation of
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

# Transferred from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/unet.py

## Original head information
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._blocks import Conv3x3, MaxPool2x2, ConvTransposed3x3
from ._utils import Identity


class Unet(nn.Layer):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv11 = Conv3x3(in_ch, 16, norm=True, act=True)
        self.do11 = self.make_dropout()
        self.conv12 = Conv3x3(16, 16, norm=True, act=True)
        self.do12 = self.make_dropout()
        self.pool1 = MaxPool2x2()

        self.conv21 = Conv3x3(16, 32, norm=True, act=True)
        self.do21 = self.make_dropout()
        self.conv22 = Conv3x3(32, 32, norm=True, act=True)
        self.do22 = self.make_dropout()
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(32, 64, norm=True, act=True)
        self.do31 = self.make_dropout()
        self.conv32 = Conv3x3(64, 64, norm=True, act=True)
        self.do32 = self.make_dropout()
        self.conv33 = Conv3x3(64, 64, norm=True, act=True)
        self.do33 = self.make_dropout()
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(64, 128, norm=True, act=True)
        self.do41 = self.make_dropout()
        self.conv42 = Conv3x3(128, 128, norm=True, act=True)
        self.do42 = self.make_dropout()
        self.conv43 = Conv3x3(128, 128, norm=True, act=True)
        self.do43 = self.make_dropout()
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(128, 128, output_padding=1)

        self.conv43d = Conv3x3(256, 128, norm=True, act=True)
        self.do43d = self.make_dropout()
        self.conv42d = Conv3x3(128, 128, norm=True, act=True)
        self.do42d = self.make_dropout()
        self.conv41d = Conv3x3(128, 64, norm=True, act=True)
        self.do41d = self.make_dropout()

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(128, 64, norm=True, act=True)
        self.do33d = self.make_dropout()
        self.conv32d = Conv3x3(64, 64, norm=True, act=True)
        self.do32d = self.make_dropout()
        self.conv31d = Conv3x3(64, 32, norm=True, act=True)
        self.do31d = self.make_dropout()

        self.upconv2 = ConvTransposed3x3(32, 32, output_padding=1)

        self.conv22d = Conv3x3(64, 32, norm=True, act=True)
        self.do22d = self.make_dropout()
        self.conv21d = Conv3x3(32, 16, norm=True, act=True)
        self.do21d = self.make_dropout()

        self.upconv1 = ConvTransposed3x3(16, 16, output_padding=1)

        self.conv12d = Conv3x3(32, 16, norm=True, act=True)
        self.do12d = self.make_dropout()
        self.conv11d = Conv3x3(16, out_ch)

    def forward(self, t1, t2):
        x = paddle.concat([t1, t2], axis=1)

        # Stage 1
        x11 = self.do11(self.conv11(x))
        x12 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43)

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, x43.shape[3]-x4d.shape[3], 0, x43.shape[2]-x4d.shape[2])
        x4d = paddle.concat([F.pad(x4d, pad=pad4, mode='replicate'), x43], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, x33.shape[3]-x3d.shape[3], 0, x33.shape[2]-x3d.shape[2])
        x3d = paddle.concat([F.pad(x3d, pad=pad3, mode='replicate'), x33], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, x22.shape[3]-x2d.shape[3], 0, x22.shape[2]-x2d.shape[2])
        x2d = paddle.concat([F.pad(x2d, pad=pad2, mode='replicate'), x22], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, x12.shape[3]-x1d.shape[3], 0, x12.shape[2]-x1d.shape[2])
        x1d = paddle.concat([F.pad(x1d, pad=pad1, mode='replicate'), x12], 1)
        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)

        return x11d

    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2D(p=0.2)
        else:
            return Identity()