# Implementation of
# C. Zhang et al., “A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images,” 2020, doi: 10.1016/J.ISPRSJPRS.2020.06.003.

# Adapted from https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/blob/master/pytorch%20version/DSIFN.py

## Original head information
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import vgg16

from ._blocks import BasicConv, Conv3x3, Conv7x7


class VGG16FeaturePicker(nn.Layer):
    def __init__(self, indices=(3,8,15,22,29)):
        super().__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.LayerList(features)
        self.indices = set(indices)

    def forward(self,x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


class ChannelAttention(nn.Layer):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = BasicConv(in_ch, in_ch//ratio, 1, bias=False, act=nn.ReLU())
        self.fc2 = BasicConv(in_ch//ratio, in_ch, 1, bias=False)

    def forward(self,x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = Conv7x7(2, 1, bias=False)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return F.sigmoid(x)


def conv2d_bn(in_ch, out_ch):
    return nn.Sequential(
        Conv3x3(in_ch, out_ch, norm=nn.BatchNorm2D(out_ch), act=nn.PReLU()),
        nn.Dropout(p=0.6),
    )


class DSIFN(nn.Layer):
    def __init__(self):
        super().__init__()

        self.encoder1 = VGG16FeaturePicker()
        self.encoder2 = VGG16FeaturePicker()

        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = nn.BatchNorm2D(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = nn.BatchNorm2D(512)
        self.o1_conv3 = nn.Conv2D(512, 1, 1)
        self.trans_conv1 = nn.Conv2DTranspose(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = nn.BatchNorm2D(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = nn.BatchNorm2D(256)
        self.o2_conv4 = nn.Conv2D(256, 1, 1)
        self.trans_conv2 = nn.Conv2DTranspose(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = nn.BatchNorm2D(128)
        self.o3_conv4 = nn.Conv2D(128, 1, 1)
        self.trans_conv3 = nn.Conv2DTranspose(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = nn.BatchNorm2D(64)
        self.o4_conv4 = nn.Conv2D(64, 1, 1)
        self.trans_conv4 = nn.Conv2DTranspose(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = nn.BatchNorm2D(16)
        self.o5_conv4 = nn.Conv2D(16, 1, 1)

    def forward(self, t1, t2):
        # Extract bi-temporal features
        with paddle.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29,= t2_feats

        # Multi-level decoding
        x = paddle.concat([t1_f_l29, t2_f_l29], axis=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        out1 = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = paddle.concat([x, t1_f_l22, t2_f_l22], axis=1)
        x = self.ca2(x)*x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        out2 = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = paddle.concat([x, t1_f_l15, t2_f_l15], axis=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        out3 = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = paddle.concat([x, t1_f_l8, t2_f_l8], axis=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        out4 = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = paddle.concat([x, t1_f_l3, t2_f_l3], axis=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        out5 = self.o5_conv4(x)

        return out5, out4, out3, out2, out1