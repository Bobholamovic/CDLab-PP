# Implementation of 
# H. Chen and Z. Shi, “A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection,” Remote Sensing, vol. 12, no. 10, p. 1662, 2020, doi: 10.3390/rs12101662.

# Refer to https://github.com/justchenhao/STANet
# The resnet implementation differs from the original work. 
# The most notable difference is that multiple dilation rates are not used in layer4.

# TODO: re-implement resnet backbone in this repo.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .backbones import resnet
from ._blocks import Conv1x1, Conv3x3
from ._utils import KaimingInitMixin, Identity


def build_feat_extractor(in_ch, width):
    return nn.Sequential(
        Backbone(in_ch, 'resnet18'),
        Decoder(width)
    )


def build_sta_module(in_ch, att_type, ds):
    if att_type == 'BAM':
        return Attention(BAM(in_ch, ds))
    elif att_type == 'PAM':
        return Attention(PAM(in_ch, ds))
    else:
        raise ValueError


class Backbone(nn.Layer, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2,1,2,2,2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides)
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides)
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides)
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2D(
                in_ch, 
                self.resnet.inplanes,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias_attr=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class Decoder(nn.Layer, KaimingInitMixin):
    def __init__(self, f_ch):
        super().__init__()
        self.dr1 = Conv1x1(64, 96, norm=nn.BatchNorm2D(96), act=nn.ReLU())
        self.dr2 = Conv1x1(128, 96, norm=nn.BatchNorm2D(96), act=nn.ReLU())
        self.dr3 = Conv1x1(256, 96, norm=nn.BatchNorm2D(96), act=nn.ReLU())
        self.dr4 = Conv1x1(512, 96, norm=nn.BatchNorm2D(96), act=nn.ReLU())
        self.conv_out = nn.Sequential(
            Conv3x3(384, 256, norm=nn.BatchNorm2D(256), act=nn.ReLU()),
            nn.Dropout(0.5),
            Conv1x1(256, f_ch, norm=nn.BatchNorm2D(f_ch), act=nn.ReLU())
        )

        self._init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])
        f2 = self.dr2(feats[1])
        f3 = self.dr3(feats[2])
        f4 = self.dr4(feats[3])

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        x = paddle.concat([f1, f2, f3, f4], axis=1)
        y = self.conv_out(x)

        return y


class BAM(nn.Layer):
    def __init__(self, in_ch, ds):
        super().__init__()

        self.ds = ds
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2D(self.ds)

        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch)
        self.conv_k = Conv1x1(in_ch, self.key_ch)
        self.conv_v = Conv1x1(in_ch, self.val_ch)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        x = x.flatten(-2)
        x_rs = self.pool(x)
        
        b, c, h, w = x_rs.shape
        query = self.conv_q(x_rs).reshape((b,-1,h*w)).transpose((0,2,1))
        key = self.conv_k(x_rs).reshape((b,-1,h*w))
        energy = paddle.bmm(query, key)
        energy = (self.key_ch**(-0.5)) * energy

        attention = self.softmax(energy)

        value = self.conv_v(x_rs).reshape((b,-1,w*h))

        out = paddle.bmm(value, attention.transpose((0,2,1)))
        out = out.reshape((b,c,h,w))

        out = F.interpolate(out, scale_factor=self.ds)
        out = out + x
        return out.reshape(out.shape[:-1]+[out.shape[-1]//2, 2])


class PAMBlock(nn.Layer):
    def __init__(self, in_ch, scale=1, ds=1):
        super().__init__()

        self.scale = scale
        self.ds = ds
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2D(self.ds)

        self.val_ch = in_ch
        self.key_ch = in_ch // 8
        self.conv_q = Conv1x1(in_ch, self.key_ch, norm=nn.BatchNorm2D(self.key_ch))
        self.conv_k = Conv1x1(in_ch, self.key_ch, norm=nn.BatchNorm2D(self.key_ch))
        self.conv_v = Conv1x1(in_ch, self.val_ch)

    def forward(self, x):
        x_rs = self.pool(x)

        # Get query, key, and value
        query = self.conv_q(x_rs)
        key = self.conv_k(x_rs)
        value = self.conv_v(x_rs)
        
        # Split the whole image into subregions
        b, c, h, w = x_rs.shape
        query = self._split_subregions(query)
        key = self._split_subregions(key)
        value = self._split_subregions(value)
        
        # Perform subregion-wise attention
        out = self._attend(query, key, value)

        # Stack subregions to reconstruct the whole image
        out = self._recons_whole(out, b, c, h, w)
        out = F.interpolate(out, scale_factor=self.ds)
        return out

    def _attend(self, query, key, value):
        energy = paddle.bmm(query.transpose((0,2,1)), key)  # batch matrix multiplication
        energy = (self.key_ch**(-0.5)) * energy
        attention = F.softmax(energy, axis=-1)
        out = paddle.bmm(value, attention.transpose((0,2,1)))
        return out

    def _split_subregions(self, x):
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0
        x = x.reshape((b, c, self.scale, h//self.scale, self.scale, w//self.scale))
        x = x.transpose((0,2,4,1,3,5)).reshape((b*self.scale*self.scale, c, -1))
        return x

    def _recons_whole(self, x, b, c, h, w):
        x = x.reshape((b, self.scale, self.scale, c, h//self.scale, w//self.scale))
        x = x.transpose((0,3,1,4,2,5)).reshape((b, c, h, w))
        return x


class PAM(nn.Layer):
    def __init__(self, in_ch, ds, scales=(1,2,4,8)):
        super().__init__()

        self.stages = nn.LayerList([
            PAMBlock(in_ch, scale=s, ds=ds)
            for s in scales
        ])
        self.conv_out = Conv1x1(in_ch*len(scales), in_ch, bias=False)

    def forward(self, x):
        x = x.flatten(-2)
        res = [stage(x) for stage in self.stages]
        out = self.conv_out(paddle.concat(res, axis=1))
        return out.reshape(out.shape[:-1]+[out.shape[-1]//2, 2])


class Attention(nn.Layer):
    def __init__(self, att):
        super().__init__()
        self.att = att

    def forward(self, x1, x2):
        x = paddle.stack([x1, x2], axis=-1)
        y = self.att(x)
        return y[...,0], y[...,1]


class STANet(nn.Layer):
    def __init__(self, in_ch, width=64, att_type='BAM', ds=1):
        super().__init__()
        self.extract = build_feat_extractor(in_ch=in_ch, width=width)
        self.attend = build_sta_module(in_ch=width, att_type=att_type, ds=ds)
        self.calc_dist = nn.PairwiseDistance(keepdim=True)

    def forward(self, t1, t2):
        f1 = self.extract(t1)
        f2 = self.extract(t2)
        f1, f2 = self.attend(f1, f2)
        dist = self.calc_dist(f1, f2)
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)
        return dist