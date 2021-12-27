import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def relu():
    return nn.ReLU()


class BasicConv(nn.Layer):
    def __init__(self, in_ch, out_ch, kernel, pad_mode='constant', bn=False, act=False, **kwargs):
        super().__init__()
        seq = []
        if kernel >= 2:
            seq.append(nn.Pad2D(kernel//2, mode=pad_mode))
        seq.append(
            nn.Conv2D(
                in_ch, out_ch, kernel,
                stride=1, padding=0,
                bias_attr=False if bn else None,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm2D(out_ch))
        if act:
            seq.append(relu())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='constant', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, bn=bn, act=act, **kwargs)


class Conv7x7(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='constant', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 7, pad_mode, bn, act, **kwargs)


class MaxPool2x2(nn.MaxPool2D):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


class MaxUnPool2x2(nn.MaxUnPool2D):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


class ConvTransposed3x3(nn.Layer):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **kwargs):
        super().__init__()
        seq = []
        seq.append(
            nn.Conv2DTranspose(
                in_ch, out_ch, 3,
                stride=2, padding=1,
                bias_attr=False if bn else None,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm2D(out_ch))
        if act:
            seq.append(relu())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)
        