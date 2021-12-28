import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BasicConv(nn.Layer):
    def __init__(
        self, in_ch, out_ch, 
        kernel, pad_mode='constant', 
        bias='auto', norm=None, act=None, 
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel >= 2:
            seq.append(nn.Pad2D(kernel//2, mode=pad_mode))
        seq.append(
            nn.Conv2D(
                in_ch, out_ch, kernel,
                stride=1, padding=0,
                bias_attr=(False if norm else None) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            seq.append(norm)
        if act:
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto', norm=None, act=None, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, norm=norm, act=act, **kwargs)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto', norm=None, act=None, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, norm=norm, act=act, **kwargs)


class Conv7x7(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='constant', bias='auto', norm=None, act=None, **kwargs):
        super().__init__(in_ch, out_ch, 7, pad_mode=pad_mode, norm=norm, act=act, **kwargs)


class MaxPool2x2(nn.MaxPool2D):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


class MaxUnPool2x2(nn.MaxUnPool2D):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


class ConvTransposed3x3(nn.Layer):
    def __init__(
        self, in_ch, out_ch,
        bias='auto', norm=None, act=None, 
        **kwargs
    ):
        super().__init__()
        seq = []
        seq.append(
            nn.Conv2DTranspose(
                in_ch, out_ch, 3,
                stride=2, padding=1,
                bias_attr=(False if norm else None) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            seq.append(norm)
        if act:
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)
        