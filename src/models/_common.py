import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._blocks import Conv1x1, BasicConv


class ChannelAttention(nn.Layer):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = Conv1x1(in_ch, in_ch//ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch//ratio, in_ch, bias=False)

    def forward(self,x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return F.sigmoid(x)


class CBAM(nn.Layer):
    def __init__(self, in_ch, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y
