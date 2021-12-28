import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._blocks import BasicConv, Conv7x7


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