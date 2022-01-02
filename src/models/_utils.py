# Refer to https://github.com/3SPP/PdRSCD/blob/3d492b86ffb5db7d2812a1be3490dd4245502767/ppcd/models/layers/initialize.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def kaiming_normal_init(param, *args, **kwargs):
    return nn.initializer.KaimingNormal(*args, **kwargs)(param)


def constant_init(param, *args, **kwargs):
    return nn.initializer.Constant(*args, **kwargs)(param)


class KaimingInitMixin:
    def _init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                kaiming_normal_init(layer.weight)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x