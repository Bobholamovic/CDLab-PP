import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CompactnessLoss(nn.MSELoss):
    def forward(self, posclr_s, posclr):
        return super().forward(posclr_s[:,:2], posclr[:,:2].detach())