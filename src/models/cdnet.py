# Implementation of 
# Alcantarilla, P. F., Stent, S., Ros, G., Arroyo, R., & Gherardi, R. (2018). Street-view change detection with deconvolutional networks. Autonomous Robots, 42(7), 1301–1322.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._blocks import Conv7x7, MaxPool2x2, MaxUnPool2x2


class CDNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = Conv7x7(in_ch, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.pool1 = MaxPool2x2(return_mask=True)
        self.conv2 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.pool2 = MaxPool2x2(return_mask=True)
        self.conv3 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.pool3 = MaxPool2x2(return_mask=True)
        self.conv4 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.pool4 = MaxPool2x2(return_mask=True)
        self.conv5 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.upool4 = MaxUnPool2x2()
        self.conv6 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.upool3 = MaxUnPool2x2()
        self.conv7 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.upool2 = MaxUnPool2x2()
        self.conv8 = Conv7x7(64, 64, norm=nn.BatchNorm2D(64), act=nn.ReLU())
        self.upool1 = MaxUnPool2x2()
        
        self.conv_out = Conv7x7(64, out_ch, norm=False, act=False)
    
    def forward(self, t1, t2):
        # Concatenation
        x = paddle.concat([t1, t2], axis=1)

        # Contraction
        x, ind1 = self.pool1(self.conv1(x))
        x, ind2 = self.pool2(self.conv2(x))
        x, ind3 = self.pool3(self.conv3(x))
        x, ind4 = self.pool4(self.conv4(x))

        # Expansion
        x = self.conv5(self.upool4(x, ind4))
        x = self.conv6(self.upool3(x, ind3))
        x = self.conv7(self.upool2(x, ind2))
        x = self.conv8(self.upool1(x, ind1))

        # Out
        return self.conv_out(x)
