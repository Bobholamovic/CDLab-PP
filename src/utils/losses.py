import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MixedLoss(nn.Layer):
    def __init__(self, critns, coeffs=1.0):
        super().__init__()
        self.critns = critns
        if isinstance(coeffs, float):
            coeffs = [coeffs]*len(critns)
        if len(coeffs) != len(critns):
            raise ValueError
        self.coeffs = coeffs

    def forward(self, pred, tar):
        loss = 0.0
        for critn, coeff in zip(self.critns, self.coeffs):
            loss += coeff * critn(pred, tar)
        return loss


class CombinedLoss(nn.Layer):
    def __init__(self, critn, coeffs=1.0):
        super().__init__()
        self.critn = critn
        self.coeffs = coeffs

    def forward(self, preds, tar):
        if isinstance(self.coeffs, float):
            coeffs = [self.coeffs]*len(preds)
        else:
            coeffs = self.coeffs
        if len(coeffs) != len(preds):
            raise ValueError
        loss = 0.0
        for coeff, pred in zip(coeffs, preds):
            loss += coeff * self.critn(pred, tar)
        return loss


class DiceLoss(nn.Layer):
    def forward(self, pred, tar):
        pred, tar = pred.flatten(1), tar.flatten(1)
        prob = F.sigmoid(pred)
        loss = 2. * (prob * tar).sum(1) / ((prob + tar).sum(1) + 1e-32)
        return 1 - loss.mean()