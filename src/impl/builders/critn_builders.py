# Custom criterion builders

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from core.misc import CRITNS


@CRITNS.register_func('WNLL_critn')
def build_weighted_nll_critn(C):
    return nn.NLLLoss(weight=paddle.to_tensor(C['weights']))


@CRITNS.register_func('WBCE_critn')
def build_weighted_bce_critn(C):
    assert len(C['weights']) == 2
    pos_weight = C['weights'][1]/C['weights'][0]
    return nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([pos_weight]))
    

@CRITNS.register_func('Cmpct_critn')
def build_compactness_critn(C):
    from utils.losses import CompactnessLoss
    return CompactnessLoss()