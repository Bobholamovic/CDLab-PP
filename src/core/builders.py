# Built-in builders

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .misc import (MODELS, OPTIMS, CRITNS, DATA)


# Optimizer builders
@OPTIMS.register_func('Adam_optim')
def build_Adam_optim(params, lr, C):
    return paddle.optimizer.Adam(
        learning_rate=lr,
        parameters=params,
        beta1=0.9,
        beta2=0.999,
        weight_decay=C['weight_decay']
    )


@OPTIMS.register_func('SGD_optim')
def build_SGD_optim(params, lr, C):
    return paddle.optimizer.Momentum(
        learning_rate=lr,
        parameters=params,
        momentum=0.9,
        weight_decay=C['weight_decay']
    )


# Criterion builders
@CRITNS.register_func('L1_critn')
def build_L1_critn(C):
    return nn.L1Loss()


@CRITNS.register_func('MSE_critn')
def build_MSE_critn(C):
    return nn.MSELoss()


@CRITNS.register_func('CE_critn')
def build_CE_critn(C):
    return nn.CrossEntropyLoss()


@CRITNS.register_func('NLL_critn')
def build_NLL_critn(C):
    return nn.NLLLoss()


def build_scheduler(name, cfg):
    sched_cls = getattr(paddle.optimizer.lr, name)
    sched_obj = sched_cls(**cfg)
    return sched_obj