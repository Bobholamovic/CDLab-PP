from inspect import isfunction, isgeneratorfunction, getmembers
from collections.abc import Sequence
from abc import ABC
import paddle
import paddle.nn as nn
import paddle.io as io

from .misc import (R, MODELS, OPTIMS, CRITNS, DATA)


class _AttrDesc:
    def __init__(self, key):
        self.key = key
    def __get__(self, instance, owner):
        return tuple(getattr(ele, self.key) for ele in instance)
    def __set__(self, instance, value):
        for ele in instance:
            setattr(ele, self.key, value)


def _func_deco(func_name):
    # FIXME: The signature of the wrapped function will be lost.
    def _wrapper(self, *args, **kwargs):
        return tuple(getattr(ele, func_name)(*args, **kwargs) for ele in self)
    return _wrapper


def _generator_deco(func_name):
    # FIXME: The signature of the wrapped function will be lost.
    def _wrapper(self, *args, **kwargs):
        for ele in self:
            yield from getattr(ele, func_name)(*args, **kwargs)
    return _wrapper


# Duck typing
class Duck(Sequence, ABC):
    __ducktype__ = object
    __ava__ = ()
    def __init__(self, *args):
        if not all(map(self._check, args)):
            raise TypeError("Please check the input type.")
        self._seq = tuple(args)

    def __getitem__(self, key):
        return self._seq[key]

    def __len__(self):
        return len(self._seq)

    def __repr__(self):
        return repr(self._seq)

    @classmethod
    def _check(cls, obj):
        for attr in cls.__ava__:
            try:
                getattr(obj, attr)
            except AttributeError:
                return False
        return True


def duck_it(cls):
    members = dict(getmembers(cls.__ducktype__))  # Trade space for time
    for k in cls.__ava__:
        if k in members:
            v = members[k]
            if isgeneratorfunction(v):
                setattr(cls, k, _generator_deco(k))
            elif isfunction(v):
                setattr(cls, k, _func_deco(k))
            else:
                setattr(cls, k, _AttrDesc(k))
    return cls


class DuckModel(nn.Layer):
    __ducktype__ = nn.Layer
    __ava__ = ('state_dict', 'set_state_dict', 'forward', '__call__', 'train', 'eval', 'to')

    def __init__(self, *models):
        super().__init__()
        # XXX: The state_dict will be a little larger in size,
        # since some extra bytes are stored in every key.
        self._m = nn.LayerList(models)

    def __len__(self):
        return len(self._m)

    def __getitem__(self, idx):
        return self._m[idx]

    def __contains__(self, m):
        return m in self._m

    def __repr__(self):
        return repr(self._m)

    def forward(self, *args, **kwargs):
        return tuple(m(*args, **kwargs) for m in self._m)


Duck.register(DuckModel)


@duck_it
class DuckOptimizer(Duck):
    __ducktype__ = paddle.optimizer.Optimizer
    __ava__ = ('state_dict', 'set_state_dict', 'clear_grad', 'step', 'get_lr')

    # Sepcial dispatching rule
    def set_state_dict(self, state_dicts):
        for optim, state_dict in zip(self, state_dicts):
            optim.set_state_dict(state_dict)


@duck_it
class DuckCriterion(Duck):
    __ducktype__ = nn.Layer
    __ava__ = ('forward', '__call__', 'train', 'eval', 'to')

    pass


@duck_it
class DuckDataLoader(Duck):
    __ducktype__ = io.DataLoader
    __ava__ = ()
    
    pass


def single_model_factory(model_name, C):
    builder_name = '_'.join([model_name, C['model'], C['dataset'], 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    builder_name = '_'.join([model_name, C['dataset'], 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    builder_name = '_'.join([model_name, C['model'], 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    builder_name = '_'.join([model_name, 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    else:
        raise RuntimeError("{} is not a supported architecture.".format(model_name))


def single_optim_factory(optim_name, params, lr, C):
    builder_name = '_'.join([optim_name, 'optim'])
    if builder_name not in OPTIMS:
        raise RuntimeError("{} is not a supported optimizer type.".format(optim_name))
    return OPTIMS[builder_name](params, lr, C)
        

def single_critn_factory(critn_name, C):
    builder_name = '_'.join([critn_name, 'critn'])
    if builder_name not in CRITNS:
        raise RuntimeError("{} is not a supported criterion type.".format(critn_name))
    return CRITNS[builder_name](C)
        

def single_data_factory(dataset_name, phase, C):
    builder_name = '_'.join([dataset_name, C['dataset'], C['model'], phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    builder_name = '_'.join([dataset_name, C['model'], phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    builder_name = '_'.join([dataset_name, C['dataset'], phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    builder_name = '_'.join([dataset_name, phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    else:
        raise RuntimeError("{} is not a supported dataset.".format(dataset_name))


def _parse_input_names(name_str):
    return name_str.split('+')


def model_factory(model_names, C):
    name_list = _parse_input_names(model_names)
    if len(name_list) > 1:
        return DuckModel(*(single_model_factory(name, C) for name in name_list))
    else:
        return single_model_factory(model_names, C)


def optim_factory(optim_names, models, lrs, C):
    name_list = _parse_input_names(optim_names)
    num_models = len(models) if isinstance(models, DuckModel) else 1
    if not isinstance(lrs, Sequence):
        lrs = [lrs]*num_models
    if num_models > 1:
        optimizers = []
        for name, model, lr in zip(name_list, models, lrs):
            optimizers.append(single_optim_factory(name, model.parameters(), lr, C))
        return DuckOptimizer(*optimizers)
    else:
        return single_optim_factory(optim_names, models.parameters(), lrs[0], C)


def critn_factory(critn_names, C):
    name_list = _parse_input_names(critn_names)
    if len(name_list) > 1:
        return DuckCriterion(*(single_critn_factory(name, C) for name in name_list))
    else:
        return single_critn_factory(critn_names, C)


def data_factory(dataset_names, phase, C):
    name_list = _parse_input_names(dataset_names)
    if len(name_list) > 1:
        return DuckDataLoader(*(single_data_factory(name, phase, C) for name in name_list))
    else:
        return single_data_factory(dataset_names, phase, C)