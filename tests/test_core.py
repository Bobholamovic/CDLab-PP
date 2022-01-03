#!/usr/bin/env python3

import sys
sys.path.insert(0, '../src')

import core.factories as F
import core.builders as B
import impl.builders


if __name__ == '__main__':
    C = dict(
        model='UNet+UNet+SNUNet',
        dataset='SVCD',
        optimizer='Adam+SGD+Adam',
        weight_decay=0.0,
        sched_on=True,
        schedulers=[
            dict(
                name='StepDecay',
                learning_rate=0.1,
                step_size=10,
                gamma=0.1
            ),
            dict(
                name='ExponentialDecay',
                learning_rate=0.05,
                gamma=0.5
            ),
            dict(
                name='StepDecay',
                learning_rate=0.001,
                step_size=10,
                gamma=0.8
            ),
        ]
    )
    model = F.model_factory(C['model'], C)
    schedulers = [B.build_scheduler(sched_cfg.pop('name'), sched_cfg) for sched_cfg in C['schedulers']]
    optimizer = F.optim_factory(C['optimizer'], model, schedulers, C)
    breakpoint()