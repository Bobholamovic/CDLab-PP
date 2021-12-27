from core.misc import R
from .cd_trainer import CDTrainer


__all__ = []

trainer_switcher = R['Trainer_switcher']
trainer_switcher.add_item(lambda C: not (C['vdl_on'] and C['dataset'] == 'OSCD'), CDTrainer)