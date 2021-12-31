from core.misc import R
from .cd_trainer import CDTrainer
from .cd_trainer_nll import CDTrainer_NLL
from .cd_trainer_bce import CDTrainer_BCE
from .ifn_trainer import IFNTrainer
from .stanet_trainer import STANetTrainer
from .dsamnet_trainer import DSAMNetTrainer


__all__ = []

trainer_switcher = R['Trainer_switcher']
trainer_switcher.add_default(CDTrainer_BCE)
trainer_switcher.add_item(lambda C: C['argmax_on'], CDTrainer_NLL)
trainer_switcher.add_item(lambda C: C['model'] in ('IFN', 'P2V'), IFNTrainer)
trainer_switcher.add_item(lambda C: C['model']=='STANet', STANetTrainer)
trainer_switcher.add_item(lambda C: C['model']=='DSAMNet', DSAMNetTrainer)