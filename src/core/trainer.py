import shutil
import os
from abc import ABCMeta, abstractmethod

import paddle

import constants
from .misc import Logger, OutPathGetter, R
from .factories import (model_factory, optim_factory, critn_factory, data_factory)
from .builders import build_scheduler


class Trainer(metaclass=ABCMeta):
    def __init__(self, model, dataset, criterion, optimizer, settings):
        super().__init__()
        self.ctx = settings # Context
        self.mode = ('train', 'eval').index(settings['cmd'])
        self.debug = settings['debug_on']
        self.log = not settings['log_off']
        self.batch_size = settings['batch_size']
        self.checkpoint = settings['resume']
        self.load_checkpoint = (len(self.checkpoint)>0)
        self.num_epochs = settings['num_epochs']
        self.lr = settings['lr']
        self.track_intvl = settings['track_intvl']
        self.device = settings['device']

        self.gpc = OutPathGetter(
            root=os.path.join(settings['exp_dir'], settings['tag']), 
            suffix=settings['suffix']
        )   # Global Path Controller
        
        self.logger = Logger(
            scrn=True,
            log_dir=self.gpc.get_dir('log') if self.log else '',
            phase=settings['cmd']
        )
        self.path = self.gpc.get_path
        
        for k, v in sorted(settings.items()):
            self.logger.show("{}: {}".format(k,v))

        self.model = model_factory(model, settings)
        self.criterion = critn_factory(criterion, settings)

        if self.is_training:
            self.train_loader = data_factory(dataset, 'train', settings)
            self.eval_loader = data_factory(dataset, 'eval', settings)
            self.sched_on = settings['sched_on']
            if self.sched_on:
                schedulers = []
                for sched_cfg in settings['schedulers']:
                    schedulers.append(build_scheduler(sched_cfg.pop('name'), sched_cfg))
                self.optimizer = optim_factory(optimizer, self.model, schedulers, settings)
                self.schedulers = schedulers
            else:
                self.optimizer = optim_factory(optimizer, self.model, self.lr, settings)
        else:
            self.eval_loader = data_factory(dataset, 'eval', settings)
        
        self.start_epoch = 0
        self._init_acc_epoch = (0.0, -1)

    @property
    def is_training(self):
        return self.mode == 0

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def evaluate_epoch(self, epoch):
        return 0.0

    def _write_prompt(self):
        self.logger.dump(input("\nWrite some notes: "))

    def run(self):
        if self.is_training:
            if self.log and not self.debug:
                self._write_prompt()
            self.train()
        else:
            self.evaluate()

    def train(self):
        if self.load_checkpoint:
            self._resume_from_checkpoint()

        max_acc, best_epoch = self._init_acc_epoch
        lr = self.init_learning_rate()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.show_nl("Epoch: [{0}]\tlr {1:.06f}".format(epoch, lr))

            # Train for one epoch
            self.model.train()
            self.train_epoch(epoch)
            
            # Evaluate the model
            self.logger.show_nl("Evaluate")
            self.model.eval()
            acc = self.evaluate_epoch(epoch=epoch)
            
            is_best = acc > max_acc
            if is_best:
                max_acc = acc
                best_epoch = epoch
            self.logger.show_nl("Current: {:.6f} ({:03d})\tBest: {:.6f} ({:03d})\t".format(
                                acc, epoch, max_acc, best_epoch))

            # Do not save checkpoints in debugging mode
            if not self.debug:
                self._save_checkpoint(
                    self.model.state_dict(), 
                    self.optimizer.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best
                )

            lr = self.adjust_learning_rate(epoch, acc)
        
    def evaluate(self):
        if self.checkpoint: 
            if self._resume_from_checkpoint():
                self.model.eval()
                self.evaluate_epoch(self.start_epoch)
        else:
            self.logger.error("No checkpoint assigned.")

    def init_learning_rate(self):
        if self.sched_on:
            self.lr = self.schedulers[0]()
        return self.lr

    def adjust_learning_rate(self, epoch, acc):
        if self.sched_on:
            for scheduler in self.schedulers:
                if isinstance(scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                    scheduler.step(acc)
                else:
                    scheduler.step()
            return self.schedulers[0]()
        else:
            return self.lr

    def _resume_from_checkpoint(self):
        # XXX: This could be slow!
        if not os.path.isfile(self.checkpoint):
            self.logger.error("=> No checkpoint was found at '{}'.".format(self.checkpoint))
            return False

        self.logger.show("=> Loading checkpoint '{}'...".format(self.checkpoint))
        checkpoint = paddle.load(self.checkpoint)

        state_dict = self.model.state_dict()
        ckp_dict = checkpoint.get('state_dict', checkpoint)
        update_dict = {
            k:v for k,v in ckp_dict.items() 
            if k in state_dict and state_dict[k].shape == v.shape and state_dict[k].dtype == v.dtype
        }
        
        num_to_update = len(update_dict)
        if (num_to_update < len(state_dict)) or (len(state_dict) < len(ckp_dict)):
            if not self.is_training and (num_to_update < len(state_dict)):
                self.logger.error("=> Mismatched checkpoint for evaluation")
                return False
            self.logger.warn("Trying to load a mismatched checkpoint.")
            if num_to_update == 0:
                self.logger.error("=> No parameter is to be loaded.")
                return False
            else:
                self.logger.warn("=> {} params are to be loaded.".format(num_to_update))
            ckp_epoch = -1
        else:
            ckp_epoch = checkpoint.get('epoch', -1)
            if not self.is_training:
                self.start_epoch = ckp_epoch
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))
            elif not self.ctx['anew']:
                self.start_epoch = ckp_epoch+1
                if self.ctx['load_optim']:
                    # XXX: Note that weight decay might be modified here.
                    self.optimizer.set_state_dict(checkpoint['optimizer'])
                    self.logger.warn("Weight decay might have been modified.")
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))

        state_dict.update(update_dict)
        self.model.set_state_dict(state_dict)

        if ckp_epoch == -1:
            self.logger.show("=> Loaded checkpoint '{}'".format(self.checkpoint))
        else:
            self.logger.show("=> Loaded checkpoint '{}' (epoch {}, max_acc {:.4f} at epoch {}).".format(
                self.checkpoint, ckp_epoch, *self._init_acc_epoch
                ))
        return True
        
    def _save_checkpoint(self, state_dict, optim_state, max_acc, epoch, is_best):
        state = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optim_state, 
            'max_acc': max_acc
        } 
        if (epoch+1) % self.track_intvl == 0:
            # Save history
            # epoch+1 instead of epoch is contained in the checkpoint name so that it will be easy for 
            # one to recognize "the next start_epoch". 
            history_path = self.path(
                'weight', constants.CKP_COUNTED.format(e=epoch+1), 
                suffix=True
            )
            paddle.save(state, history_path)
        # Save latest
        latest_path = self.path(
            'weight', constants.CKP_LATEST, 
            suffix=True
        )
        paddle.save(state, latest_path)
        if is_best:
            shutil.copyfile(
                latest_path, self.path(
                    'weight', constants.CKP_BEST, 
                    suffix=True
                )
            )


class TrainerSwitcher:
    r"""A simple utility class to help dispatch actions to different trainers."""
    def __init__(self, *pairs):
        self._trainer_list = list(pairs)

    def __call__(self, args, return_obj=True):
        for p, t in self._trainer_list:
            if p(args):
                return t(args) if return_obj else t
        return None

    def add_item(self, predicate, trainer):
        # Newly added items have higher priority
        self._trainer_list.insert(0, (predicate, trainer))

    def add_default(self, trainer):
        self._trainer_list.append((lambda _: True, trainer))


R.register('Trainer_switcher', TrainerSwitcher())