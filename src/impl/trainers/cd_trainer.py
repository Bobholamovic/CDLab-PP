import os
import os.path as osp

import paddle
from visualdl import LogWriter
from skimage import io
from tqdm import tqdm

from core.trainer import Trainer
from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    quantize_8bit as quantize
)
from utils.utils import HookHelper
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score)


class CDTrainer(Trainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)

        self.vdl_on = (hasattr(self.logger, 'log_path') or self.debug) and self.ctx['vdl_on']
        if self.vdl_on:
            # Initialize visualdl
            if hasattr(self.logger, 'log_path'):
                vdl_dir = self.path(
                    'log', 
                    osp.join('vdl', osp.splitext(osp.basename(self.logger.log_path))[0], '.'), 
                    name='vdl', 
                    auto_make=True, 
                    suffix=False
                )
            else:
                vdl_dir = self.path(
                    'log', 
                    osp.join('vdl', 'debug', '.'), 
                    name='vdl', 
                    auto_make=True, 
                    suffix=False
                )
                for root, dirs, files in os.walk(self.gpc.get_dir('vdl'), False):
                    for f in files:
                        os.remove(osp.join(root, f))
                    for d in dirs:
                        os.rmdir(osp.join(root, d))
            self.vdl_writer = LogWriter(logdir=vdl_dir)
            self.logger.show_nl("Tensorboard logdir: {}\n".format(osp.abspath(self.gpc.get_dir('vdl'))))
            self.vdl_intvl = self.ctx['vdl_intvl']
            
            # Global steps
            self.train_step = 0
            self.eval_step = 0

        # Whether to save network output
        self.out_dir = self.ctx['out_dir']
        self.save = self.ctx['save_on'] and not self.debug

        self.use_argmax = self.ctx['argmax_on']

    def train_epoch(self, epoch):
        losses = Meter()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        
        self.model.train()
        
        for i, (t1, t2, tar) in enumerate(pb):
            show_imgs_on_vdl = self.vdl_on and (i%self.vdl_intvl == 0)
            
            pred = self.model(t1, t2)
            if self.use_argmax:
                pred = paddle.nn.functional.log_softmax(pred, axis=1)
            else:
                pred = pred.squeeze(1)
                tar = tar.cast('float32')
            
            loss = self.criterion(pred, tar)
            
            losses.update(loss.item(), n=self.batch_size)

            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()

            desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_train, losses.val, losses.avg)

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

            if self.vdl_on:
                # Write to tensorboard
                self.vdl_writer.add_scalar("Train/loss", losses.val, self.train_step)
                if show_imgs_on_vdl:
                    t1 = to_array(t1.detach()[0])
                    t2 = to_array(t2.detach()[0])
                    self.vdl_writer.add_image("Train/t1_picked", t1, self.train_step, dataformats='HWC')
                    self.vdl_writer.add_image("Train/t2_picked", t2, self.train_step, dataformats='HWC')
                    self.vdl_writer.add_image("Train/labels_picked", tar[0].unsqueeze(0), self.train_step)
                    self.vdl_writer.flush()
                self.train_step += 1
        
        if self.vdl_on:
            self.vdl_writer.add_scalar("Train/lr", self.lr, self.train_step)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()

        with paddle.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                pred = self.model(t1, t2)
                if self.use_argmax:
                    pred = paddle.nn.functional.log_softmax(pred, dim=1)
                else:
                    pred = pred.squeeze(1)
                    tar = tar.cast('float32')

                loss = self.criterion(pred, tar)
                losses.update(loss.item())

                # Convert to numpy arrays
                if self.use_argmax:
                    cm = to_array(paddle.argmax(pred[0], 0)).astype('uint8')
                else:
                    cm = to_array(paddle.nn.functional.sigmoid(pred[0] > 0.5)).astype('uint8')
                tar = to_array(tar[0]).astype('uint8')

                for m in metrics:
                    m.update(cm, tar)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    desc += " {} {:.4f}".format(m.__name__, m.val)

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval//10) == 0)
                if dump:
                    self.logger.dump(desc)

                if self.vdl_on:
                    if dump:
                        t1 = to_array(t1[0])
                        t2 = to_array(t2[0])
                        self.vdl_writer.add_image("Eval/t1", t1, self.eval_step, dataformats='HWC')
                        self.vdl_writer.add_image("Eval/t2", t2, self.eval_step, dataformats='HWC')
                        self.vdl_writer.add_image("Eval/labels", quantize(tar), self.eval_step, dataformats='HW')
                        if self.use_argmax:
                            prob = quantize(to_array(paddle.exp(pred[0,1])))
                        else:
                            prob = to_array(paddle.nn.functional.sigmoid(pred[0]))
                        self.vdl_writer.add_image("Eval/prob", to_pseudo_color(prob), self.eval_step, dataformats='HWC')
                        self.vdl_writer.add_image("Eval/cm", quantize(cm), self.eval_step, dataformats='HW')
                    self.eval_step += 1
                
                if self.save:
                    self.save_image(name[0], quantize(cm), epoch)

        if self.vdl_on:
            self.vdl_writer.add_scalar("Eval/loss", losses.avg, self.eval_step)
            for m in metrics:
                self.vdl_writer.add_scalar(f"Eval/{m.__name__.lower()}", m.val, self.eval_step)
            self.vdl_writer.flush()

        return metrics[2].val   # F1-score

    def save_image(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return io.imsave(out_path, image)