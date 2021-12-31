import paddle

from .cd_trainer import CDTrainer


class STANetTrainer(CDTrainer):
    def _init_trainer(self):
        self.thresh = self.ctx['threshold']

    def _prepare_data(self, t1, t2, tar):
        return t1, t2, tar.astype('float32')

    def _pred_to_prob(self, pred):
        return (pred>self.thresh).astype('float32')