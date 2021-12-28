import paddle

from .cd_trainer import CDTrainer


class CDTrainer_NLL(CDTrainer):
    def _process_model_out(self, out):
        return paddle.nn.functional.log_softmax(out, axis=1)

    def _pred_to_prob(self, pred):
        return paddle.exp(pred[:,1])