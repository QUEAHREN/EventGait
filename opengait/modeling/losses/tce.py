from .base import BaseLoss
import torch.nn.functional as F


class TraditionalCE(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(TraditionalCE, self).__init__()
        self.ce = F.cross_entropy
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels, label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels)
        loss = self.ce(logits*self.scale, labels)
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info