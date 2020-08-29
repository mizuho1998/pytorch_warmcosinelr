import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class WarmCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup=10, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup = warmup
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(WarmCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [self.eta_min + 1e-8 for _ in self.base_lrs]
        if self.last_epoch < self.warmup:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi * self.last_epoch / self.warmup)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup) / (self.T_max - self.warmup))) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup - 1) / (self.T_max - self.warmup))) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
