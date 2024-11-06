import torch


class MSELoss:
    def __init__(self, weight=1, reduction='mean'):
        self.weight = weight

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def __call__(self, pred, target):
        loss = self.mse_loss(pred, target)

        return loss
