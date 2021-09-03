import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target, num_classes):
        return self.loss(logit, target.to(torch.int64))


class RankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, pred_loss, target_loss):
        target = (target_loss - target_loss.flip(0))[:target_loss.size(0) // 2]
        target = target.detach()
        ones = torch.sign(torch.clamp(target, min=0))

        pred_loss = (pred_loss - pred_loss.flip(0))[:pred_loss.size(0) // 2]
        pred_loss = torch.sigmoid(pred_loss)

        return self.bce(pred_loss, ones)