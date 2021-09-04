import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target, num_classes):
        return self.loss(logit, target)


class LossPredLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_loss, target_loss):
        pred_loss = (pred_loss - pred_loss.flip(0))[:len(pred_loss)//2]
        
        target_loss = (target_loss - target_loss.flip(0))[:len(target_loss)//2]
        target_loss = target_loss.detach()

        one = 2 * torch.sign(torch.clamp(target_loss, min=0)) - 1

        loss = torch.sum(torch.clamp(1.0 - one * pred_loss, min=0))
        loss = loss / pred_loss.size(0)

        return loss


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