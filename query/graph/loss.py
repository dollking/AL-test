import torch
import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, recon, target):
        return self.loss(recon, target)


class SelfClusteringLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, indices, inverse_distances, num_classes):
        return self.loss(inverse_distances, indices)


class CodeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, origin_code, trans_code):
        code_balance_loss = (torch.mean(torch.abs(torch.sum(origin_code, dim=1))) +
                             torch.mean(torch.abs(torch.sum(trans_code, dim=1)))) / 2
        code_loss = self.loss(trans_code, origin_code.detach())

        return code_balance_loss, code_loss


class HashLoss(nn.Module):
    def __int__(self):
        super(HashLoss, self).__init__()

    def forward(self, code, cls, m):
        y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
        dist = ((code.unsqueeze(0) - code.unsqueeze(1)) ** 2).sum(dim=2).view(-1)

        loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)

        loss = loss.mean() + 0.01 * (code.abs() - 1).abs().sum(dim=1).mean() * 2

        return loss