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

        self.loss = nn.BCELoss()

    def forward(self, origin_code, trans_code):
        code_loss = (torch.mean(torch.abs(1 - torch.sum(origin_code, dim=1))) +
                     torch.mean(torch.abs(1 - torch.sum(trans_code, dim=1)))) / 2

        return code_loss + self.loss(origin_code, trans_code)