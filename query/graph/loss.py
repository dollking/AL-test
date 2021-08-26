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

        self.loss = nn.NLLLoss()
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, indices_1, indices_2, num_classes):
        one_hot = nn.functional.one_hot(indices_2, num_classes).type(torch.float).cuda()

        return self.loss(self.m(one_hot), indices_1)
