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

    def forward(self, indices_1, indices_2, num_classes):
        one_hot_1 = nn.functional.one_hot(indices_1, num_classes)
        one_hot_2 = nn.functional.one_hot(indices_2, num_classes)

        return self.loss(one_hot_1, one_hot_2)
