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
