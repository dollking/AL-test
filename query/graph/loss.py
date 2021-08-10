import torch
import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, recon, target):
        return self.loss(recon, target)
