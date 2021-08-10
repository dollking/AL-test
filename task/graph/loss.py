import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, logit, target, num_classes):
        return self.loss(logit, target.to(torch.int64))