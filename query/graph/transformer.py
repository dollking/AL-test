import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, ae_dim, num_channels=[64, 128, 256, 512], dim=128):
        super(Transformer, self).__init__()

        self.dim = self.dim
        self.ae_dim = self.ae_dim

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.FC1 = nn.Linear(num_channels[0], dim)
        self.FC2 = nn.Linear(num_channels[1], dim)
        self.FC3 = nn.Linear(num_channels[2], dim)
        self.FC4 = nn.Linear(num_channels[3], dim)

        self._conv_1 = nn.Conv2d(in_channels=dim * 4,
                                 out_channels=dim * 2,
                                 kernel_size=1, stride=1)
        self._conv_2 = nn.Conv2d(in_channels=dim * 1,
                                 out_channels=dim,
                                 kernel_size=1, stride=1)
        self._conv_3 = nn.Conv2d(in_channels=dim,
                                 out_channels=ae_dim,
                                 kernel_size=1, stride=1)

        self.batch_norm1 = nn.BatchNorm2d(dim * 2)
        self.batch_norm2 = nn.BatchNorm2d(dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x1 = self.avg_pool(features[0])
        x1 = x1.view(x1.size(0), -1)
        x1 = self.relu(self.FC1(x1))

        x2 = self.avg_pool(features[1])
        x2 = x2.view(x2.size(0), -1)
        x2 = self.relu(self.FC2(x2))

        x3 = self.avg_pool(features[2])
        x3 = x3.view(x3.size(0), -1)
        x3 = self.relu(self.FC3(x3))

        x4 = self.avg_pool(features[3])
        x4 = x4.view(x4.size(0), -1)
        x4 = self.relu(self.FC4(x4))

        x = torch.cat((x1, x2, x3, x4), 1).view([-1, self.dim * 4, 1, 1])

        x = self._conv_1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self._conv_2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self._conv_3(x)

        return x.view([-1, self.ae_dim])
