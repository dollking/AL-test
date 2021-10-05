import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], dim=128, f_dim=16):
        super(FeatureNet, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.FC1 = nn.Linear(num_channels[0], dim)
        self.FC2 = nn.Linear(num_channels[1], dim)
        self.FC3 = nn.Linear(num_channels[2], dim)
        self.FC4 = nn.Linear(num_channels[3], dim)

        self.linear = nn.Linear(4 * dim, f_dim)

    def forward(self, features):
        out1 = self.avg_pool(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.avg_pool(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.avg_pool(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.avg_pool(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out