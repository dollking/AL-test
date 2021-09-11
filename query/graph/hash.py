import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels // 2,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels // 2,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class Extract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Extract, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=4,
                              stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Hash(nn.Module):
    def __init__(self, code_dim, channels=[64, 128, 256, 512]):
        super(Hash, self).__init__()

        self.channels = channels
        self.module1 = self._make_layer(self.channels[0], 3)
        self.module2 = self._make_layer(self.channels[1], 2)
        self.module3 = self._make_layer(self.channels[2], 1)

        self.linear = nn.Linear(4 * self.channels[3], code_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, in_channel, repeat):
        layers = []
        for i in range(repeat):
            layers.append(Extract(in_channel, in_channel * 2))
            in_channel *= 2
        layers.append(Residual(in_channel * (2 ** repeat)))
        return nn.Sequential(*layers)

    def forward(self, x):
        feature1 = self.avg_pool(self.module1(x[0].detach())).view([-1, self.channels[3]])
        feature2 = self.avg_pool(self.module2(x[1].detach())).view([-1, self.channels[3]])
        feature3 = self.avg_pool(self.module3(x[2].detach())).view([-1, self.channels[3]])
        feature4 = self.avg_pool(x[3]).view([-1, self.channels[3]])

        feature = torch.cat([feature1, feature2, feature3, feature4], dim=1)

        return self.linear(feature)
