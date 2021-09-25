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

    def forward(self, origin_logit, trans_logit):
        origin_code, trans_code = torch.sign(origin_logit), torch.sign(trans_logit)

        code_balance_loss = (torch.mean(torch.abs(torch.sum(origin_code, dim=1))) +
                             torch.mean(torch.abs(torch.sum(trans_code, dim=1)))) / 2

        code_loss = self.loss(trans_code, origin_code.detach())

        return code_balance_loss, code_loss


class HashLoss(nn.Module):
    def __init__(self):
        super(HashLoss, self).__init__()

    def forward(self, code, cls, m):
        y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
        dist = ((code.unsqueeze(0) - code.unsqueeze(1)) ** 2).sum(dim=2).view(-1)

        loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)

        loss = loss.mean() + 0.01 * (code.abs() - 1).abs().sum(dim=1).mean() * 2

        return loss


class BHLoss(nn.Module):
    def __init__(self):
        super(BHLoss, self).__init__()

        self.loss = nn.MSELoss()

    def forward(self, b, x, size):
        target_b = nn.functional.cosine_similarity(b[:int(size / 2)], b[int(size / 2):])
        target_x = nn.functional.cosine_similarity(x[:int(size / 2)], x[int(size / 2):])

        return self.loss(target_b, target_x)


class KldLoss(nn.Module):
    def __init__(self):
        super(KldLoss, self).__init__()

    def forward(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)