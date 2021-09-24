import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)

        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=num_hiddens // 4,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens // 4,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)

        return self._conv_trans_3(x)


class hash(Function):
    @staticmethod
    def forward(ctx, U):
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).cuda()
        B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)

        ctx.save_for_backward(U, B)

        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B) / (B.numel())

        grad = g + 1 * add_g

        return grad


def hash_layer(input):
    return hash.apply(input)


class VAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(VAE, self).__init__()

        self.embedding_dim = embedding_dim

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.encoder_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=num_hiddens,
                                      kernel_size=3,
                                      stride=2, padding=1)
        self.encoder_fc = nn.Linear(num_hiddens * 4 * 4, embedding_dim)

        self._decoder = Decoder(num_hiddens,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.decoder_fc = nn.Linear(embedding_dim, num_hiddens * 4 * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        encoder_out = self._encoder(x)
        encoder_out = self.relu(self.encoder_conv(encoder_out))
        _, c, w, h = encoder_out.size()
        encoder_out = torch.flatten(encoder_out, start_dim=1)
        encoder_out = self.encoder_fc(encoder_out)

        code = hash_layer(encoder_out)

        decoder_in = self.relu(self.decoder_fc(code)).view([-1, c, w, h])
        x_recon = self._decoder(decoder_in)

        return x_recon, encoder_out, code
