
import torch
import torch.nn as nn


class AWGM(nn.Module):

    def __init__(
        self,
        channels=96,
        reduction=16
    ):

        super().__init__()

        hidden_dim = channels // reduction

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(

            nn.Linear(
                channels * 2,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                channels
            ),

            nn.Sigmoid()
        )

    def forward(self, Fs, Ff):

        bs = Fs.size(0)

        zs = self.gap(Fs).view(bs, -1)

        zf = self.gap(Ff).view(bs, -1)

        z = torch.cat(
            [zs, zf],
            dim=1
        )

        W = self.mlp(z)

        W = W.view(
            bs,
            -1,
            1,
            1
        )

        return W
