
import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):

    def __init__(
        self,
        channels=96,
        out_channels=1
    ):

        super().__init__()

        self.reconstruction = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.reconstruction(x)
