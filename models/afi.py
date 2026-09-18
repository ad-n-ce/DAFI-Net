
import torch
import torch.nn as nn


class AFI(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        Fs,
        Ff,
        W
    ):

        F_afi = (
            W * Ff
            +
            (1 - W) * Fs
        )

        return F_afi
