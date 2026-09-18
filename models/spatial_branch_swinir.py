
import torch
import torch.nn as nn

from models.network_swinir import (
    RSTB
)


class SpatialBranchSwinIR(nn.Module):

    def __init__(
        self,
        img_size=256,
        in_channels=1,
        embed_dim=96,
        num_heads=6,
        window_size=8
    ):

        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        self.rstb1 = RSTB(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            depth=2,
            num_heads=num_heads,
            window_size=window_size,
            img_size=img_size,
            patch_size=1,
            resi_connection='1conv'
        )

        self.rstb2 = RSTB(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            depth=2,
            num_heads=num_heads,
            window_size=window_size,
            img_size=img_size,
            patch_size=1,
            resi_connection='1conv'
        )

        self.tail = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.head(x)

        B, C, H, W = x.shape

        residual = x

        # BCHW -> BLC
        x = x.flatten(2).transpose(1, 2)

        x = self.rstb1(x, (H, W))

        x = self.rstb2(x, (H, W))

        # BLC -> BCHW
        x = x.transpose(1, 2).reshape(
            B,
            C,
            H,
            W
        )

        x = self.tail(x)

        x = x + residual

        return x
