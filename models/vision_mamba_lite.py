import torch
import torch.nn as nn


class VisionMambaLite(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.in_proj = nn.Linear(
            dim,
            dim * 2
        )

        self.dwconv = nn.Conv1d(
            dim * 2,
            dim * 2,
            kernel_size=5,
            padding=2,
            groups=dim * 2
        )

        self.gate = nn.Linear(
            dim * 2,
            dim * 2
        )

        self.act = nn.GELU()

        self.out_proj = nn.Linear(
            dim * 2,
            dim
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.in_proj(x)

        x_conv = self.dwconv(
            x.transpose(1, 2)
        ).transpose(1, 2)

        x_gate = torch.sigmoid(
            self.gate(x)
        )

        x = x_conv * x_gate

        x = self.act(x)

        x = self.out_proj(x)

        return residual + x