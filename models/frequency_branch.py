import torch
import torch.nn as nn

from models.vision_mamba_lite import VisionMambaLite
from models.eca import ECALayer
# import torch
# import torch.nn as nn


# # =====================================================
# # Vision Mamba Block
# # =====================================================

# class VisionMambaBlock(nn.Module):

#     def __init__(
#         self,
#         dim
#     ):
#         super().__init__()

#         self.norm = nn.LayerNorm(dim)

#         self.fc1 = nn.Linear(
#             dim,
#             dim * 2
#         )

#         self.dwconv = nn.Conv1d(
#             dim * 2,
#             dim * 2,
#             kernel_size=3,
#             padding=1,
#             groups=dim * 2
#         )

#         self.act = nn.GELU()

#         self.fc2 = nn.Linear(
#             dim * 2,
#             dim
#         )

#     def forward(
#         self,
#         x
#     ):

#         residual = x

#         x = self.norm(x)

#         x = self.fc1(x)

#         x = x.transpose(
#             1,
#             2
#         )

#         x = self.dwconv(x)

#         x = x.transpose(
#             1,
#             2
#         )

#         x = self.act(x)

#         x = self.fc2(x)

#         return residual + x


# # =====================================================
# # Frequency Branch
# # =====================================================

# class FrequencyBranch(nn.Module):

#     def __init__(
#         self,
#         in_channels=2,
#         embed_dim=96
#     ):

#         super().__init__()

#         # --------------------------------
#         # Patch Embedding
#         # --------------------------------

#         self.patch_embed = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 embed_dim,
#                 kernel_size=2,
#                 stride=2
#             ),
#             nn.BatchNorm2d(embed_dim),
#             nn.GELU()
#         )

#         # --------------------------------
#         # Vision Mamba Blocks (×4)
#         # --------------------------------

#         self.mamba1 = VisionMambaBlock(
#             embed_dim
#         )

#         self.mamba2 = VisionMambaBlock(
#             embed_dim
#         )

#         self.mamba3 = VisionMambaBlock(
#             embed_dim
#         )

#         self.mamba4 = VisionMambaBlock(
#             embed_dim
#         )

#         # --------------------------------
#         # Upsampling
#         # --------------------------------

#         self.proj = nn.Sequential(

#             nn.ConvTranspose2d(
#                 embed_dim,
#                 embed_dim,
#                 kernel_size=2,
#                 stride=2
#             ),

#             nn.Conv2d(
#                 embed_dim,
#                 embed_dim,
#                 kernel_size=3,
#                 padding=1
#             ),

#             nn.GELU()
#         )

#     def forward(
#         self,
#         kspace
#     ):

#         B, C, H, W = kspace.shape

#         # --------------------------------
#         # Patch Embedding
#         # --------------------------------

#         x = self.patch_embed(
#             kspace
#         )

#         B, C_embed, Hh, Wh = x.shape

#         # --------------------------------
#         # Tokens
#         # --------------------------------

#         x = x.flatten(2)

#         x = x.transpose(
#             1,
#             2
#         )

#         # Shape:
#         # B, N, embed_dim

#         # --------------------------------
#         # Vision Mamba × 4
#         # --------------------------------

#         x = self.mamba1(x)

#         x = self.mamba2(x)

#         x = self.mamba3(x)

#         x = self.mamba4(x)

#         # --------------------------------
#         # Reshape
#         # --------------------------------

#         x = x.transpose(
#             1,
#             2
#         )

#         x = x.reshape(
#             B,
#             C_embed,
#             Hh,
#             Wh
#         )

#         # --------------------------------
#         # Projection
#         # --------------------------------

#         out = self.proj(x)

#         return out
class FrequencyBranch(nn.Module):

    def __init__(
        self,
        in_channels=2,
        embed_dim=96
    ):

        super().__init__()

        # ==================================
        # Multi-Scale Frequency Embedding
        # ==================================

        self.embed3 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU()
        )

        self.embed5 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.GELU()
        )

        self.embed7 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.GELU()
        )

        # ==================================
        # VisionMambaLite ×4
        # ==================================

        self.mamba1 = VisionMambaLite(embed_dim)

        self.mamba2 = VisionMambaLite(embed_dim)

        self.mamba3 = VisionMambaLite(embed_dim)

        self.mamba4 = VisionMambaLite(embed_dim)

        # ==================================
        # ECA
        # ==================================

        self.eca = ECALayer(
            embed_dim,
            k_size=3
        )

        # ==================================
        # Projection
        # ==================================

        self.proj = nn.Sequential(

            nn.ConvTranspose2d(
                embed_dim,
                embed_dim,
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )

    def forward(self, kspace):

        B = kspace.size(0)

        # ==================================
        # Multi-scale embedding
        # ==================================

        x1 = self.embed3(kspace)

        x2 = self.embed5(kspace)

        x3 = self.embed7(kspace)

        x = torch.cat(
            [x1, x2, x3],
            dim=1
        )

        B, C, H, W = x.shape

        # ==================================
        # Tokens
        # ==================================

        x = x.flatten(2)

        x = x.transpose(1, 2)

        # ==================================
        # VisionMambaLite
        # ==================================

        x = self.mamba1(x)

        x = self.mamba2(x)

        x = self.mamba3(x)

        x = self.mamba4(x)

        # ==================================
        # Back To Feature Map
        # ==================================

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            C,
            H,
            W
        )

        # ==================================
        # ECA
        # ==================================

        x = self.eca(x)

        # ==================================
        # Projection
        # ==================================

        out = self.proj(x)

        return out