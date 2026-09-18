import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.frequency_branch import FrequencyBranch
from models.ifft_layer import IFFTLayer
from models.spatial_branch_swinir import SpatialBranchSwinIR
from models.awgm import AWGM
from models.afi import AFI
from models.reconstruction_head import ReconstructionHead

class DAFINet(nn.Module):

    def __init__(self):

        super().__init__()

        self.frequency_branch = FrequencyBranch()

        self.ifft_layer = IFFTLayer()

        self.spatial_branch_swinir = SpatialBranchSwinIR()

        self.awgm = AWGM()

        self.afi = AFI()

        self.reconstruction_head = ReconstructionHead()

    def forward(self, kspace):

        Ff = self.frequency_branch(kspace)

        image = self.ifft_layer(kspace)

        Fs = self.spatial_branch_swinir(image)

        W = self.awgm(Fs, Ff)

        F_afi = self.afi(Fs, Ff, W)

        out = self.reconstruction_head(F_afi)

        return out

# Without AWGM
# class DAFINet(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.frequency_branch = FrequencyBranch()

#         self.ifft_layer = IFFTLayer()

#         self.spatial_branch_swinir = SpatialBranchSwinIR()

#         # AWGM removed

#         self.afi = AFI()

#         self.reconstruction_head = ReconstructionHead()

#     def forward(self, kspace):

#         Ff = self.frequency_branch(kspace)

#         image = self.ifft_layer(kspace)

#         Fs = self.spatial_branch_swinir(image)

#         # Fixed fusion instead of AWGM
#         F_afi = 0.5 * Fs + 0.5 * Ff

#         out = self.reconstruction_head(F_afi)

#         return out


# Without Frequncy Branch
# class DAFINet(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.ifft_layer = IFFTLayer()

#         self.spatial_branch_swinir = SpatialBranchSwinIR()

#         self.reconstruction_head = ReconstructionHead()

#     def forward(self, kspace):

#         image = self.ifft_layer(kspace)

#         Fs = self.spatial_branch_swinir(image)

#         out = self.reconstruction_head(Fs)

#         return out