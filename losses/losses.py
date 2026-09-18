import torch
import torch.nn as nn

from pytorch_msssim import SSIM
from pytorch_wavelets import DWTForward


# =====================================================
# Charbonnier Loss
# =====================================================

class CharbonnierLoss(nn.Module):

    def __init__(
        self,
        eps=1e-6
    ):
        super().__init__()

        self.eps = eps

    def forward(
        self,
        pred,
        target
    ):

        diff = pred - target

        loss = torch.mean(
            torch.sqrt(
                diff * diff + self.eps
            )
        )

        return loss


# =====================================================
# SSIM Loss
# =====================================================

class SSIMLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ssim = SSIM(
            data_range=1.0,
            size_average=True,
            channel=1
        )

    def forward(
        self,
        pred,
        target
    ):

        return 1.0 - self.ssim(
            pred,
            target
        )


# =====================================================
# Edge Loss
# =====================================================

class EdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred,
        target
    ):

        pred_dx = (
            pred[:, :, :, 1:]
            -
            pred[:, :, :, :-1]
        )

        pred_dy = (
            pred[:, :, 1:, :]
            -
            pred[:, :, :-1, :]
        )

        target_dx = (
            target[:, :, :, 1:]
            -
            target[:, :, :, :-1]
        )

        target_dy = (
            target[:, :, 1:, :]
            -
            target[:, :, :-1, :]
        )

        loss_x = self.l1(
            pred_dx,
            target_dx
        )

        loss_y = self.l1(
            pred_dy,
            target_dy
        )

        return loss_x + loss_y


# =====================================================
# Wavelet Loss
# =====================================================

class WaveletLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.dwt = DWTForward(
            J=1,
            wave='haar',
            mode='zero'
        )
        self.dwt.requires_grad_(False)
    def forward(
        self,
        pred,
        target
    ):

        pred_ll, pred_high = self.dwt(
            pred
        )

        target_ll, target_high = self.dwt(
            target
        )

        loss_ll = self.l1(
            pred_ll,
            target_ll
        )

        pred_h = pred_high[0]
        target_h = target_high[0]

        loss_high = self.l1(
            pred_h,
            target_h
        )

        return loss_ll + loss_high


# =====================================================
# Total DAFI Loss
# =====================================================

class DAFILoss(nn.Module):

    def __init__(
        self,
        lambda_ssim=0.10,
        lambda_edge=0.03,
        lambda_wavelet=0.10
    ):

        super().__init__()

        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_wavelet = lambda_wavelet

        self.pixel_loss = CharbonnierLoss()

        self.ssim_loss = SSIMLoss()

        self.edge_loss = EdgeLoss()

        self.wavelet_loss = WaveletLoss()

    def forward(
        self,
        pred,
        target
    ):

        lpixel = self.pixel_loss(
            pred,
            target
        )

        lssim = self.ssim_loss(
            pred,
            target
        )

        ledge = self.edge_loss(
            pred,
            target
        )

        lwave = self.wavelet_loss(
            pred,
            target
        )

        total = (
            lpixel
            +
            self.lambda_ssim * lssim
            +
            self.lambda_edge * ledge
            +
            self.lambda_wavelet * lwave
        )

        return total, {
            "pixel": lpixel.item(),
            "ssim": lssim.item(),
            "edge": ledge.item(),
            "wavelet": lwave.item()
        }