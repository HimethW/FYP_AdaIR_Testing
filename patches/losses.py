"""Restoration loss: pixel L1 + 0.1 * frequency-domain L1.

    L = ||x_hat - x||_1 + w * ||FFT(x_hat) - FFT(x)||_1,   w = 0.1

FFT term as in the AdaIR author's SFNet / FocalNet training code (and MIMO-UNet): torch.fft.fft2
over (H, W) with the default (unnormalised) norm, real and imaginary parts stacked, mean L1.
Unnormalised means the FFT term grows with the crop size: on 128 crops it is ~90x the pixel term for
noise-like errors, so at w = 0.1 it is ~90% of the loss (~46% for haze-like errors); results.md A7.
norm="ortho" (energy-preserving FFT) makes it ~sqrt(H*W) times smaller (~7% of the loss at w = 0.1),
i.e. a small add-on instead of the main term. Default follows the author's convention; A/B to decide.

Computed in fp32 also under autocast (torch.fft has no bf16 support).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_l1(pred, target, norm="backward"):
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred_f = torch.fft.fft2(pred.float(), dim=(-2, -1), norm=norm)
        target_f = torch.fft.fft2(target.float(), dim=(-2, -1), norm=norm)
        return F.l1_loss(torch.stack((pred_f.real, pred_f.imag), -1),
                         torch.stack((target_f.real, target_f.imag), -1))


class PixelFFTLoss(nn.Module):
    """forward returns (total, {"l_pix": ..., "l_fft": ...}) for logging."""

    def __init__(self, fft_weight=0.1, norm="backward"):
        super().__init__()
        self.fft_weight = fft_weight
        self.norm = norm

    def forward(self, pred, target):
        l_pix = F.l1_loss(pred, target)
        l_fft = fft_l1(pred, target, self.norm)
        return l_pix + self.fft_weight * l_fft, {"l_pix": l_pix.detach(), "l_fft": l_fft.detach()}
