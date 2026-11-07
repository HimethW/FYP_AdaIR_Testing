"""Restoration losses: pixel L1 + 0.1 * frequency-domain L1 (PixelFFTLoss), and the L1 / MS-SSIM mix (MixLoss, below).

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


# ---------------------------------------------------------------------------------------------------------------
# MS-SSIM and the L1 / MS-SSIM mix (CDD-11 run).
#
#   L = a * ||x_hat - x||_1 + (1 - a) * (1 - MS-SSIM(x_hat, x)) + w * FFT-L1,   w = 0.1 (as above)
#
# The L1 / MS-SSIM pair and the start a = 0.8 follow EvoIR (Ma et al., arXiv 2512.05104), which searches
# (a, 1 - a) during training; train_fyp.py --mix_search picks a (see there). a = 1 is our original loss.
# MS-SSIM: Wang et al. 2003 (Gaussian window 11, sigma 1.5, K1 0.01, K2 0.03, data range 1). Training crops are
# 128 px, too small for the usual 5 scales (the 5th would be 8 px, below the 11 px window), so we use the first 3
# scales with their standard weights (0.0448, 0.2856, 0.3001) renormalised to sum to 1. Contrast terms are
# clamped at 1e-6 (instead of a ReLU) so the fractional powers never see 0 and their gradients stay finite.
# ---------------------------------------------------------------------------------------------------------------
MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001)


def _gauss(win, sigma, device):
    x = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _blur(x, g):
    c = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def ms_ssim(pred, target, weights=MSSSIM_WEIGHTS, win=11, sigma=1.5, k1=0.01, k2=0.03):
    """Mean MS-SSIM of NCHW images in [0, 1] (fp32)."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        x, y = pred.float(), target.float()
        g = _gauss(win, sigma, x.device)
        w = torch.tensor(weights, dtype=torch.float32, device=x.device)
        w = w / w.sum()
        c1, c2 = k1 ** 2, k2 ** 2
        terms = []
        for i in range(len(weights)):
            mx, my = _blur(x, g), _blur(y, g)
            sxx, syy, sxy = _blur(x * x, g) - mx ** 2, _blur(y * y, g) - my ** 2, _blur(x * y, g) - mx * my
            cs = ((2 * sxy + c2) / (sxx + syy + c2)).flatten(2).mean(-1)  # N x C
            if i < len(weights) - 1:
                terms.append(cs.clamp_min(1e-6))
                x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
            else:
                lum = (2 * mx * my + c1) / (mx ** 2 + my ** 2 + c1)
                ssim = (lum * (2 * sxy + c2) / (sxx + syy + c2)).flatten(2).mean(-1)
                terms.append(ssim.clamp_min(1e-6))
        return torch.prod(torch.stack(terms) ** w.view(-1, 1, 1), 0).mean()


class MixLoss(nn.Module):
    """a * L1 + (1 - a) * (1 - MS-SSIM) + fft_weight * FFT-L1; `a` can be changed between steps."""

    def __init__(self, a=0.8, fft_weight=0.1, norm="backward"):
        super().__init__()
        self.a = a
        self.fft_weight = fft_weight
        self.norm = norm

    def forward(self, pred, target):
        l_pix = F.l1_loss(pred, target)
        l_ms = 1 - ms_ssim(pred, target)
        l_fft = fft_l1(pred, target, self.norm)
        total = self.a * l_pix + (1 - self.a) * l_ms + self.fft_weight * l_fft
        return total, {"l_pix": l_pix.detach(), "l_msssim": l_ms.detach(), "l_fft": l_fft.detach()}
