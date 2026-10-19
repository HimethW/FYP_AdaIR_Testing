"""PSNR / SSIM as in AdaIR's utils/val_utils.py (RGB, [0, 1], skimage), with the current skimage API.

The upstream compute_psnr_ssim passes `multichannel=True`, which scikit-image >= 0.21 no longer has
(it then treats the channel axis as spatial and fails); here `channel_axis` is used instead.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_numpy(img):
    """CHW or 1xCHW torch tensor in [0, 1] -> HWC float64, clipped."""
    if hasattr(img, "detach"):
        img = img.detach().float().cpu()
        if img.dim() == 4:
            img = img[0]
        img = img.permute(1, 2, 0).numpy()
    return np.clip(img.astype(np.float64), 0, 1)


def psnr_ssim(restored, clean):
    a, b = to_numpy(restored), to_numpy(clean)
    return (peak_signal_noise_ratio(b, a, data_range=1),
            structural_similarity(b, a, data_range=1, channel_axis=2))
