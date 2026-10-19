"""Checks for patches/losses.py (pixel L1 + 0.1 FFT L1).

1. fft_l1 equals the SFNet training-code computation exactly.
2. Size of the two terms on real errors (Kodak24 crops): noisy input vs clean, released AdaIR
   output vs clean (a realistic residual), and a hazy input (low-frequency error); at 128 and
   256 crops. Reports l_fft / l_pix and the FFT share of the total loss at weight 0.1.
3. Works under bf16 autocast, gradients finite.

Usage: python tools/check_losses.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import amp  # noqa: E402
from patches.losses import PixelFFTLoss, fft_l1  # noqa: E402

dev = "cuda"


def sfnet_fft(pred, label):
    # verbatim logic of SFNet/Image_dehazing/train.py (one scale)
    criterion = nn.L1Loss()
    label_fft = torch.fft.fft2(label, dim=(-2, -1))
    label_fft = torch.stack((label_fft.real, label_fft.imag), -1)
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
    return criterion(pred_fft, label_fft)


def crops(size, n_per_image=2):
    rng = np.random.default_rng(0)
    out = []
    for p in sorted(glob.glob("data/test/denoise/kodak24/*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        for _ in range(n_per_image):
            y = rng.integers(0, img.shape[0] - size)
            x = rng.integers(0, img.shape[1] - size)
            out.append(img[y:y + size, x:x + size])
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float().div(255).to(dev)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    a, b = torch.rand(4, 3, 128, 128, device=dev), torch.rand(4, 3, 128, 128, device=dev)
    print(f"1) fft_l1 vs SFNet code: |diff| = {(fft_l1(a, b) - sfnet_fft(a, b)).abs().item():.2e}")

    ortho = PixelFFTLoss(0.1, norm="ortho")
    model = amp.fp32_frequency(AdaIR(decoder=True))
    sd = torch.load("ckpt/adair3d.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[4:]: v for k, v in sd.items()})
    model = model.to(dev).eval()
    loss = PixelFFTLoss(0.1)
    print("2) term sizes on Kodak24 crops (mean over crops)")
    print("   error type                     crop   l_pix    l_fft    l_fft/l_pix  FFT share (w=0.1)   ortho: l_fft/l_pix, share")
    for size in (128, 256):
        clean = crops(size)
        noisy = (clean + torch.randn_like(clean) * 25 / 255).clamp(0, 1)
        t, A = 0.5, 0.9
        hazy = clean * t + A * (1 - t)
        restored = torch.cat([model(noisy[i:i + 8]).clamp(0, 1) for i in range(0, len(noisy), 8)])
        for name, pred in [("noisy sigma 25 input", noisy), ("AdaIR output on sigma 25", restored),
                           ("hazy input (t 0.5, A 0.9)", hazy)]:
            total, parts = loss(pred, clean)
            lp, lf = parts["l_pix"].item(), parts["l_fft"].item()
            t_o, parts_o = ortho(pred, clean)
            lo = parts_o["l_fft"].item()
            print(f"   {name:30} {size:>4}  {lp:.4f}  {lf:8.4f}  {lf / lp:9.1f}      {0.1 * lf / total.item() * 100:5.1f}%"
                  f"           {lo / lp:5.2f}, {0.1 * lo / t_o.item() * 100:4.1f}%")

    print("3) bf16 autocast + gradients")
    with torch.enable_grad():
        p = torch.rand(2, 3, 128, 128, device=dev, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, _ = loss(p.to(torch.bfloat16), a[:2])
        total.backward()
        print(f"   loss dtype {total.dtype}, grad finite {bool(torch.isfinite(p.grad).all())}")


if __name__ == "__main__":
    main()
