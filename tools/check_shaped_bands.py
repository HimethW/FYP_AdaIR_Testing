"""Checks for patches/shaped_bands.py (4 shaped bands + band self-attention).

1. Masks partition the spectrum: values in [0, 1], sum to 1 (even and odd sizes).
2. Band areas at the starting shapes are the same at every map size (resolution-independent).
3. Bands are real and add back up: masks are point-symmetric, so IFFT(M_b * F) is real,
   and sum_b IFFT(M_b * F) == IFFT(F) == P.
4. BSA starts as identity (zero-init output projection).
5. Every new parameter gets a gradient (after one optimiser step, since zero-inits block
   some gradients at step 0 by design).
6. Parameters, speed and memory: original AdaIR vs shaped bands.
7. Real images (Kodak24, clean and noisy): 50%-energy radius and band areas per AFLB.

Usage: python tools/check_shaped_bands.py [--images data/test/denoise/kodak24]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import shaped_bands as sb  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def params_at_init(r50, b=1):
    raw = sb.init_raw().to(dev)[None].expand(b, -1)
    return sb.decode_params(raw, torch.full((b,), r50, device=dev))


def check_partition():
    print("1) partition: min, max, max |sum - 1|")
    torch.manual_seed(0)
    for h, w in [(16, 16), (33, 47), (64, 96), (256, 384)]:
        raw = sb.init_raw().to(dev)[None] + 0.5 * torch.randn(4, 8, device=dev)  # perturbed shapes
        m = sb.band_masks(sb.decode_params(raw, torch.rand(4, device=dev) * 0.4), h, w)
        print(f"   {h:>3}x{w:<3}  min {m.min().item():.2e}  max {m.max().item():.4f}  "
              f"|sum-1| {(m.sum(1) - 1).abs().max().item():.2e}")


def check_resolution():
    print("2) band areas at the starting shapes (r50 = 0.10), dot / star / lobe / rest")
    for s in [16, 32, 64, 128, 256, 512]:
        a = sb.band_masks(params_at_init(0.10), s, s).mean((2, 3))[0]
        print(f"   {s:>3}x{s:<3} " + "  ".join(f"{n} {x * 100:5.1f}%" for n, x in zip(sb.BANDS, a.tolist())))


def check_real_and_sum():
    print("3) bands real and complete: max |imag| / max |real|,  max |sum of bands - P|")
    torch.manual_seed(0)
    for h, w in [(16, 16), (33, 47), (64, 96)]:
        p = torch.randn(2, 8, h, w, device=dev)
        spec = torch.roll(torch.fft.fft2(p, norm='forward'), (h // 2, w // 2), (2, 3))
        raw = sb.init_raw().to(dev)[None] + 0.5 * torch.randn(2, 8, device=dev)
        m = sb.band_masks(sb.decode_params(raw, torch.rand(2, device=dev) * 0.4), h, w)
        banded = torch.roll(spec[:, None] * m[:, :, None], (-(h // 2), -(w // 2)), (3, 4))
        bands = torch.fft.ifft2(banded, norm='forward')
        print(f"   {h:>3}x{w:<3}  imag/real {bands.imag.abs().max().item() / bands.real.abs().max().item():.2e}   "
              f"sum - P {(bands.sum(1).real - p).abs().max().item():.2e}")


def check_bsa_identity():
    bsa = sb.BandSelfAttention(96).to(dev)
    xs = [torch.randn(2, 96, 16, 16, device=dev) for _ in range(4)]
    with torch.no_grad():
        out = bsa(xs)
    print("4) BSA at init == identity: max diff", max((o - x).abs().max().item() for o, x in zip(out, xs)))


def check_gradients():
    print("5) gradients after one Adam step (para1 = 0.1 so the AFLB branch is on)")
    torch.manual_seed(0)
    model = sb.apply(AdaIR(decoder=True)).to(dev).train()
    with torch.no_grad():
        for n in sb.AFLBS:
            getattr(model, n).para1.fill_(0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, t = torch.rand(2, 3, 128, 128, device=dev), torch.rand(2, 3, 128, 128, device=dev)
    for _ in range(2):
        opt.zero_grad()
        (model(x) - t).abs().mean().backward()
        if _ == 0:
            opt.step()
    for n in sb.AFLBS:
        fre = getattr(model, n)
        missing = [k for k, prm in fre.named_parameters() if prm.grad is None or prm.grad.abs().sum() == 0]
        base = fre.sgb.base.grad.abs()
        print(f"   {n}: params without gradient: {missing or 'none'};  "
              f"sgb.base grad per shape param: " + " ".join(f"{g:.1e}" for g in base.tolist()))


def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def bench(model, x, train):
    model.train(train)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(6):
        t0 = time.time()
        if train:
            model.zero_grad(); model(x).mean().backward()
        else:
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - t0)
    return np.mean(times) * 1000, torch.cuda.max_memory_allocated() / 2 ** 30


def check_cost():
    print("6) cost: params, train step (4x3x128x128 fwd+bwd), inference (1x3x512x768)")
    for label, make in [("original", lambda: AdaIR(decoder=True)), ("shaped", lambda: sb.apply(AdaIR(decoder=True)))]:
        model = make().to(dev)
        tr = bench(model, torch.rand(4, 3, 128, 128, device=dev), True)
        inf = bench(model, torch.rand(1, 3, 512, 768, device=dev), False)
        print(f"   {label:8} {count(model):6.2f}M params   train {tr[0]:6.1f} ms  {tr[1]:5.2f} GB   "
              f"infer {inf[0]:6.1f} ms  {inf[1]:5.2f} GB")
        del model; torch.cuda.empty_cache()


@torch.no_grad()
def check_real(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        print("7) skipped (no images)"); return
    print(f"7) {len(paths)} real images, untrained shaped model: mean r50 and band areas per AFLB")
    torch.manual_seed(0)
    model = sb.apply(AdaIR(decoder=True)).to(dev).eval()
    rng = np.random.default_rng(0)
    for sigma in [0, 15, 25, 50]:
        acc = {n: ([], []) for n in sb.AFLBS}
        for p in paths:
            img = np.array(Image.open(p).convert("RGB")).astype(np.float64)
            if sigma:
                img = np.clip(img + rng.standard_normal(img.shape) * sigma, 0, 255)
            model(torch.from_numpy(img.astype(np.uint8)).permute(2, 0, 1)[None].float().div(255).to(dev))
            for n in sb.AFLBS:
                fre = getattr(model, n)
                acc[n][0].append(fre.last_r50.item()); acc[n][1].append(fre.last_band_area[0].cpu().numpy())
        print(f"   sigma {sigma:>2}: " + "   ".join(
            f"{n} r50 {np.mean(acc[n][0]):.3f} [" + "/".join(f"{a * 100:.1f}" for a in np.mean(acc[n][1], 0)) + "]%"
            for n in sb.AFLBS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/test/denoise/kodak24")
    args = ap.parse_args()
    check_partition()
    check_resolution()
    check_real_and_sum()
    check_bsa_identity()
    check_gradients()
    check_cost()
    check_real(args.images)
