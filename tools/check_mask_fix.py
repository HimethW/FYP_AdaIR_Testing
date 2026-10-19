"""Check that patches/mask_fix.py builds the FreModule mask correctly.

1. Reference test: the low/high split from FreModuleMaskFix.fft is compared with an
   independent implementation that builds the same soft mask directly on the unshifted
   FFT grid (torch.fft.fftfreq), with no roll / shift / unshift. Even and odd sizes.
2. Released rate_conv weights: are they still at their random init? The original mask
   passes no gradient to rate_conv, so they should be. conv1 (which does get gradient,
   through the high branch) is shown for contrast.
3. Released checkpoint on real images: per AFLB the learned cutoff, the mask area and how
   the non-DC spectral energy splits between the low and high band, original vs fixed.
4. Denoising PSNR / SSIM with the released weights, original vs fixed mask (no retraining).
5. A figure per AFLB: spectrum with the mask outline, the mask, low and high band.

Usage:
    python tools/check_mask_fix.py --ckpt ckpt/adair3d.ckpt \
        --images data/test/denoise/kodak24 --sigma 25 --fig figures/mask_check_adair3d.png
"""
import argparse
import copy
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kstest
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import mask_fix  # noqa: E402

AFLBS = ["fre1", "fre2", "fre3"]


def load_adair(ckpt):
    model = AdaIR(decoder=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")})
    return model


def reference_split(feat, cutoff, tau):
    """Low/high split built on the unshifted FFT grid, independent of FreModule.shift."""
    h, w = feat.shape[-2:]
    u = torch.fft.fftfreq(h, device=feat.device) * 2  # fraction of Nyquist
    v = torch.fft.fftfreq(w, device=feat.device) * 2
    mask = (torch.sigmoid((cutoff[:, 0, None] - u.abs()) / tau)[:, :, None]
            * torch.sigmoid((cutoff[:, 1, None] - v.abs()) / tau)[:, None, :])[:, None]
    spec = torch.fft.fft2(feat, norm="forward", dim=(-2, -1))
    low = torch.fft.ifft2(spec * mask, norm="forward", dim=(-2, -1)).abs()
    high = torch.fft.ifft2(spec * (1 - mask), norm="forward", dim=(-2, -1)).abs()
    return high, low


@torch.no_grad()
def check_reference(model, device):
    print("1) fixed mask vs independent reference (unshifted FFT grid), max abs diff")
    fre = model.fre3
    for h, w in [(16, 16), (32, 48), (64, 64), (33, 47), (127, 128)]:
        x = torch.rand(2, 3, h, w, device=device)
        high, low = fre.fft(x)
        ref_high, ref_low = reference_split(fre.conv1(x), fre.last_cutoff, fre.mask_tau)
        print(f"   {h:>3}x{w:<3}  low {(low - ref_low).abs().max().item():.2e}   "
              f"high {(high - ref_high).abs().max().item():.2e}")


def check_init(model):
    print("2) released weights vs PyTorch default init  (U(-b, b), b = 1/sqrt(fan_in);"
          " untouched => max|w|/b <= 1, std/(b/sqrt(3)) ~ 1, KS p not small)")
    for name in AFLBS:
        fre = getattr(model, name)
        for label, wgt in [("rate_conv.0", fre.rate_conv[0].weight), ("rate_conv.2", fre.rate_conv[2].weight),
                           ("conv1      ", fre.conv1.weight)]:
            b = 1 / np.sqrt(wgt[0].numel())
            vals = wgt.detach().flatten().cpu().numpy()
            p = kstest(vals, "uniform", args=(-b, 2 * b)).pvalue
            print(f"   {name}.{label}  n={vals.size:<6} max|w|/b = {np.abs(vals).max() / b:5.2f}   "
                  f"std/(b/sqrt3) = {vals.std() / (b / np.sqrt(3)):5.2f}   KS p vs U(-b,b) = {p:.3g}")


def read_images(folder, limit):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))[:limit]
    return [(os.path.basename(p), np.array(Image.open(p).convert("RGB"))) for p in paths]


def add_noise(clean, sigma, rng):
    # same as utils/dataset_utils.py DenoiseTestDataset._add_gaussian_noise
    return np.clip(clean + rng.standard_normal(clean.shape) * sigma, 0, 255).astype(np.uint8)


def to_tensor(img, device):
    return torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)


def capture_bands(model):
    """Record (conv1 features, high, low) of each AFLB on the next forward pass."""
    store = {}
    for name in AFLBS:
        fre = getattr(model, name)
        real_fft = fre.fft

        def spy(x, n=128, fre=fre, name=name, real_fft=real_fft):
            high, low = real_fft(x)
            store[name] = (fre.conv1(x), high, low)
            return high, low
        fre.fft = spy

        def contrib_hook(mod, inp, out, name=name):
            y = inp[1]
            store[name + "_contrib"] = ((out - y * mod.para2).norm() / (y * mod.para2).norm()).item()
        store.setdefault("_hooks", []).append(fre.register_forward_hook(contrib_hook))
    return store


def release_capture(model, store):
    for hook in store.pop("_hooks", []):
        hook.remove()
    for name in AFLBS:
        getattr(model, name).__dict__.pop("fft", None)


def original_mask(fre, feat, n=128):
    """The released hard mask (net/model.py FreModule.fft), in the centred layout."""
    h, w = feat.shape[-2:]
    threshold = fre.rate_conv(F.adaptive_avg_pool2d(feat, 1)).sigmoid()
    mask = torch.zeros(feat.shape[0], 1, h, w, device=feat.device)
    for i in range(feat.shape[0]):
        h_ = (h // n * threshold[i, 0, :, :]).int()
        w_ = (w // n * threshold[i, 1, :, :]).int()
        mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
    return mask


def nondc_low_share(feat, mask_centred):
    spec = torch.fft.fftshift(torch.fft.fft2(feat, norm="forward", dim=(-2, -1)), dim=(-2, -1))
    energy = spec.abs() ** 2
    h, w = feat.shape[-2:]
    energy[..., h // 2, w // 2] = 0
    return ((energy * mask_centred).sum() / energy.sum()).item()


@torch.no_grad()
def check_images(orig, fixed, images, sigma, device):
    rng = np.random.default_rng(0)
    print(f"3) released weights on {len(images)} images (noise sigma {sigma}), mean per AFLB")
    stats = {m: {n: [] for n in AFLBS} for m in ("orig", "fixed")}
    contrib = {m: {n: [] for n in AFLBS} for m in ("orig", "fixed")}
    psnr = {"noisy": [], "orig": [], "fixed": []}
    ssim = {"noisy": [], "orig": [], "fixed": []}
    kept = None
    for i, (name, clean) in enumerate(images):
        noisy = add_noise(clean.astype(np.float64), sigma, rng)
        x = to_tensor(noisy, device)
        for label, model in [("orig", orig), ("fixed", fixed)]:
            bands = capture_bands(model)
            out = model(x).clamp(0, 1)
            release_capture(model, bands)
            for n in AFLBS:
                contrib[label][n].append(bands.pop(n + "_contrib"))
            restored = out[0].permute(1, 2, 0).cpu().numpy()
            psnr[label].append(peak_signal_noise_ratio(clean / 255.0, restored, data_range=1))
            ssim[label].append(structural_similarity(clean / 255.0, restored, data_range=1, channel_axis=2))
            for n in AFLBS:
                fre = getattr(model, n)
                feat, high, low = bands[n]
                cutoff = getattr(fre, "last_cutoff", None)
                h, w = feat.shape[-2:]
                mask = original_mask(fre, feat) if cutoff is None else mask_fix.soft_rect_mask(cutoff, h, w, fre.mask_tau)
                area = mask.mean().item()
                stats[label][n].append((area, nondc_low_share(feat, mask),
                                        low.abs().mean().item(), high.abs().mean().item(),
                                        None if cutoff is None else cutoff[0].tolist()))
            if i == 0:
                kept = kept or {}
                kept[label] = {n: tuple(t[0].float().cpu() for t in bands[n]) for n in AFLBS}
                kept[label + "_cutoff"] = {n: getattr(getattr(model, n), "last_cutoff", None) for n in AFLBS}
        psnr["noisy"].append(peak_signal_noise_ratio(clean / 255.0, noisy / 255.0, data_range=1))
        ssim["noisy"].append(structural_similarity(clean / 255.0, noisy / 255.0, data_range=1, channel_axis=2))

    for label in ("orig", "fixed"):
        print(f"   {label}:")
        for n in AFLBS:
            s = stats[label][n]
            cut = [c for *_, c in s if c is not None]
            cut_txt = (f"cutoff h {np.mean([c[0] for c in cut]):.3f} w {np.mean([c[1] for c in cut]):.3f}"
                       f" (range {min(min(c) for c in cut):.3f}-{max(max(c) for c in cut):.3f})") if cut else "no cutoff (hard mask)"
            print(f"     {n}: mask area {np.mean([a for a, *_ in s]) * 100:6.3f}% (images with any mask: {sum(a > 0 for a, *_ in s)}/{len(s)})   "
                  f"non-DC energy in low band {np.mean([e for _, e, *_ in s]) * 100:6.3f}%   "
                  f"mean|low| {np.mean([l for _, _, l, *_ in s]):.4f}   mean|high| {np.mean([h for _, _, _, h, _ in s]):.4f}   {cut_txt}")
    print("   size of the AFLB's change to the decoder feature, |out*para1| / |y*para2| (mean):")
    for label in ("orig", "fixed"):
        print(f"     {label:5} " + "   ".join(f"{n} {np.mean(contrib[label][n]):.4f}" for n in AFLBS))
    print("   para1 (learned weight of the frequency branch), mean |.|: " + "   ".join(
        f"{n} {getattr(orig, n).para1.abs().mean().item():.4f}" for n in AFLBS) + "   (para2: " + "   ".join(
        f"{getattr(orig, n).para2.abs().mean().item():.3f}" for n in AFLBS) + ")")
    print(f"4) denoising, released weights, no retraining (mean over {len(images)} images)")
    for label in ("noisy", "orig", "fixed"):
        print(f"   {label:6} PSNR {np.mean(psnr[label]):.2f}   SSIM {np.mean(ssim[label]):.4f}")
    diffs = np.array(psnr["fixed"]) - np.array(psnr["orig"])
    print(f"   fixed - orig PSNR: mean {diffs.mean():+.3f}, min {diffs.min():+.3f}, max {diffs.max():+.3f},"
          f" improved on {(diffs > 0).sum()}/{len(diffs)}")
    return kept


def save_figure(kept, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["spectrum + fixed mask (0.5 contour)", "fixed mask", "orig low", "fixed low", "orig high", "fixed high"]
    fig, axes = plt.subplots(3, 6, figsize=(19, 9.5))
    for r, n in enumerate(AFLBS):
        feat, _, _ = kept["fixed"][n]
        h, w = feat.shape[-2:]
        spec = torch.fft.fftshift(torch.fft.fft2(feat, norm="forward").abs().mean(0))
        cutoff = kept["fixed_cutoff"][n].cpu()
        mask = mask_fix.soft_rect_mask(cutoff, h, w)[0, 0]
        panels = [torch.log(spec + 1e-6), mask,
                  kept["orig"][n][2].mean(0), kept["fixed"][n][2].mean(0),
                  kept["orig"][n][1].mean(0), kept["fixed"][n][1].mean(0)]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img.numpy(), cmap="magma" if c < 2 else "gray")
            if c == 0:
                ax.contour(mask.numpy(), levels=[0.5], colors="cyan", linewidths=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{n}  {h}x{w}\ncutoff {cutoff[0, 0]:.2f} / {cutoff[0, 1]:.2f}", fontsize=10)
            if c in (2, 3, 4, 5):
                ax.set_xlabel(f"mean |x| = {img.abs().mean():.3f}", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=80)
    print(f"5) figure saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/adair3d.ckpt")
    p.add_argument("--images", default="data/test/denoise/kodak24")
    p.add_argument("--sigma", type=float, default=25)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--fig", default="figures/mask_check_adair3d.png")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig = load_adair(args.ckpt).to(device).eval()
    fixed = mask_fix.apply(copy.deepcopy(orig))

    check_reference(fixed, device)
    check_init(orig)
    images = read_images(args.images, args.limit)
    kept = check_images(orig, fixed, images, args.sigma, device)
    save_figure(kept, args.fig, f"{os.path.basename(args.ckpt)}: {images[0][0]} + noise sigma {args.sigma:g}, "
                                f"released weights, original vs fixed mask (no retraining)")


if __name__ == "__main__":
    main()
