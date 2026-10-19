"""Plot the shaped bands of a shaped_bands AdaIR model on real images.

Figure 1 (--fig-masks): per AFLB (rows) and condition (columns), the mean log spectrum of the
AFLB's conv features with the 0.5 outlines of dot (blue), star (orange) and lobe (green),
and the mean band areas dot / star / lobe / rest.
Figure 2 (--fig-bands): what each band passes for one image, per AFLB.

Without --ckpt the model is untrained, so the shapes are the starting shapes (the dot still
adapts per image through its 50%-energy radius).

Usage:
    python tools/plot_bands.py --images data/test/denoise/kodak24 --sigmas 0 15 25 50 \
        --fig-masks figures/bands_start_masks.png --fig-bands figures/bands_start_kodim.png
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.model import AdaIR  # noqa: E402
from patches import shaped_bands as sb  # noqa: E402

COLORS = {"dot": "#3b82f6", "star": "#f97316", "lobe": "#22c55e"}


def load_model(ckpt, device):
    model = sb.apply(AdaIR(decoder=True))
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        model.load_state_dict({k[4:] if k.startswith("net.") else k: v for k, v in sd.items()})
    return model.to(device).eval()


def degrade(img, sigma, rng):
    if sigma:
        img = np.clip(img + rng.standard_normal(img.shape) * sigma, 0, 255)
    return img.astype(np.uint8)


@torch.no_grad()
def collect(model, paths, sigma, device, keep_first=False):
    """Mean centred log spectrum, mean params and band areas per AFLB; optionally bands of image 0."""
    rng = np.random.default_rng(0)
    acc = {n: {"spec": 0, "params": [], "area": []} for n in sb.AFLBS}
    first = {}
    store = {}
    hooks = []
    for n in sb.AFLBS:
        fre = getattr(model, n)
        real_split = fre.split

        def spy(x, fre=fre, n=n, real_split=real_split):
            bands = real_split(x)
            p = fre.conv1(x)
            h, w = p.shape[-2:]
            spec = torch.roll(torch.fft.fft2(p, norm="forward"), (h // 2, w // 2), (2, 3)).abs().mean(1)[0]
            store[n] = (spec.cpu(), [b[0].mean(0).cpu() for b in bands])
            return bands
        fre.split = spy
        hooks.append(fre)
    for i, p in enumerate(paths):
        img = np.array(Image.open(p).convert("RGB")).astype(np.float64)
        hh, ww = img.shape[:2]
        s = min(hh, ww, 512)  # centre square crop so spectra can be averaged across images
        img = img[(hh - s) // 2:(hh - s) // 2 + s, (ww - s) // 2:(ww - s) // 2 + s]
        x = torch.from_numpy(degrade(img, sigma, rng)).permute(2, 0, 1)[None].float().div(255).to(device)
        model(x)
        for n in sb.AFLBS:
            fre = getattr(model, n)
            acc[n]["spec"] = acc[n]["spec"] + torch.log(store[n][0] + 1e-8)
            acc[n]["params"].append({k: v[0].item() for k, v in fre.last_params.items()})
            acc[n]["area"].append(fre.last_band_area[0].cpu().numpy())
            if keep_first and i == 0:
                first[n] = (x[0].cpu(), store[n][1], fre.last_band_area[0].cpu().numpy())
    for fre in hooks:
        fre.__dict__.pop("split", None)
    out = {}
    for n in sb.AFLBS:
        params = {k: torch.tensor([np.mean([d[k] for d in acc[n]["params"]])]) for k in acc[n]["params"][0]}
        out[n] = (acc[n]["spec"] / len(paths), params, np.mean(acc[n]["area"], 0))
    return out, first


def outlines(params, h, w):
    """Unpartitioned dot / star / lobe masks (for the 0.5 outlines)."""
    ref = params["dot_r"]
    u, v = sb.centred_grid(h, w, ref.device, ref.dtype)
    r = torch.sqrt(u ** 2 + v ** 2)
    out = {"dot": torch.sigmoid((params["dot_r"] - r) / 0.05)}
    th, size, aspect, pw = (params[k] for k in ("star_theta", "star_size", "star_aspect", "star_p"))
    x = torch.cos(th) * v + torch.sin(th) * u
    y = -torch.sin(th) * v + torch.cos(th) * u
    g = (((x.abs() + 1e-4) / (size * aspect)) ** pw + ((y.abs() + 1e-4) / (size / aspect)) ** pw) ** (1 / pw)
    out["star"] = torch.sigmoid(size * (1 - g) / 0.05)
    th, vertex, curv = (params[k] for k in ("lobe_theta", "lobe_vertex", "lobe_curv"))
    ul = torch.cos(th) * u - torch.sin(th) * v
    vl = torch.sin(th) * u + torch.cos(th) * v
    out["lobe"] = torch.sigmoid((vl.abs() - vertex - curv * ul ** 2) / 0.05)
    return out


def fig_masks(results, sigmas, path, title):
    fig, axes = plt.subplots(len(sb.AFLBS), len(sigmas), figsize=(3.2 * len(sigmas), 3.4 * len(sb.AFLBS)),
                             squeeze=False)
    for c, sigma in enumerate(sigmas):
        for r, n in enumerate(sb.AFLBS):
            spec, params, area = results[sigma][n]
            h, w = spec.shape
            ax = axes[r, c]
            ax.imshow(spec.numpy(), cmap="gray")
            for name, m in outlines(params, h, w).items():
                ax.contour(m.numpy(), levels=[0.5], colors=COLORS[name], linewidths=1.6)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel("dot/star/lobe/rest\n" + " / ".join(f"{a * 100:.1f}%" for a in area), fontsize=8)
            if r == 0:
                ax.set_title("clean" if sigma == 0 else f"noise sigma {sigma:g}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{n} ({h}x{w})", fontsize=10)
    fig.suptitle(title + "\nblue = dot, orange = star, green = lobe, rest = outside all three", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_bands(first, path, title):
    fig, axes = plt.subplots(len(sb.AFLBS), 5, figsize=(16, 2.9 * len(sb.AFLBS)), squeeze=False)
    for r, n in enumerate(sb.AFLBS):
        img, bands, area = first[n]
        axes[r, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[r, 0].set_ylabel(n, fontsize=10)
        for c, (name, band) in enumerate(zip(sb.BANDS, bands), start=1):
            b = band.numpy()
            axes[r, c].imshow(b, cmap="gray", vmin=0, vmax=np.percentile(b, 99.5))
            axes[r, c].set_xlabel(f"{name} band, {area[c - 1] * 100:.1f}% of spectrum", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_title("input", fontsize=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="state dict of a shaped_bands AdaIR; omit for starting shapes")
    ap.add_argument("--images", default="data/test/denoise/kodak24")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 15, 25, 50])
    ap.add_argument("--fig-masks", default="figures/bands_start_masks.png")
    ap.add_argument("--fig-bands", default="figures/bands_start_kodim.png")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    paths = sorted(glob.glob(os.path.join(args.images, "*.png")))
    label = os.path.basename(args.ckpt) if args.ckpt else "untrained (starting shapes)"
    results, first = {}, None
    for sigma in args.sigmas:
        results[sigma], f = collect(model, paths, sigma, device, keep_first=(sigma == args.sigmas[-1]))
        first = f or first
    os.makedirs(os.path.dirname(args.fig_masks) or ".", exist_ok=True)
    fig_masks(results, args.sigmas, args.fig_masks,
              f"{label}: mean log spectrum of each AFLB's features over {len(paths)} images (centre 512x512 crops)")
    fig_bands(first, args.fig_bands,
              f"{label}: what each band passes, {os.path.basename(paths[0])} + noise sigma {args.sigmas[-1]:g} "
              "(mean |band| over channels)")
    print("saved", args.fig_masks, "and", args.fig_bands)


if __name__ == "__main__":
    main()
