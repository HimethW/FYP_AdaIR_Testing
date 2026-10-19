"""Mask shapes of a trained shaped-band model, per AFLB block and per degradation type.

For each degradation (clean, noise sigma 15/25/50, rain, haze, haze+noise, rain+noise, haze+rain ours, haze+rain
CDD-11) it runs N test images (centre 256x256 crops) through the model and records, per AFLB (fre1/fre2/fre3):
the mean log spectrum of the AFLB's conv features, the decoded shape params (SGB output, per image) and the band
areas. Writes to --out:
  masks_grid.png              rows = AFLB, columns = degradation: spectrum + dot / star / lobe outlines + areas
  masks_<degradation>.png     per degradation: outlines per AFLB, and the 4-band partition filled in
  shape_params.csv / .json    per degradation x AFLB: mean and std of each shape param and band area
  README.md                   which checkpoint, weights, images and settings

Runs on the CPU by default (--device cpu) so it never takes GPU memory from a training run.

  python tools/plot_mask_shapes.py --ckpt runs/ours_ft/last.pt --ema --out /home/minura/Himeth/results/mask_shapes
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.evaluate import load_sample, test_sets  # noqa: E402
from fyp.models import build_model, load_weights  # noqa: E402
from patches import shaped_bands as sb  # noqa: E402
from tools.plot_bands import COLORS, outlines  # noqa: E402

# (label, test set, noisy?) - "clean" uses the clean BSD68 images of the sigma-15 set
DEGRADATIONS = [("clean", "bsd68_s15", False), ("noise s15", "bsd68_s15", True), ("noise s25", "bsd68_s25", True),
                ("noise s50", "bsd68_s50", True), ("rain", "rain100l", True), ("haze", "sots_unseen", True),
                ("haze+noise s25", "haze_noise25_unseen", True), ("rain+noise s25", "rain_noise25", True),
                ("haze+rain (ours)", "haze_rain_unseen", True), ("haze+rain (CDD-11)", "haze_rain_cdd11", True)]
AFLBS = ("fre1", "fre2", "fre3")
PARAMS = ("dot_r", "star_theta", "star_p", "star_size", "star_aspect", "lobe_vertex", "lobe_curv", "lobe_theta")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def centre_crop(img, size):
    h, w = img.shape[:2]
    s = min(size, h, w) // 16 * 16
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


@torch.no_grad()
def collect(model, sets, n, crop, device):
    """{degradation: {aflb: {"spec": mean log spectrum, "params": [dicts], "area": [B4]}}}"""
    store = {}

    def spy_factory(fre, name):
        real = fre.split

        def spy(x):
            bands = real(x)
            p = fre.conv1(x.float())
            h, w = p.shape[-2:]
            spec = torch.roll(torch.fft.fft2(p, norm="forward"), (h // 2, w // 2), (2, 3)).abs().mean(1)[0]
            store[name] = spec.cpu()
            return bands
        return spy

    for name in AFLBS:
        fre = getattr(model, name)
        fre.split = spy_factory(fre, name)
    out = {}
    for label, set_name, degraded in DEGRADATIONS:
        spec_sum = {a: 0 for a in AFLBS}
        params = {a: [] for a in AFLBS}
        areas = {a: [] for a in AFLBS}
        spec = sets[set_name]
        k = min(n, len(spec.get("pairs") or spec.get("clean")))
        for i in range(k):
            _, lq, gt = load_sample(spec, i, "adair")
            img = centre_crop(lq if degraded else gt, crop)
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)
            model(x)
            for a in AFLBS:
                fre = getattr(model, a)
                spec_sum[a] = spec_sum[a] + torch.log(store[a] + 1e-8)
                params[a].append({p: fre.last_params[p][0].item() for p in PARAMS})
                areas[a].append(fre.last_band_area[0].cpu().numpy())
        out[label] = {a: {"spec": spec_sum[a] / k, "params": params[a], "area": np.array(areas[a]), "n": k} for a in AFLBS}
        print(f"  {label:20} {k} images", flush=True)
    for name in AFLBS:
        getattr(model, name).__dict__.pop("split", None)
    return out


def mean_params(plist):
    return {p: torch.tensor([float(np.mean([d[p] for d in plist]))]) for p in PARAMS}


def partition(params, h, w):
    m = sb.band_masks(params, h, w)[0]  # (4, h, w): dot, star, lobe, rest
    return m.argmax(0).numpy()


def draw_spectrum(ax, spec, params):
    h, w = spec.shape
    ax.imshow(spec.numpy(), cmap="gray")
    for name, m in outlines(params, h, w).items():
        ax.contour(m.numpy(), levels=[0.5], colors=COLORS[name], linewidths=1.4)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ema", action="store_true", help="use the EMA weights stored in the checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20, help="images per degradation")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    ckpt, _, _ = load_weights(model := build_model("shaped"), args.ckpt, use_ema=args.ema)
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    targs = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    per_epoch = 159567 if targs.get("tasks") == "aio3_mix" else None
    epochs = step * targs.get("batch", 1) * targs.get("accum", 1) / per_epoch if (step and per_epoch) else None
    model = model.to(args.device).eval()
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.ckpt} step {step} ({'EMA' if args.ema else 'raw'} weights), {args.n} images per degradation, "
          f"{args.crop} crops, {args.device}", flush=True)
    res = collect(model, test_sets(args.data), args.n, args.crop, args.device)

    labels = [d[0] for d in DEGRADATIONS]
    tag = f"step {step}" + (f" = {epochs:.2f} epochs" if epochs else "") + (" (EMA weights)" if args.ema else " (raw weights)")

    # 1. grid
    fig, axes = plt.subplots(len(AFLBS), len(labels), figsize=(2.3 * len(labels), 2.65 * len(AFLBS)), squeeze=False)
    for c, lab in enumerate(labels):
        for r, a in enumerate(AFLBS):
            d = res[lab][a]
            ax = axes[r, c]
            draw_spectrum(ax, d["spec"], mean_params(d["params"]))
            ar = d["area"].mean(0) * 100
            ax.set_xlabel(" / ".join(f"{v:.0f}" for v in ar) + " %", fontsize=8)
            if r == 0:
                ax.set_title(lab, fontsize=9)
            if c == 0:
                h, w = d["spec"].shape
                ax.set_ylabel(f"{a} ({h}x{w})", fontsize=9)
    fig.suptitle(f"Learned mask shapes per AFLB and degradation - {os.path.basename(os.path.dirname(args.ckpt))} {tag}\n"
                 f"mean log spectrum of the AFLB features over {args.n} test images; outlines at 0.5: blue = dot, "
                 f"orange = star, green = lobe; below: band areas dot / star / lobe / rest", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "masks_grid.png"), dpi=90)
    plt.close(fig)

    # 2. per degradation: outlines + filled partition
    cmap = ListedColormap([COLORS["dot"], COLORS["star"], COLORS["lobe"], "#d9d9d6"])
    for lab in labels:
        fig, axes = plt.subplots(2, len(AFLBS), figsize=(3.6 * len(AFLBS), 7.2), squeeze=False)
        for c, a in enumerate(AFLBS):
            d = res[lab][a]
            mp = mean_params(d["params"])
            h, w = d["spec"].shape
            draw_spectrum(axes[0, c], d["spec"], mp)
            axes[0, c].set_title(f"{a} ({h}x{w})", fontsize=10)
            axes[1, c].imshow(partition(mp, h, w), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            axes[1, c].set_xticks([]), axes[1, c].set_yticks([])
            ar, sd = d["area"].mean(0) * 100, d["area"].std(0) * 100
            axes[1, c].set_xlabel("\n".join(f"{b}: {m:.1f}% (sd {s:.1f})" for b, m, s in zip(sb.BANDS, ar, sd)), fontsize=8.5)
        axes[0, 0].set_ylabel("spectrum + outlines", fontsize=9)
        axes[1, 0].set_ylabel("4-band partition\nblue dot, orange star,\ngreen lobe, grey rest", fontsize=9)
        fig.suptitle(f"{lab}: mask shapes per AFLB - {tag}, mean over {d['n']} images", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"masks_{slug(lab)}.png"), dpi=90)
        plt.close(fig)

    # 3. numbers
    rows = []
    start = dict(sb.INIT_SHAPE)
    for lab in labels:
        for a in AFLBS:
            d = res[lab][a]
            row = {"degradation": lab, "aflb": a, "n_images": d["n"]}
            for p in PARAMS:
                vals = np.array([x[p] for x in d["params"]])
                if p.endswith("theta"):
                    vals = np.degrees(vals)
                row[f"{p}_mean"], row[f"{p}_std"] = round(float(vals.mean()), 5), round(float(vals.std()), 5)
            for b, m, s in zip(sb.BANDS, d["area"].mean(0), d["area"].std(0)):
                row[f"area_{b}_mean"], row[f"area_{b}_std"] = round(float(m), 5), round(float(s), 5)
            rows.append(row)
    with open(os.path.join(args.out, "shape_params.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "shape_params.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "step": step, "epochs": epochs, "ema": args.ema, "n_images": args.n,
                   "crop": args.crop, "starting_shapes": start, "bounds": {k: list(v) for k, v in sb.BOUNDS.items()},
                   "rows": rows}, f, indent=1)

    # 4. README
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(f"""# Mask shapes after {f'{epochs:.2f} epochs' if epochs else f'step {step}'}

- Checkpoint: `{args.ckpt}`, {tag}. Model: shaped-band AdaIR (bounded shapes).
- Images: first {args.n} images of each test set, centre {args.crop}x{args.crop} crop (AFLB feature maps:
  fre1 {args.crop // 8}px, fre2 {args.crop // 4}px, fre3 {args.crop // 2}px). Test sets: """
                + ", ".join(f"{lab} = `{s}`{'' if dg else ' (clean targets)'}" for lab, s, dg in DEGRADATIONS) + f""".
- Shapes are predicted per image by each AFLB's shape generator (SGB); figures use the mean parameters
  over the images of a degradation, on the mean log spectrum of that AFLB's conv features.

Files:
- `masks_grid.png`: rows = AFLB (fre1 deepest / lowest resolution, fre3 highest), columns = degradation.
  Outlines at mask = 0.5: blue = dot (disc at r50 x learned scale), orange = star (rotated superellipse),
  green = lobe (side lobes). Below each panel: band areas dot / star / lobe / rest in %.
- `masks_<degradation>.png`: the same per degradation, plus the 4-band partition filled in
  (blue dot, orange star, green lobe, grey rest) with mean and sd of each band's area.
- `shape_params.csv` / `.json`: mean and sd over the images of every shape param (angles in degrees, dot_r in
  fractions of Nyquist) and band area, per degradation and AFLB.

Starting shapes (before training) for comparison: {', '.join(f'{k} {v}' for k, v in start.items())}
(dot_scale multiplies the per-image 50%-energy radius r50). Bounds: {', '.join(f'{k} {list(v)[:2]}' for k, v in sb.BOUNDS.items())}.

Made by `tools/plot_mask_shapes.py` on {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f} s, {args.device}).
""")
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
