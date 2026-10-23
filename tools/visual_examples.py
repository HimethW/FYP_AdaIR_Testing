"""Side-by-side examples for slides: degraded input | model A | model B | ground truth, one row per test set.

Rows are not cherry-picked: for each set it takes the image whose PSNR gain (B - A, from two eval_fyp.py --out JSONs
with per-image results) is closest to that set's median gain. Runs on the CPU by default (true fp32).
  python tools/visual_examples.py --out results/visual_examples.png \
      --a-ckpt ckpt/adair3d.ckpt --a-variant original --a-json runs/eval_released/adair3d_main.json --a-label "Released AdaIR" \
      --b-ckpt runs/ours_ft/epochs/step_0099730.pt --b-ema --b-json runs/eval_ours_ft/epoch20_main.json --b-label "Ours"
"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.evaluate import load_sample, run_model, size, test_sets  # noqa: E402
from fyp.metrics import psnr_ssim  # noqa: E402
from fyp.models import build_model, load_weights  # noqa: E402

NAMES = {"haze_rain_cdd11": "Haze + rain (CDD-11)", "haze_rain": "Haze + rain", "rain_noise25": "Rain + noise σ25",
         "haze_noise25": "Haze + noise σ25", "rain100l": "Rain (Rain100L)", "sots": "Haze (SOTS)"}


def load(ckpt, variant, ema):
    variant = variant or torch.load(ckpt, map_location="cpu", weights_only=False).get("variant")
    model = build_model(variant)
    load_weights(model, ckpt, use_ema=ema)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("a", "b"):
        ap.add_argument(f"--{k}-ckpt", required=True)
        ap.add_argument(f"--{k}-variant", default=None)
        ap.add_argument(f"--{k}-ema", action="store_true")
        ap.add_argument(f"--{k}-json", required=True)
        ap.add_argument(f"--{k}-label", required=True)
    ap.add_argument("--sets", nargs="+", default=["haze_rain_cdd11", "haze_rain", "rain_noise25", "haze_noise25"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ra = json.load(open(args.a_json))["results"]
    rb = json.load(open(args.b_json))["results"]
    ma = load(args.a_ckpt, args.a_variant, args.a_ema)
    mb = load(args.b_ckpt, args.b_variant, args.b_ema)
    sets = test_sets(args.data)

    fig, axes = plt.subplots(len(args.sets), 4, figsize=(16, 3.4 * len(args.sets)), dpi=150)
    axes = np.atleast_2d(axes)
    for row, name in enumerate(args.sets):
        pa, pb = ra[name]["per_image"], rb[name]["per_image"]
        gain = {k: pb[k][0] - pa[k][0] for k in pa}
        med = float(np.median(list(gain.values())))
        pick = min(gain, key=lambda k: abs(gain[k] - med))
        spec = sets[name]
        idx = next(i for i in range(size(spec)) if load_sample(spec, i, "adair")[0] == pick)
        _, lq, gt = load_sample(spec, idx, "adair")
        x = torch.from_numpy(lq).permute(2, 0, 1)[None].float().div(255)
        t = torch.from_numpy(gt).permute(2, 0, 1).float().div(255)
        with torch.no_grad():
            ya, yb = run_model(ma, x).clamp(0, 1), run_model(mb, x).clamp(0, 1)
        sa, sb, sx = psnr_ssim(ya, t), psnr_ssim(yb, t), psnr_ssim(x, t)
        print(f"{name}: {pick} (median gain {med:+.2f} dB) input {sx[0]:.2f}  {args.a_label} {sa[0]:.2f} "
              f"(eval {pa[pick][0]:.2f})  {args.b_label} {sb[0]:.2f} (eval {pb[pick][0]:.2f})", flush=True)
        panels = [(lq / 255.0, f"Input  {sx[0]:.2f} dB"), (ya[0].permute(1, 2, 0).numpy(), f"{args.a_label}  {sa[0]:.2f} dB"),
                  (yb[0].permute(1, 2, 0).numpy(), f"{args.b_label}  {sb[0]:.2f} dB"), (gt / 255.0, "Ground truth")]
        for col, (img, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(NAMES.get(name, name), fontsize=12)
    fig.suptitle(f"{args.a_label} vs {args.b_label}: one typical image per set (the one closest to the set's median "
                 f"PSNR gain); PSNR vs ground truth", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
