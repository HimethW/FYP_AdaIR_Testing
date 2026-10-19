"""Show how far the bounded mask shapes can move: starting shapes, each shape param at its
lower / upper bound (one at a time), and random draws inside the bounds.

Usage: python tools/plot_bounds.py --out figures/bands_bounds.png
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patches import shaped_bands as sb  # noqa: E402
from tools.plot_bands import COLORS, outlines  # noqa: E402

N = 128
R50 = 0.10


def params_from(values):
    p = {k: torch.tensor([float(v)]) for k, v in values.items()}
    p["dot_r"] = R50 * p.pop("dot_scale")
    return p


def draw(ax, values, title):
    p = params_from(values)
    areas = sb.band_masks(p, N, N).mean((2, 3))[0]
    ax.imshow(torch.zeros(N, N).numpy(), cmap="gray", vmin=0, vmax=1)
    for name, m in outlines(p, N, N).items():
        ax.contour(m.numpy(), levels=[0.5], colors=COLORS[name], linewidths=1.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(" / ".join(f"{a * 100:.0f}%" for a in areas.tolist()), fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/bands_bounds.png")
    args = ap.parse_args()
    panels = [(dict(sb.INIT_SHAPE), "start")]
    for n in sb.PARAM_NAMES:
        lo, hi, _ = sb.BOUNDS[n]
        for v, tag in ((lo, "min"), (hi, "max")):
            vals = dict(sb.INIT_SHAPE)
            vals[n] = v
            shown = f"{math.degrees(v):.1f} deg" if "theta" in n else f"{v:.2f}"
            panels.append((vals, f"{n} {tag} = {shown}"))
    g = torch.Generator().manual_seed(0)
    for i in range(5):
        raw = sb.init_raw(True) + torch.randn(8, generator=g) * 1.5
        dec = sb.decode_params(raw[None], torch.ones(1), True)
        vals = {n: (dec["dot_r"] if n == "dot_scale" else dec[n]).item() for n in sb.PARAM_NAMES}
        panels.append((vals, f"random draw {i + 1}"))
    cols = 6
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (vals, title) in zip(axes.flat, panels):
        ax.axis("on")
        draw(ax, vals, title)
    fig.suptitle(f"Bounded mask shapes on a {N}x{N} map (dot at r50 = {R50}); blue dot, orange star, green lobe; "
                 "areas dot / star / lobe / rest", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print("saved", args.out)


if __name__ == "__main__":
    main()
