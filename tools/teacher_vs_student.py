"""Teacher (ours) vs a student model on the test sets both were evaluated on: grouped-bar figure + table.

Every model's numbers come from an eval JSON ({"results": {set: {"psnr", "ssim"}}}): ours / baseline / released from
eval_fyp.py --out, the student from the student project's own per-image evaluation (same AdaIR protocol).
  python tools/teacher_vs_student.py --out results/teacher_vs_student \
      --model "Ours (teacher, epoch 20)=runs/eval_ours_ft/epoch20_main.json" \
      --model "Student S70-W16=runs/eval_student/s70_w16_main.json" \
      --model "Baseline (fine-tuned AdaIR, epoch 8)=runs/eval_baseline/epoch8_main.json" \
      --model "Released AdaIR=runs/eval_released/adair3d_main.json"
"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = [("haze_rain_cdd11", "Haze + rain\n(CDD-11)"), ("rain100l", "Rain\n(Rain100L)"), ("sots_unseen", "Haze\n(SOTS, 424)"),
        ("bsd68_s15", "Noise σ15\n(BSD68)"), ("bsd68_s25", "Noise σ25\n(BSD68)"), ("bsd68_s50", "Noise σ50\n(BSD68)")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz reference palette slots 1-5 (validated, light)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="LABEL=path/to/eval.json (2-5 models)")
    ap.add_argument("--label_models", type=int, default=2, help="value labels on the first N models' bars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    models = []
    for m in args.model:
        label, path = m.split("=", 1)
        models.append((label, json.load(open(path))["results"]))
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "comparison.md"), "w") as f, \
            open(os.path.join(args.out, "comparison.csv"), "w", newline="") as fc:
        w = csv.writer(fc)
        head = ["Test set"] + [lbl for lbl, _ in models]
        w.writerow(head)
        f.write("PSNR (dB) / SSIM, full test sets\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for key, name in SETS:
            row = [name.replace("\n", " ")] + [f"{r[key]['psnr']:.2f} / {r[key]['ssim']:.4f}" for _, r in models]
            w.writerow(row)
            f.write("| " + " | ".join(row) + " |\n")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), dpi=170, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(SETS))
    width = 0.8 / len(models)
    for ax, metric, unit in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for i, ((label, r), color) in enumerate(zip(models, COLORS)):
            vals = [r[k][metric] for k, _ in SETS]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, vals, width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
            if i < args.label_models:
                for p, v in zip(pos, vals):
                    ax.annotate(f"{v:.2f}" if metric == "psnr" else f"{v:.3f}", (p, v), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2,
                                rotation=90)
        ax.set_ylabel(unit, color=INK2)
        ax.set_ylim(0, (44 if metric == "psnr" else 1.12))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n for _, n in SETS], fontsize=10, color=INK2)
    axes[0].legend(frameon=False, ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9,
                   labelcolor=INK2)
    title = "Teacher vs student on the same test sets (full sets, AdaIR protocol)"
    fig.suptitle(title + (f"\n{args.note}" if args.note else ""), fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), bbox_inches="tight", facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
