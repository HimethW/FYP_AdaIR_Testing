"""Comparison table of eval_fyp.py results (full test sets): Markdown, CSV and a PNG for slides.

Each --col is LABEL=path/to/eval.json (eval_fyp.py --out). --ref names the column the differences are taken against;
--diff names the columns that get a "- ref" column right after them.
  python tools/compare_table.py --out results/table \
      --col "Released (measured)=runs/eval_released/adair3d_main.json" \
      --col "Baseline, epoch 6=runs/eval_baseline/epoch6_main.json" \
      --col "Ours, epoch 8=runs/eval_ours_ft/epoch8_main.json" \
      --col "Ours, epoch 20=runs/eval_ours_ft/epoch20_main.json" \
      --ref "Baseline, epoch 6" --diff "Ours, epoch 8" "Ours, epoch 20"
"""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = [("bsd68_s15", "Noise σ15 (BSD68)"), ("bsd68_s25", "Noise σ25 (BSD68)"), ("bsd68_s50", "Noise σ50 (BSD68)"),
        ("rain100l", "Rain (Rain100L)"), ("sots", "Haze (SOTS outdoor)"), ("haze_noise25", "Haze + noise σ25"),
        ("rain_noise25", "Rain + noise σ25"), ("haze_rain", "Haze + rain"), ("haze_rain_cdd11", "Haze + rain (CDD-11)")]
AVERAGES = [("Average, single (5)", ["bsd68_s15", "bsd68_s25", "bsd68_s50", "rain100l", "sots"]),
            ("Average, mixed (4)", ["haze_noise25", "rain_noise25", "haze_rain", "haze_rain_cdd11"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--col", action="append", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--diff", nargs="*", default=[])
    ap.add_argument("--out", required=True, help="output prefix: writes .md, .csv, .png")
    ap.add_argument("--title", default="Full test sets, PSNR (dB) / SSIM")
    args = ap.parse_args()
    cols = []
    for c in args.col:
        label, path = c.split("=", 1)
        cols.append((label, json.load(open(path))["results"]))
    res = dict(cols)
    ref = res[args.ref]

    header = ["Degradation"]
    for label, _ in cols:
        header.append(label)
        if label in args.diff:
            header.append(f"Δ vs {args.ref.split(',')[0].lower()}")
    table, deltas = [], []
    for key, name in ROWS:
        row, drow = [name], [None]
        for label, r in cols:
            row.append(f"{r[key]['psnr']:.2f} / {r[key]['ssim']:.4f}")
            drow.append(None)
            if label in args.diff:
                d = r[key]["psnr"] - ref[key]["psnr"]
                row.append(f"{d:+.2f}")
                drow.append(d)
        table.append(row)
        deltas.append(drow)
    for name, keys in AVERAGES:
        row, drow = [name], [None]
        for label, r in cols:
            m = sum(r[k]["psnr"] for k in keys) / len(keys)
            row.append(f"{m:.2f}")
            drow.append(None)
            if label in args.diff:
                d = m - sum(ref[k]["psnr"] for k in keys) / len(keys)
                row.append(f"{d:+.2f}")
                drow.append(d)
        table.append(row)
        deltas.append(drow)

    with open(args.out + ".md", "w") as f:
        f.write(f"{args.title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in table:
            f.write("| " + " | ".join(row) + " |\n")
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)

    widths = [2.3] + [1.15 if h.startswith("Δ") else 2.0 for h in header[1:]]  # inches
    fig = plt.figure(figsize=(sum(widths), 0.36 * (len(table) + 1)), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    labels = [h.replace(" (", "\n(").replace(", ", ",\n") if len(h) > 16 else h for h in header]
    t = ax.table(cellText=table, colLabels=labels, loc="center", cellLoc="center",
                 colWidths=[w / sum(widths) for w in widths])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
            cell.set_height(cell.get_height() * 1.7)
        elif c == 0:
            cell.set_text_props(ha="left")
        if r > 0 and deltas[r - 1][c] is not None:
            d = deltas[r - 1][c]
            cell.set_facecolor("#dff0d8" if d > 0.02 else "#f6dcdc" if d < -0.02 else "white")
            cell.set_text_props(weight="bold")
        if r > len(ROWS):
            cell.set_text_props(weight="bold")
    fig.suptitle(args.title, fontsize=11, y=1.04)
    fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.1)
    print("wrote", args.out + ".md / .csv / .png")


if __name__ == "__main__":
    main()
