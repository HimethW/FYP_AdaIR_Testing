"""Per-epoch CDD-11 table (our 6 classes) next to the published results of EvoIR and OneRestore.

Reads <run>/test/epoch_XX.json (written by train_fyp.py --test_sets cdd6 after every epoch) and writes a markdown
table: PSNR / SSIM per class, the 6-class mean, and the mean's difference to EvoIR and OneRestore.
  python tools/cdd_table.py --run runs/cdd6_ft [--out results.md]

Published numbers: EvoIR (arXiv 2512.05104) Table 4, which repeats OneRestore's (ECCV 2024) CDD-11 table for the
other methods. Both use OneRestore's protocol (full 1080 x 720 test images, 8-bit outputs, skimage PSNR / SSIM with
win_size 7), the same as our `onerestore` protocol. They train on all 11 CDD-11 classes; their 6-class mean here is
the plain mean of their per-class numbers for our 6 classes.
"""
import argparse
import glob
import json
import os
import re

CLASSES = ("low", "haze", "rain", "haze_rain", "low_haze", "low_rain")
HEAD = ("low", "haze", "rain", "haze+rain", "low+haze", "low+rain")
PUBLISHED = {  # PSNR, SSIM per class, in CLASSES order
    "EvoIR": [(27.06, .830), (32.24, .991), (34.03, .970), (29.76, .965), (25.92, .824), (26.02, .806)],
    "OneRestore": [(26.48, .826), (32.52, .990), (33.40, .964), (29.99, .957), (25.79, .822), (25.58, .799)],
}


def mean(rows):
    return sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)


def ours(path):
    """Per-class (PSNR, SSIM); None where a class was not evaluated."""
    res = json.load(open(path))["results"]
    return [(res[f"cdd11_{c}"]["psnr"], res[f"cdd11_{c}"]["ssim"]) if f"cdd11_{c}" in res else None for c in CLASSES]


def table(run):
    files = sorted(glob.glob(os.path.join(run, "test", "epoch_*.json")))
    ref = {k: mean(v) for k, v in PUBLISHED.items()}
    lines = ["| model | " + " | ".join(HEAD) + " | **mean (6)** | Δ EvoIR | Δ OneRestore |",
             "|---|" + "---|" * (len(HEAD) + 3)]
    for name, rows in PUBLISHED.items():
        m = ref[name]
        lines.append(f"| {name} (published) | " + " | ".join(f"{p:.2f} / {s:.3f}" for p, s in rows)
                     + f" | **{m[0]:.2f} / {m[1]:.3f}** | | |")
    best = None
    for f in files:
        ep = int(re.search(r"epoch_(\d+)", f).group(1))
        rows = ours(f)
        cells = " | ".join(f"{r[0]:.2f} / {r[1]:.4f}" if r else "-" for r in rows)
        if any(r is None for r in rows):  # incomplete (smoke test): no mean
            lines.append(f"| ours, epoch {ep} | {cells} | - | - | - |")
            continue
        m = mean(rows)
        if best is None or m[0] > best[1]:
            best = (ep, m[0])
        d = {k: (m[0] - v[0], m[1] - v[1]) for k, v in ref.items()}
        lines.append(f"| ours, epoch {ep} | {cells}"
                     + f" | **{m[0]:.2f} / {m[1]:.4f}** | {d['EvoIR'][0]:+.2f} / {d['EvoIR'][1]:+.4f}"
                     + f" | {d['OneRestore'][0]:+.2f} / {d['OneRestore'][1]:+.4f} |")
    note = (f"\nPSNR / SSIM, CDD-11 test (200 images per class, full resolution, OneRestore protocol). "
            f"Ours: EMA weights; per-epoch evaluation runs with TF32 inference (mean difference to true fp32 "
            f"-0.0004 dB on a 12-image check). Published methods train on all 11 classes."
            + (f" Best epoch so far by mean PSNR: {best[0]}." if best else ""))
    return "\n".join(lines) + "\n" + note + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default: <run>/test/table.md")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "test", "table.md")
    text = table(args.run)
    with open(out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
