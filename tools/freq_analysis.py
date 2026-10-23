"""How each degradation changes the frequency content of the images, and how our learned band shapes respond.

Part 1 (data, no model): for every test pair, the luma of a centre 256x256 crop, Hann-windowed, |FFT|^2 in dB, and
the difference degraded - clean, averaged over the set. Red = energy the degradation adds, blue = energy it removes.
Summarised as a radial profile (vs normalised frequency radius: 0 = DC, 1 = Nyquist) and an orientation profile
(mid / high frequencies, 0 deg = the horizontal frequency axis = vertical structures in the image, e.g. rain streaks).
Also the 50%-energy radius r50 of each degraded input, computed as the model computes it for its dot band
(non-DC energy, no window).
Part 2 (model): the per-degradation shape parameters written by tools/plot_mask_shapes.py (shape_params.csv):
dot radius per AFLB, and the shift of every other shape parameter from its clean value, as a % of its allowed range.
  python tools/freq_analysis.py --shape_csv results/meeting_10_25/mask_shapes_epoch20/shape_params.csv \
      --out results/meeting_10_25/frequency_analysis
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.evaluate import load_sample, size, test_sets  # noqa: E402

# label, test set, colour (dataviz reference palette, light mode: noise = one-hue ordinal ramp, the rest fixed slots)
SETS = [("noise σ15", "bsd68_s15", "#6da7ec"), ("noise σ25", "bsd68_s25", "#2a78d6"), ("noise σ50", "bsd68_s50", "#184f95"),
        ("rain", "rain100l", "#eb6834"), ("haze", "sots_unseen", "#1baf7a"),
        ("haze + noise", "haze_noise25_unseen", "#eda100"), ("rain + noise", "rain_noise25", "#e87ba4"),
        ("haze + rain", "haze_rain_unseen", "#008300"), ("haze + rain (CDD-11)", "haze_rain_cdd11", "#6250d6")]
CSV_NAME = {"noise σ15": "noise s15", "noise σ25": "noise s25", "noise σ50": "noise s50", "rain": "rain", "haze": "haze",
            "haze + noise": "haze+noise s25", "rain + noise": "rain+noise s25", "haze + rain": "haze+rain (ours)",
            "haze + rain (CDD-11)": "haze+rain (CDD-11)"}
SHAPE_PARAMS = [("star_theta", "star angle", np.pi / 4 * 180 / np.pi), ("star_p", "star roundness", 0.6),
                ("star_size", "star size", 0.55), ("star_aspect", "star aspect", 1.5 - 2 / 3),
                ("lobe_vertex", "lobe position", 0.5), ("lobe_curv", "lobe curvature", 1.4),
                ("lobe_theta", "lobe angle", np.pi / 3 * 180 / np.pi)]  # (csv key, label, allowed range in csv units)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b2b0", "#e34948", "#8f1d1d"])
N = 256


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def luma(img):
    x = img.astype(np.float64) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def crop(img):
    h, w = img.shape[:2]
    y, x = (h - N) // 2, (w - N) // 2
    return img[y:y + N, x:x + N]


GRID_U = (np.arange(N) - N // 2) / (N / 2)
U, V = np.meshgrid(GRID_U, GRID_U, indexing="ij")  # U = vertical frequency (rows), V = horizontal (columns)
RHO = np.sqrt(U ** 2 + V ** 2)
ANGLE = np.degrees(np.arctan2(U, V)) % 180  # 0 = horizontal frequency axis
HANN = np.outer(np.hanning(N), np.hanning(N))
ORDER = np.argsort(RHO.ravel(), kind="stable")


def power_db(y, window=True):
    f = np.fft.fftshift(np.fft.fft2((y - y.mean()) * HANN if window else y))
    return 10 * np.log10(np.abs(f) ** 2 + 1e-8)


def r50(y):
    e = np.abs(np.fft.fftshift(np.fft.fft2(y))) ** 2
    e = e.ravel()[ORDER]
    e[0] = 0  # DC
    c = np.cumsum(e)
    return float(RHO.ravel()[ORDER][np.searchsorted(c / c[-1], 0.5)])


def radial(m, bins=48):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(RHO, edges) - 1
    return (edges[:-1] + edges[1:]) / 2, np.array([m[idx == i].mean() for i in range(bins)])


def angular(m, lo=0.25, hi=0.95, step=5):
    sel = (RHO >= lo) & (RHO <= hi)
    centers = np.arange(step / 2, 180, step)
    vals = np.array([m[sel & (np.abs(ANGLE - c) < step / 2)].mean() for c in centers])
    return centers, vals


def spectra(data, limit):
    sets = test_sets(data)
    out = {}
    for label, key, _ in SETS:
        spec = sets[key]
        n = min(limit, size(spec))
        diff = np.zeros((N, N))
        r_deg, r_clean = [], []
        for i in range(n):
            _, lq, gt = load_sample(spec, i, "adair")
            a, b = luma(crop(lq)), luma(crop(gt))
            diff += power_db(a) - power_db(b)
            r_deg.append(r50(a))
            r_clean.append(r50(b))
        diff /= n
        out[label] = {"set": key, "n": n, "diff": diff, "r50_degraded": float(np.mean(r_deg)),
                      "r50_degraded_sd": float(np.std(r_deg)), "r50_clean": float(np.mean(r_clean))}
        print(f"{label:22} n={n:4d}  r50 degraded {np.mean(r_deg):.3f} (clean {np.mean(r_clean):.3f})  "
              f"mean change {diff.mean():+.2f} dB", flush=True)
    return out


def fig_maps(sp, path):
    lim = max(np.percentile(np.abs(s["diff"]), 99) for s in sp.values())
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.4), dpi=170, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, _, _) in zip(axes.ravel(), SETS):
        im = ax.imshow(sp[label]["diff"], cmap=DIVERGING, vmin=-lim, vmax=lim, extent=[-1, 1, 1, -1])
        ax.set_title(label, fontsize=12, color=INK)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(AXIS)
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    cb.set_label("degraded − clean power (dB)", color=INK2)
    cb.ax.tick_params(colors=MUTED)
    fig.suptitle("What each degradation does to the spectrum (mean over the test set; centre = DC, edges = Nyquist)\n"
                 "red = energy added, blue = energy removed; horizontal axis = horizontal frequency",
                 fontsize=12, color=INK)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def direct_label(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.plot([x[-1]], [y[-1]], "o", ms=4, color=color)


def fig_profiles(sp, path, fn, xlabel, title, xlim):
    groups = [("Noise levels", SETS[0:3]), ("Single degradations", SETS[3:5]), ("Mixtures", SETS[5:9])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=170, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, members) in zip(axes, groups):
        style(ax)
        ax.axhline(0, color=AXIS, linewidth=1)
        for label, _, color in members:
            x, y = fn(sp[label]["diff"])
            ax.plot(x, y, color=color, linewidth=2, label=label)
            direct_label(ax, x, y, label, color)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[0].set_ylabel("degraded − clean power (dB)")
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def read_shapes(path):
    rows = list(csv.DictReader(open(path)))
    return {(r["degradation"], r["aflb"]): r for r in rows}


def fig_dot(sp, shapes, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), dpi=170, gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    labels = ["clean"] + [s[0] for s in SETS]
    keys = ["clean"] + [CSV_NAME[s[0]] for s in SETS]
    ypos = np.arange(len(labels))[::-1]
    for ax, aflb in zip(axes[:3], ("fre1", "fre2", "fre3")):
        style(ax)
        ax.grid(axis="y", visible=False)
        vals = [float(shapes[(k, aflb)]["dot_r_mean"]) for k in keys]
        ax.barh(ypos, vals, height=0.62, color="#2a78d6", edgecolor=SURFACE, linewidth=2)
        ax.axvline(vals[0], color=INK2, linewidth=1, linestyle="--")
        for y, v in zip(ypos, vals):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels if aflb == "fre1" else [], fontsize=9, color=INK2)
        ax.set_title(f"{aflb}: dot radius", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("radius (fraction of Nyquist)")
        ax.set_xlim(0, max(vals) * 1.25)
    ax = axes[3]
    style(ax)
    xs = [sp[s[0]]["r50_degraded"] for s in SETS]
    ys = [float(shapes[(CSV_NAME[s[0]], "fre2")]["dot_r_mean"]) for s in SETS]
    xc = np.mean([sp[s[0]]["r50_clean"] for s in SETS[:3]])
    ax.scatter([xc] + xs, [float(shapes[("clean", "fre2")]["dot_r_mean"])] + ys, s=60, color="#2a78d6",
               edgecolor=SURFACE, linewidth=2, zorder=3)
    offsets = {"clean": (10, -8), "rain": (-24, 6), "noise σ15": (6, 2), "haze + noise": (-20, -16), "noise σ25": (6, -4),
               "rain + noise": (6, 4), "haze + rain": (-60, 2), "haze": (6, -2), "noise σ50": (-46, 8),
               "haze + rain (CDD-11)": (-112, -4)}
    for x, y, t in zip([xc] + xs, [float(shapes[("clean", "fre2")]["dot_r_mean"])] + ys, ["clean"] + [s[0] for s in SETS]):
        ax.annotate(t, (x, y), xytext=offsets.get(t, (5, 3)), textcoords="offset points", fontsize=8, color=INK2)
    rank = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    r = np.corrcoef(rank(xs), rank(ys))[0, 1]
    keep = [i for i, s in enumerate(SETS) if "CDD" not in s[0]]
    r_wo = np.corrcoef([xs[i] for i in keep], [ys[i] for i in keep])[0, 1]
    ax.set_xlabel("r50 of the degraded input image (data)")
    ax.set_ylabel("learned dot radius, fre2 (model)")
    ax.set_title(f"Dot vs the input's spectral spread: rank corr. {r:.2f};\nPearson {r_wo:.2f} without CDD-11",
                 fontsize=11, color=INK, loc="left")
    fig.suptitle("The dot band adapts per degradation: wider for noise and CDD-11, narrower for haze (dashed = clean)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return {"spearman": float(r), "pearson_without_cdd11": float(r_wo)}


def fig_heat(shapes, path):
    rows, mat = [], []
    for aflb in ("fre1", "fre2", "fre3"):
        for key, name, rng in SHAPE_PARAMS:
            base = float(shapes[("clean", aflb)][f"{key}_mean"])
            rows.append(f"{aflb}  {name}")
            mat.append([100 * (float(shapes[(CSV_NAME[s[0]], aflb)][f"{key}_mean"]) - base) / rng for s in SETS])
    mat = np.array(mat)
    lim = max(np.abs(mat).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(mat, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([s[0] for s in SETS], rotation=30, ha="right", fontsize=10, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9, color=INK2)
    for y in (len(SHAPE_PARAMS) - 0.5, 2 * len(SHAPE_PARAMS) - 0.5):
        ax.axhline(y, color=SURFACE, linewidth=4)
    for (i, j), v in np.ndenumerate(mat):
        if abs(v) >= 0.5:
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * lim else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("shift from clean (% of the parameter's allowed range)", color=INK2)
    cb.ax.tick_params(colors=MUTED)
    ax.set_title("How the star and lobe shape parameters move per degradation (vs clean images; numbers shown for "
                 "|shift| ≥ 0.5%)", fontsize=12, color=INK, loc="left")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return rows, mat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10000, help="images per test set (default: all)")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sp = spectra(args.data, args.limit)
    fig_maps(sp, os.path.join(args.out, "1_spectrum_change_maps.png"))
    fig_profiles(sp, os.path.join(args.out, "2_radial_profile.png"), radial, "frequency radius (0 = DC, 1 = Nyquist)",
                 "Where in frequency each degradation adds or removes energy (radial average)", (0, 1.15))
    fig_profiles(sp, os.path.join(args.out, "3_orientation_profile.png"), angular,
                 "frequency orientation (deg)",
                 "Direction of the change at mid / high frequencies (radius 0.25-0.95); 0 / 180 deg = horizontal "
                 "frequency axis = vertical structures such as rain streaks", (0, 225))
    shapes = read_shapes(args.shape_csv)
    r = fig_dot(sp, shapes, os.path.join(args.out, "4_dot_radius.png"))
    rows, mat = fig_heat(shapes, os.path.join(args.out, "5_shape_param_shifts.png"))
    summary = {"sets": {k: {kk: vv for kk, vv in v.items() if kk != "diff"} for k, v in sp.items()},
               "radial": {k: [list(map(float, a)) for a in radial(v["diff"])] for k, v in sp.items()},
               "angular": {k: [list(map(float, a)) for a in angular(v["diff"])] for k, v in sp.items()},
               "dot_vs_r50_corr_fre2": r,
               "shape_shift_pct_of_range": {row: dict(zip([s[0] for s in SETS], map(float, m))) for row, m in
                                            zip(rows, mat)}}
    with open(os.path.join(args.out, "frequency_analysis.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "spectrum_change_maps.npz"), **{s[1]: sp[s[0]]["diff"] for s in SETS})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
