"""Build the fixed mixed-degradation test sets and the SOTS / OTS overlap split.

Run after tools/prepare_data.py. Writes (data/ is gitignored):
  data/test/mix/haze_noise25/{input,target}/  SOTS outdoor hazy + Gaussian noise sigma 25      (500)
  data/test/mix/rain_noise25/{input,target}/  Rain100L test rainy + Gaussian noise sigma 25    (100)
  data/test/mix/haze_rain/{input,target}/     SOTS outdoor hazy + Rain100L *test* streaks     (500)
  data/test/mix/haze_rain_cdd11/{input,target}/  CDD-11 test haze_rain vs clear              (200)
  data/test/mix/manifest.json                 sources, seeds, counts
and (tracked in git):
  splits/sots_outdoor_seen_in_ots.txt         SOTS outdoor hazy images whose clear scene is also an
                                              OTS training image (content match), one name per line
Every set uses identical file names in input/ and target/ (target = the clean image, PNG).
Noise follows AdaIR's convention; numpy seed 0 per set, so the sets are reproducible.

Usage: python tools/build_mixtures.py [--root data]
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mixtures import add_noise, add_streaks, rain_streaks  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p):
    return np.array(Image.open(p).convert("RGB"))


def write(img, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(img).save(p)


def thumb(p, s=32):
    t = np.asarray(Image.open(p).convert("L").resize((s, s), Image.BILINEAR), dtype=np.float64).ravel()
    t = t - t.mean()
    return t / (np.linalg.norm(t) + 1e-8)


def sots_ots_overlap(root):
    """Match each SOTS outdoor gt to its most similar OTS clear image (32x32 grey, cosine)."""
    sots = sorted(glob.glob(f"{root}/test/dehaze/target/*.png"))
    ots = sorted(glob.glob(f"{root}/Train/Dehaze/original/*.jpg"))
    T = np.stack([thumb(p) for p in ots])
    best, best_id = [], []
    for p in sots:
        sim = T @ thumb(p)
        j = int(sim.argmax())
        best.append(sim[j]), best_id.append(os.path.basename(ots[j]))
    best = np.array(best)
    hist = np.histogram(best, bins=[0, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01])[0]
    # confirm candidates at 128x128 (mean abs diff in grey levels)
    seen = []
    for p, s, oid in zip(sots, best, best_id):
        if s < 0.98:
            continue
        a = np.asarray(Image.open(p).convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float64)
        b = np.asarray(Image.open(f"{root}/Train/Dehaze/original/{oid}").convert("L").resize((128, 128),
                       Image.BILINEAR), dtype=np.float64)
        if np.abs(a - b).mean() < 3.0:
            seen.append((os.path.basename(p)[:-4], oid, float(s), float(np.abs(a - b).mean())))
    return seen, hist, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data"))
    args = ap.parse_args()
    root = args.root
    out = f"{root}/test/mix"
    manifest = {}

    sots_in = sorted(glob.glob(f"{root}/test/dehaze/input/*"))
    rain_in = sorted(glob.glob(f"{root}/test/derain/Rain100L/input/*.png"))
    assert len(sots_in) == 500 and len(rain_in) == 100, "run tools/prepare_data.py first"
    sots_gt = lambda p: f"{root}/test/dehaze/target/{os.path.basename(p).split('_')[0]}.png"  # noqa: E731
    rain_gt = lambda p: f"{root}/test/derain/Rain100L/target/{os.path.basename(p)}"  # noqa: E731
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731

    rng = np.random.default_rng(0)
    for p in sots_in:
        write(add_noise(read(p), 25, rng), f"{out}/haze_noise25/input/{stem(p)}.png")
        write(read(sots_gt(p)), f"{out}/haze_noise25/target/{stem(p)}.png")
    manifest["haze_noise25"] = dict(n=len(sots_in), source="SOTS outdoor hazy + N(0, 25^2)", seed=0)

    rng = np.random.default_rng(0)
    for p in rain_in:
        write(add_noise(read(p), 25, rng), f"{out}/rain_noise25/input/{stem(p)}.png")
        write(read(rain_gt(p)), f"{out}/rain_noise25/target/{stem(p)}.png")
    manifest["rain_noise25"] = dict(n=len(rain_in), source="Rain100L test rainy + N(0, 25^2)", seed=0)

    streaks = [rain_streaks(read(p), read(rain_gt(p))) for p in rain_in]
    rng = np.random.default_rng(0)
    pairing = {}
    for i, p in enumerate(sots_in):
        k = int(rng.integers(len(streaks)))
        write(add_streaks(read(p), streaks[k], rng), f"{out}/haze_rain/input/{stem(p)}.png")
        write(read(sots_gt(p)), f"{out}/haze_rain/target/{stem(p)}.png")
        pairing[stem(p)] = stem(rain_in[k])
    manifest["haze_rain"] = dict(n=len(sots_in), seed=0, streak_source_for_each_input=pairing,
                                 source="SOTS outdoor hazy + Rain100L test streak layer max(rainy - clean, 0), "
                                        "resized to the image size, added and clipped")

    cdd = sorted(glob.glob(f"{root}/CDD-11/test/haze_rain/*.png"))
    for p in cdd:
        os.makedirs(f"{out}/haze_rain_cdd11/input", exist_ok=True)
        shutil.copy(p, f"{out}/haze_rain_cdd11/input/{os.path.basename(p)}")
        write(read(f"{root}/CDD-11/test/clear/{os.path.basename(p)}"),
              f"{out}/haze_rain_cdd11/target/{os.path.basename(p)}")
    manifest["haze_rain_cdd11"] = dict(n=len(cdd), source="CDD-11 test haze_rain vs clear (gy65896/CDD-11)")

    seen, hist, best = sots_ots_overlap(root)
    seen_gt = {s[0] for s in seen}
    seen_inputs = [os.path.basename(p) for p in sots_in if os.path.basename(p).split("_")[0] in seen_gt]
    os.makedirs(f"{REPO}/splits", exist_ok=True)
    with open(f"{REPO}/splits/sots_outdoor_seen_in_ots.txt", "w") as f:
        f.write("\n".join(seen_inputs) + "\n")
    manifest["sots_ots_overlap"] = dict(
        gt_scenes_matched=len(seen), hazy_inputs_affected=len(seen_inputs), unseen_inputs=500 - len(seen_inputs),
        best_cosine_hist_bins=[0, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01], best_cosine_hist=hist.tolist(),
        matches=[dict(sots_gt=a, ots_clear=b, cosine=round(c, 5), mad_128=round(m, 3)) for a, b, c, m in seen])
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    for name in ("haze_noise25", "rain_noise25", "haze_rain", "haze_rain_cdd11"):
        print(f"{name:16} input {len(os.listdir(f'{out}/{name}/input'))}  target {len(os.listdir(f'{out}/{name}/target'))}")
    print(f"SOTS/OTS overlap: best-match cosine histogram {hist.tolist()} "
          f"(bins {manifest['sots_ots_overlap']['best_cosine_hist_bins']})")
    print(f"  {len(seen)} SOTS gt scenes found in OTS -> {len(seen_inputs)} of 500 hazy inputs seen, "
          f"{500 - len(seen_inputs)} unseen; list in splits/sots_outdoor_seen_in_ots.txt")


if __name__ == "__main__":
    main()
