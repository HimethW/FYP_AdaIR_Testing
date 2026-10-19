"""Test sets and evaluation (PSNR / SSIM, RGB, [0, 1]) shared by eval_fyp.py and training validation.

Protocols:
  adair  AdaIR's test.py: every image (clean and degraded) centre-cropped to a multiple of 16
         (utils.image_utils.crop_img), noise added after cropping. Use this to compare with the paper.
  full   whole image, reflect-padded to a multiple of 16 for the network, output cropped back.
Noise for the synthetic denoise sets is seeded per image (crc32 of name and sigma), so results do
not depend on file order (AdaIR's test.py uses one global seed in os.listdir order).
Very large images (O-HAZE ~ 12 MP) are run in overlapping tiles (`max_pixels`, `tile`, `overlap`).
"""
import glob
import os
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fyp.metrics import psnr_ssim
from tools.mixtures import add_noise
from utils.image_utils import crop_img

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN_SPLIT = os.path.join(REPO, "splits/sots_outdoor_seen_in_ots.txt")


def _pairs(folder, names=None):
    """(input, target) paths; target has the same name, or <id>.png with id = name up to the first '_'."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "input", "*"))):
        name = os.path.basename(p)
        if names is not None and name not in names and os.path.splitext(name)[0] not in names:
            continue
        t = os.path.join(folder, "target", name)
        if not os.path.exists(t):
            t = os.path.join(folder, "target", name.split("_")[0] + ".png")
        out.append((p, t))
    return out


def _seen():
    if not os.path.exists(SEEN_SPLIT):
        return None
    return {l.strip() for l in open(SEEN_SPLIT) if l.strip()}


def _split(pairs, seen, keep_seen):
    if seen is None:
        return None
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
    seen_stems = {os.path.splitext(s)[0] for s in seen}
    return [(a, b) for a, b in pairs if (stem(a) in seen_stems) == keep_seen]


def test_sets(root="data"):
    """name -> dict(kind='noise', clean=[paths], sigma) or dict(kind='pairs', pairs=[(lq, gt)])."""
    t = os.path.join(root, "test")
    sets = {}
    for name, folder in (("bsd68", "denoise/bsd68"), ("urban100", "denoise/urban100"), ("kodak24", "denoise/kodak24")):
        clean = sorted(glob.glob(os.path.join(t, folder, "*")))
        for s in (15, 25, 50):
            sets[f"{name}_s{s}"] = dict(kind="noise", clean=clean, sigma=s)
    sets["rain100l"] = dict(kind="pairs", pairs=_pairs(f"{t}/derain/Rain100L"))
    seen = _seen()
    for name, folder in (("sots", f"{t}/dehaze"), ("haze_noise25", f"{t}/mix/haze_noise25"),
                         ("haze_rain", f"{t}/mix/haze_rain")):
        pairs = _pairs(folder)
        sets[name] = dict(kind="pairs", pairs=pairs)
        for tag, keep in (("unseen", False), ("seen", True)):
            sub = _split(pairs, seen, keep)
            if sub is not None:
                sets[f"{name}_{tag}"] = dict(kind="pairs", pairs=sub)
    sets["rain_noise25"] = dict(kind="pairs", pairs=_pairs(f"{t}/mix/rain_noise25"))
    sets["haze_rain_cdd11"] = dict(kind="pairs", pairs=_pairs(f"{t}/mix/haze_rain_cdd11"))
    ohaze = _pairs(f"{t}/dehaze_ohaze")
    nhhaze = _pairs(f"{t}/dehaze_nhhaze")
    sets["ohaze"] = dict(kind="pairs", pairs=ohaze)
    sets["ohaze_41_45"] = dict(kind="pairs", pairs=[p for p in ohaze if 41 <= int(os.path.basename(p[0])[:2]) <= 45])
    sets["nhhaze"] = dict(kind="pairs", pairs=nhhaze)
    sets["nhhaze_51_55"] = dict(kind="pairs", pairs=[p for p in nhhaze if 51 <= int(os.path.basename(p[0])[:2]) <= 55])
    return {k: v for k, v in sets.items() if (v.get("pairs") or v.get("clean"))}


PRESETS = {
    "paper3": ["bsd68_s15", "bsd68_s25", "bsd68_s50", "rain100l", "sots"],
    "main": ["bsd68_s15", "bsd68_s25", "bsd68_s50", "rain100l", "sots", "sots_unseen", "sots_seen",
             "haze_noise25", "haze_noise25_unseen", "rain_noise25", "haze_rain", "haze_rain_unseen", "haze_rain_cdd11"],
    "extra": ["urban100_s15", "urban100_s25", "urban100_s50", "kodak24_s15", "kodak24_s25", "kodak24_s50",
              "ohaze_41_45", "nhhaze_51_55", "ohaze", "nhhaze"],
}


def resolve(names, available):
    out = []
    for n in names:
        out += PRESETS[n] if n in PRESETS else [n]
    return [n for n in dict.fromkeys(out) if n in available]


def _read(p):
    return np.array(Image.open(p).convert("RGB"))


def load_sample(spec, i, protocol):
    """(name, degraded uint8 HWC, clean uint8 HWC) for item i of a test set."""
    if spec["kind"] == "noise":
        p = spec["clean"][i]
        clean = _read(p)
        if protocol == "adair":
            clean = crop_img(clean, base=16)
        name = os.path.basename(p)
        rng = np.random.default_rng(zlib.crc32(f"{name}|{spec['sigma']}".encode()))
        return name, add_noise(clean, spec["sigma"], rng), clean
    lq_p, gt_p = spec["pairs"][i]
    lq, gt = _read(lq_p), _read(gt_p)
    if protocol == "adair":
        lq, gt = crop_img(lq, base=16), crop_img(gt, base=16)
    return os.path.basename(lq_p), lq, gt


def size(spec):
    return len(spec["clean"]) if spec["kind"] == "noise" else len(spec["pairs"])


@torch.no_grad()
def run_model(model, x, mult=16, max_pixels=2_600_000, tile=1024, overlap=64):
    """x: 1x3xHxW in [0, 1]. Pads to a multiple of `mult`; tiles images above max_pixels."""
    h, w = x.shape[-2:]
    if h * w <= max_pixels:
        ph, pw = (-h) % mult, (-w) % mult
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        return model(xp)[..., :h, :w]
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x[:, :1])
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) + ([h - tile] if h > tile and (h - tile) % step else [])
    xs = list(range(0, max(w - tile, 0) + 1, step)) + ([w - tile] if w > tile and (w - tile) % step else [])
    ramp = lambda n: torch.minimum(torch.arange(n, device=x.device) + 1,  # noqa: E731
                                   torch.arange(n, 0, -1, device=x.device)).clamp(max=overlap).float()
    for y in ys:
        for x0 in xs:
            patch = x[..., y:y + tile, x0:x0 + tile]
            th, tw = patch.shape[-2:]
            pred = run_model(model, patch, mult, max_pixels=tile * tile + 1)
            wgt = (ramp(th)[:, None] * ramp(tw)[None, :])[None, None]
            out[..., y:y + th, x0:x0 + tw] += pred * wgt
            weight[..., y:y + th, x0:x0 + tw] += wgt
    return out / weight


@torch.no_grad()
def evaluate(model, sets, names, device="cuda", protocol="adair", limit=None, save_dir=None, log=print):
    """Returns {set: {"psnr", "ssim", "n", "per_image": {name: [psnr, ssim]}}}."""
    model.eval()
    results = {}
    for name in names:
        spec = sets[name]
        n = size(spec) if limit is None else min(limit, size(spec))
        per = {}
        for i in range(n):
            img_name, lq, gt = load_sample(spec, i, protocol)
            x = torch.from_numpy(lq).permute(2, 0, 1)[None].float().div(255).to(device)
            y = run_model(model, x).clamp(0, 1)
            per[img_name] = psnr_ssim(y, torch.from_numpy(gt).permute(2, 0, 1).float().div(255))
            if save_dir:
                os.makedirs(os.path.join(save_dir, name), exist_ok=True)
                Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                    os.path.join(save_dir, name, os.path.splitext(img_name)[0] + ".png"))
        vals = np.array(list(per.values()))
        results[name] = {"psnr": float(vals[:, 0].mean()), "ssim": float(vals[:, 1].mean()), "n": n,
                         "per_image": {k: [float(a), float(b)] for k, (a, b) in per.items()}}
        log(f"  {name:22} n={n:4d}  PSNR {results[name]['psnr']:.3f}  SSIM {results[name]['ssim']:.4f}")
    return results
