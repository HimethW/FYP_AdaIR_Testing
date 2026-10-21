"""Are the frequency branches (AFLBs) actually used? PSNR with the branches on vs off, plus how much they change
the decoder features.

"Off" = para1 = 0 in all three AFLBs, so each block returns y * para2 (its branch output is removed). The feature
change is mean |out - y * para2| / mean |y * para2| per AFLB, i.e. |agg * para1| relative to the decoder feature.
Runs on the CPU by default (never competes with a training run for GPU memory), first N images per set, centre crops.
  python tools/check_branch_usage.py --ckpt runs/ours_ft/last.pt --ema --n 6 --crop 256
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.evaluate import load_sample, size, test_sets  # noqa: E402
from fyp.metrics import psnr_ssim  # noqa: E402
from fyp.models import build_model, load_weights  # noqa: E402

SETS = ["bsd68_s25", "rain100l", "sots_unseen", "haze_noise25_unseen", "rain_noise25", "haze_rain_cdd11"]
AFLBS = ("fre1", "fre2", "fre3")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default=None, help="default: the checkpoint's own")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="write the numbers here as JSON")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    variant = args.variant or ck.get("variant") or "original"
    model = build_model(variant)
    load_weights(model, args.ckpt, use_ema=args.ema)
    model = model.to(args.device).eval()
    fres = {n: getattr(model, n) for n in AFLBS}
    ratio = {n: [] for n in AFLBS}

    measure = {"on": True}  # record the feature change only in the branches-on pass

    def hook(name):
        def f(mod, inp, out):
            if measure["on"]:
                base = inp[1] * mod.para2
                ratio[name].append(float((out - base).abs().mean() / base.abs().mean()))
        return f
    for n, m in fres.items():
        m.register_forward_hook(hook(n))
    para1 = {n: m.para1.detach().clone() for n, m in fres.items()}

    sets = test_sets(args.data)
    res = {}
    with torch.no_grad():
        for name in args.sets:
            spec = sets[name]
            on, off = [], []
            for i in range(min(args.n, size(spec))):
                _, lq, gt = load_sample(spec, i, "adair")
                h, w = lq.shape[:2]
                c = min(args.crop, h, w) // 16 * 16
                y0, x0 = (h - c) // 2, (w - c) // 2
                x = torch.from_numpy(lq[y0:y0 + c, x0:x0 + c]).permute(2, 0, 1)[None].float().div(255)
                t = torch.from_numpy(gt[y0:y0 + c, x0:x0 + c]).permute(2, 0, 1).float().div(255)
                measure["on"] = True
                on.append(psnr_ssim(model(x.to(args.device)).clamp(0, 1), t)[0])
                measure["on"] = False
                for m in fres.values():
                    m.para1.zero_()
                off.append(psnr_ssim(model(x.to(args.device)).clamp(0, 1), t)[0])
                for n, m in fres.items():
                    m.para1.copy_(para1[n])
            res[name] = {"on": float(np.mean(on)), "off": float(np.mean(off)), "n": len(on)}

    print(f"{args.ckpt} (step {ck.get('step')}, variant {variant}, {'EMA' if args.ema else 'raw'} weights), "
          f"first {args.n} images per set, centre {args.crop} crops")
    print("| Set | Branches on | Branches off | On - off |\n|---|---|---|---|")
    for name, r in res.items():
        print(f"| {name} | {r['on']:.3f} | {r['off']:.3f} | {r['on'] - r['off']:+.3f} |")
    feat = {n: float(np.mean(v)) for n, v in ratio.items()}
    p1 = {n: float(p.abs().mean()) for n, p in para1.items()}
    print("AFLB change to the decoder feature |agg * para1| / |y * para2|: "
          + ", ".join(f"{n} {v:.3f}" for n, v in feat.items()))
    print("para1 mean |.|: " + ", ".join(f"{n} {v:.4f}" for n, v in p1.items()))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "step": ck.get("step"), "variant": variant, "ema": args.ema, "n": args.n,
                       "crop": args.crop, "psnr": res, "feature_change": feat, "para1_abs_mean": p1}, f, indent=1)


if __name__ == "__main__":
    main()
