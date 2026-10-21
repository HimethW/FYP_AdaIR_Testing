"""Evaluate an AdaIR variant on the test sets (PSNR / SSIM, RGB, true fp32: TF32 off).

  # released AdaIR, the paper's 3-task sets (should reproduce the paper's Table 1)
  python eval_fyp.py --ckpt ckpt/adair3d.ckpt --variant original --sets paper3
  # one of our runs, everything
  python eval_fyp.py --ckpt runs/shaped_aio3mix/last.pt --variant shaped --sets main extra --out runs/shaped_aio3mix/eval.json

--sets takes preset names (paper3, main, extra; see fyp/evaluate.py) and/or set names.
--protocol adair (default) centre-crops to a multiple of 16 like AdaIR's test.py; full uses whole images.
Writes a JSON with per-set means and per-image PSNR / SSIM (for paired comparisons).
"""
import argparse
import json
import os
import time

import torch

from fyp.evaluate import evaluate, resolve, test_sets
from fyp.models import VARIANTS, build_model, load_weights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--variant", choices=VARIANTS, default=None, help="default: taken from our checkpoint")
    p.add_argument("--ema", action="store_true", help="use the EMA weights of one of our checkpoints")
    p.add_argument("--sets", nargs="+", default=["main"])
    p.add_argument("--protocol", choices=("adair", "full"), default="adair")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_images", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--data", default="data")
    p.add_argument("--device", default="cuda", help="cuda, or cpu (e.g. while a training run holds the GPU)")
    args = p.parse_args()

    torch.backends.cudnn.allow_tf32 = False  # true fp32 inference
    torch.backends.cuda.matmul.allow_tf32 = False
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    variant = args.variant or (ckpt.get("variant") if isinstance(ckpt, dict) else None)
    if variant is None:
        raise SystemExit("--variant is needed for this checkpoint")
    model = build_model(variant)
    _, missing, unexpected = load_weights(model, args.ckpt, use_ema=args.ema)
    model = model.to(args.device).eval()
    sets = test_sets(args.data)
    names = resolve(args.sets, sets)
    print(f"{args.ckpt} ({variant}{', ema' if args.ema else ''}), protocol {args.protocol}, fp32")
    t0 = time.time()
    res = evaluate(model, sets, names, device=args.device, protocol=args.protocol, limit=args.limit,
                   save_dir=args.save_images)
    print(f"done in {time.time() - t0:.0f}s")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "variant": variant, "ema": args.ema, "protocol": args.protocol,
                       "step": ckpt.get("step") if isinstance(ckpt, dict) else None, "results": res}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
