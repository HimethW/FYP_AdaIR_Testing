"""Choose how to run the model on whole CDD-11 images, on held-out TRAINING scenes (never the test set), then test
the chosen setting once.

Settings (same weights, only the way the model is run changes):
  full            whole image at once (the per-epoch test)
  tileN           image cut into N-px tiles (N/4 overlap), restored separately, blended
  localN          channel attention on N-px windows (fyp/local_infer.py, TLC-style); convolutions see the whole image
  localN_aflb     + the frequency blocks (FFT, shape statistics) on the same windows
  tileN_x8        tiles, each restored under the 8 flips / rotations and averaged (self-ensemble, 8x the compute)

  # choose: 20 held-out scenes per class
  python tools/cdd_infer_sweep.py --ckpt runs/cdd6_ft/epochs/epoch_10.pt --sets cdd6val --limit 20 --out sweep.json
  # test the chosen one on the full CDD-11 test (OneRestore protocol)
  python tools/cdd_infer_sweep.py --ckpt ... --sets cdd6 --settings local256_aflb --out test.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fyp.evaluate import SelfEnsemble, Tiled, evaluate, resolve, test_sets  # noqa: E402
from fyp.local_infer import local_inference  # noqa: E402
from fyp.models import build_model  # noqa: E402

SETTINGS = {
    "full": {},
    "tile256": {"tile": 256}, "tile384": {"tile": 384}, "tile512": {"tile": 512},
    "tile256_x8": {"tile": 256, "tta": 8}, "tile384_x8": {"tile": 384, "tta": 8}, "tile512_x8": {"tile": 512, "tta": 8},
    "local192": {"base": 192}, "local256": {"base": 256}, "local384": {"base": 384},
    "local256_aflb": {"base": 256, "aflb": True}, "local384_aflb": {"base": 384, "aflb": True},
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sets", nargs="+", default=["cdd6val"])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS), help="names above, or any tileN / tileN_xK")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fp32", action="store_true", help="true fp32 (default: TF32, as the per-epoch test)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tile_batch", type=int, default=0,
                   help="tiles per forward (0 = as many as 8 tiles of 256 px; 1 = one at a time)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = not args.fp32
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = build_model(ck.get("variant", "shaped")).cuda().eval()
    net.load_state_dict(ck["ema"])
    sets = test_sets("data")
    names = resolve(args.sets, sets)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "sets": names, "limit": args.limit,
                            "precision": "fp32" if args.fp32 else "tf32", "protocol": "onerestore"})
    for s in args.settings:
        m = re.fullmatch(r"tile(\d+)(?:_x(\d+))?", s)
        cfg = SETTINGS[s] if s in SETTINGS else {"tile": int(m.group(1)), **({"tta": int(m.group(2))} if m.group(2) else {})}
        t0 = time.time()
        if "tile" in cfg:
            inner = SelfEnsemble(net, cfg["tta"]) if cfg.get("tta") else net
            r = evaluate(Tiled(inner, cfg["tile"], args.tile_batch), sets, names, protocol="onerestore", limit=args.limit,
                         workers=args.workers, log=lambda *a: None)
        else:
            with local_inference(net, base=cfg.get("base", 10 ** 6), aflb=cfg.get("aflb", False)):
                r = evaluate(net, sets, names, protocol="onerestore", limit=args.limit, workers=args.workers,
                             log=lambda *a: None)
        res[s] = {k: {"psnr": v["psnr"], "ssim": v["ssim"], "n": v["n"], "per_image": v["per_image"]}
                  for k, v in r.items()}
        mp, ms = np.mean([v["psnr"] for v in r.values()]), np.mean([v["ssim"] for v in r.values()])
        print(f"{s:14s} " + " ".join(f"{k.split('_', 1)[1]} {v['psnr']:.2f}" for k, v in r.items())
              + f" | mean {mp:.3f} / {ms:.4f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
