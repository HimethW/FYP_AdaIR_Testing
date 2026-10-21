"""Keep a copy of a training run's weights at every validation point (every 5,000 steps ~ 1 epoch), without touching
the run. train_fyp.py overwrites last.pt every 1,000 steps and keeps a full snapshot only every 20,000; at the end we
pick the epoch where our model beats the baseline most, so every epoch's weights must survive.

Polls runs/<run>/train.log; once a multiple of --every has been logged and last.pt holds exactly that step (it does
for the next 1,000 steps), it saves {model, ema, step, epoch, variant, args} (no optimiser state, ~half the size) to
<out>/step_XXXXXXX.pt. Steps it can no longer catch are logged as missed. Exits when the run finishes.
  setsid nohup python tools/keep_epochs.py --run runs/ours_ft > runs/ours_ft_keep_epochs.log 2>&1 &
"""
import argparse
import os
import re
import sys
import time

import torch


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def logged_step(path):
    step, done = 0, False
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"\] step (\d+)/(\d+)", line)
            if m:
                step = int(m.group(1))
            if "finished" in line and "iterations" in line:
                done = True
    return step, done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--out", default=None, help="default: <run>/epochs")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "epochs")
    os.makedirs(out, exist_ok=True)
    last = os.path.join(args.run, "last.pt")
    seen_mtime = None
    log(f"keeping {args.run} every {args.every} steps in {out}")
    while True:
        step, done = logged_step(os.path.join(args.run, "train.log"))
        target = step // args.every * args.every
        dest = os.path.join(out, f"step_{target:07d}.pt")
        handled = os.path.exists(dest) or os.path.exists(dest + ".missed")
        if target > 0 and not handled and os.path.exists(last):
            mtime = os.path.getmtime(last)
            if mtime != seen_mtime:
                seen_mtime = mtime
                ck = torch.load(last, map_location="cpu", weights_only=False)
                if ck["step"] == target:
                    tmp = dest + ".tmp"
                    torch.save({k: ck.get(k) for k in ("model", "ema", "step", "epoch", "variant", "stage", "args")},
                               tmp)
                    os.replace(tmp, dest)
                    log(f"saved {dest}")
                elif ck["step"] > target:
                    log(f"missed step {target}: last.pt is already at step {ck['step']}")
                    open(dest + ".missed", "w").close()
                del ck
        if done:
            log("run finished")
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
