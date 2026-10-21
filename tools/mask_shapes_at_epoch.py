"""Wait until a training run passes N epochs, then save that checkpoint's weights and plot its mask shapes.

Polls runs/<run>/train.log every --poll seconds. Once the run has passed --epochs epochs and runs/<run>/last.pt
holds a step at or past that point (checkpoints come every 1,000 steps), it copies the model + EMA weights to
<out>/weights_stepXXXXXXX.pt and runs tools/plot_mask_shapes.py on them (EMA weights, CPU only, so the training
run's GPU memory is never touched). Then it exits.

  setsid nohup python tools/mask_shapes_at_epoch.py --run runs/ours_ft --epochs 5 \
      --out /home/minura/Himeth/results/mask_shapes > /home/minura/Himeth/results/mask_shapes_watch.log 2>&1 &
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never compete with training for GPU memory

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def logged_step(run):
    path = os.path.join(run, "train.log")
    if not os.path.exists(path):
        return None, None
    step, per_epoch = 0, None
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"train set: (\d+) samples / epoch", line)
            if m:
                per_epoch = int(m.group(1))
            m = re.search(r"\] step (\d+)/", line)
            if m:
                step = int(m.group(1))
    return step, per_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ours_ft")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--out", default="/home/minura/Himeth/results/mask_shapes")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    os.chdir(REPO)
    log(f"waiting for {args.run} to pass {args.epochs} epochs")
    target = None
    while True:
        step, per_epoch = logged_step(args.run)
        if step is not None and per_epoch and os.path.exists(os.path.join(args.run, "args.json")):
            with open(os.path.join(args.run, "args.json")) as f:
                a = json.load(f)
            target = math.ceil(args.epochs * per_epoch / (a["batch"] * a["accum"] * a.get("world", 1)))
            if step >= target:
                ck = torch.load(os.path.join(args.run, "last.pt"), map_location="cpu", weights_only=False)
                if ck["step"] >= target:
                    break
                del ck
        time.sleep(args.poll)
    epochs = ck["step"] * a["batch"] * a["accum"] * a.get("world", 1) / per_epoch
    os.makedirs(args.out, exist_ok=True)
    wpath = os.path.join(args.out, f"weights_step{ck['step']:07d}.pt")
    torch.save({"model": ck["model"], "ema": ck["ema"], "step": ck["step"], "epoch": ck["epoch"],
                "variant": ck["variant"], "stage": ck["stage"], "args": ck["args"]}, wpath)
    log(f"{args.run} at step {ck['step']} = {epochs:.2f} epochs (target {args.epochs} = step {target}); saved {wpath}")
    del ck
    cmd = [sys.executable, "-W", "ignore", "tools/plot_mask_shapes.py", "--ckpt", wpath, "--out", args.out,
           "--n", str(args.n), "--threads", "8"]
    if torch.load(wpath, map_location="cpu", weights_only=False).get("ema"):
        cmd.append("--ema")
    log("running: " + " ".join(cmd))
    r = subprocess.run(["nice", "-n", "10"] + cmd)
    log(f"done, exit {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
