"""Train AdaIR variants: restoration training, or DCPT pre-training followed by fine-tuning.

  # restoration training from scratch
  python train_fyp.py --name shaped_aio3mix --variant shaped --tasks aio3_mix --iters 100000
  # DCPT pre-training, then restoration fine-tuning from its weights
  python train_fyp.py --name dcpt_shaped --stage dcpt --variant shaped --tasks aio3_mix --iters 20000 --batch 2 --accum 16
  python train_fyp.py --name shaped_dcpt_ft --variant shaped --tasks aio3_mix --init runs/dcpt_shaped/last.pt --iters 100000

Long runs: start detached so they survive session restarts, e.g.
  setsid nohup python train_fyp.py ... > runs/<name>.out 2>&1 &
Ctrl-C / SIGTERM saves runs/<name>/last.pt; running the same command again resumes from it.

Everything for a run goes to runs/<name>/ (gitignored): args.json, train.log, metrics.jsonl, tb/, last.pt,
snapshots iter_XXXXXXX.pt. One "iteration" = one optimiser step = batch x accum samples.

Precision: true fp32 by default (TF32 off; --tf32 / --precision bf16 trade accuracy for speed, results.md A12).
Defaults follow AdaIR's train.py where it has one (AdamW lr 2e-4, default weight decay 0.01, warmup then
cosine, 128 crops, effective batch 32 = 4 x 8 accumulation on one 16 GB GPU) and the DCPT code for the DCPT
stage (net AdamW lr 2e-4 wd 0, classifier AdamW lr 5e-5 wd 1e-4, focal loss, L1 on clean images).
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fyp.data import CLASS_NAMES, LABELS, TrainSet, parse_tasks, worker_init
from fyp.evaluate import evaluate, load_sample, resolve, test_sets
from fyp.models import VARIANTS, build_model, is_backbone, load_weights
from patches.dcpt import DCPT
from patches.losses import PixelFFTLoss

VAL_SETS = ["bsd68_s25", "rain100l", "sots_unseen", "haze_noise25_unseen", "rain_noise25", "haze_rain_unseen",
            "haze_rain_cdd11"]
# test sets used to check DCPT degradation classification, with their class labels
DCPT_VAL = {"bsd68_s25": LABELS["denoise"], "rain100l": LABELS["derain"], "sots_unseen": LABELS["dehaze"],
            "haze_noise25_unseen": LABELS["haze_noise"], "rain_noise25": LABELS["rain_noise"],
            "haze_rain_unseen": LABELS["haze_rain"], "haze_rain_cdd11": LABELS["haze_rain_cdd11"]}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--stage", choices=("restore", "dcpt"), default="restore")
    p.add_argument("--variant", choices=VARIANTS, default="shaped")
    p.add_argument("--tasks", default="aio3_mix", help="preset (aio3, aio3_mix) or task:count,task:count")
    p.add_argument("--iters", type=int, required=True, help="optimiser steps")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=None, help="default 0.01 (restore, AdaIR) / 0 (dcpt net, DCPT)")
    p.add_argument("--warmup", type=float, default=15 / 180, help="fraction of iters (AdaIR: 15 of 180 epochs)")
    p.add_argument("--loss", choices=("l1", "l1_fft"), default="l1_fft")
    p.add_argument("--fft_weight", type=float, default=0.1)
    p.add_argument("--fft_norm", choices=("backward", "ortho"), default="backward")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                   help="fp32 (default, max quality) or bf16 autocast (frequency path and loss stay fp32)")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 in convs / matmuls (~18%% faster, 10-bit multiply inputs); default off = true fp32")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off (DCPT's configs use 0.999)")
    p.add_argument("--clip", type=float, default=0.0, help="grad-norm clip, 0 = off")
    p.add_argument("--init", default=None, help="start from these weights (e.g. a DCPT run's last.pt)")
    p.add_argument("--init_backbone", action="store_true",
                   help="from --init load only the backbone (+ para2), all AFLB weights start fresh")
    p.add_argument("--freeze_backbone", type=int, default=0,
                   help="train only the AFLBs (and the DCPT classifier) for this many iterations; -1 = whole run")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--val_n", type=int, default=10, help="images per validation set")
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--keep_every", type=int, default=20000)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    return p.parse_args()


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.state[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
            else:
                self.state[k].copy_(v)


def lr_factor(step, total, warmup_frac, min_ratio):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def shape_stats(model):
    """Mean decoded shape params and band areas of the last forward (shaped variants)."""
    out = {}
    for n in ("fre1", "fre2", "fre3"):
        fre = getattr(model, n, None)
        if fre is None or not hasattr(fre, "last_params"):
            continue
        for k, v in fre.last_params.items():
            out[f"shape/{n}/{k}"] = v.float().mean().item()
        for b, a in zip(("dot", "star", "lobe", "rest"), fre.last_band_area.float().mean(0).tolist()):
            out[f"band_area/{n}/{b}"] = a
    return out


@torch.no_grad()
def dcpt_accuracy(dcpt, sets, n, device, crop=256):
    dcpt.eval()
    correct, total, per = 0, 0, {}
    for name, label in DCPT_VAL.items():
        if name not in sets:
            continue
        spec = sets[name]
        hits = 0
        k = min(n, len(spec.get("pairs") or spec.get("clean")))
        for i in range(k):
            _, lq, _ = load_sample(spec, i, "adair")
            h, w = lq.shape[:2]
            c = min(crop, h, w) // 16 * 16
            y, x = (h - c) // 2, (w - c) // 2
            t = torch.from_numpy(lq[y:y + c, x:x + c]).permute(2, 0, 1)[None].float().div(255).to(device)
            hits += int(dcpt.classify(t).argmax(1).item() == label)
        per[name] = hits / k
        correct, total = correct + hits, total + k
    dcpt.train()
    return correct / max(total, 1), per


def save(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = get_args()
    run = os.path.join(args.runs, args.name)
    os.makedirs(run, exist_ok=True)
    log = Logger(os.path.join(run, "train.log"))
    if args.wd is None:
        args.wd = 0.0 if args.stage == "dcpt" else 0.01
    with open(os.path.join(run, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log("args: " + json.dumps(vars(args)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.allow_tf32 = args.tf32  # PyTorch's default is True for convs
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    device = "cuda"

    model = build_model(args.variant, amp_safe=args.precision == "bf16").to(device)
    if args.init:
        _, missing, unexpected = load_weights(model, args.init, strict=False, backbone=args.init_backbone)
        log(f"init from {args.init}{' (backbone + para2 only)' if args.init_backbone else ''}: "
            f"missing {len(missing)} keys (fresh), unexpected {len(unexpected)} (ignored)")
        if missing:
            log(f"  fresh: {sorted({'.'.join(k.split('.')[:2]) for k in missing})}")

    counts = parse_tasks(args.tasks)
    if args.stage == "dcpt":
        dcpt = DCPT(model, num_classes=len(CLASS_NAMES)).to(device).train()
        opt_net, opt_cls = dcpt.optimizers(lr_net=args.lr)
        for g in opt_net.param_groups:
            g["weight_decay"] = args.wd
        optimizers = [opt_net, opt_cls]
    else:
        dcpt = None
        optimizers = [torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)]
    min_ratio = args.min_lr / args.lr
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: lr_factor(s, args.iters, args.warmup, min_ratio))
                  for o in optimizers]
    loss_fn = PixelFFTLoss(args.fft_weight, args.fft_norm) if args.loss == "l1_fft" else None

    def set_backbone_frozen(frozen):
        n = 0
        for name, prm in model.named_parameters():
            if is_backbone(name):
                prm.requires_grad_(not frozen)
            n += prm.numel() if prm.requires_grad else 0
        if dcpt is not None:
            n += sum(p.numel() for p in dcpt.classifier.parameters())
        log(f"backbone {'frozen' if frozen else 'trainable'}: {n / 1e6:.2f}M trainable params")
    ema = EMA(model, args.ema) if args.ema > 0 else None

    data = TrainSet(counts, root=args.data, patch=args.patch, seed=args.seed)
    log(f"train set: {len(data)} samples / epoch from {counts}; {args.iters} iters x {args.batch * args.accum} "
        f"= {args.iters * args.batch * args.accum / len(data):.2f} epochs")
    sets = test_sets(args.data)
    val_names = resolve(VAL_SETS, sets)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M; val sets: {val_names}")

    step, epoch = 0, 0
    last = os.path.join(run, "last.pt")
    if os.path.exists(last) and not args.no_resume:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if dcpt is not None and ck.get("classifier"):
            dcpt.classifier.load_state_dict(ck["classifier"])
        for o, s in zip(optimizers, ck["optimizers"]):
            o.load_state_dict(s)
        for sch, s in zip(schedulers, ck["schedulers"]):
            sch.load_state_dict(s)
        if ema is not None and ck.get("ema"):
            ema.state = ck["ema"]
        step, epoch = ck["step"], ck["epoch"] + 1
        log(f"resumed from {last} at step {step} (starting epoch {epoch})")

    def checkpoint(path):
        save(path, {"model": model.state_dict(), "ema": ema.state if ema else None,
                    "classifier": dcpt.classifier.state_dict() if dcpt is not None else None,
                    "optimizers": [o.state_dict() for o in optimizers],
                    "schedulers": [s.state_dict() for s in schedulers],
                    "step": step, "epoch": epoch, "variant": args.variant, "stage": args.stage, "args": vars(args)})

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True
        log(f"signal {signum}: will save and stop after this step")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    frozen = args.freeze_backbone != 0 and (args.freeze_backbone < 0 or step < args.freeze_backbone)
    set_backbone_frozen(frozen)

    tb = SummaryWriter(os.path.join(run, "tb"))
    metrics_f = open(os.path.join(run, "metrics.jsonl"), "a")

    def write_metrics(d):
        metrics_f.write(json.dumps({"step": step, **d}) + "\n")
        metrics_f.flush()
        for k, v in d.items():
            if isinstance(v, (int, float)):
                tb.add_scalar(k, v, step)

    def batches():
        nonlocal epoch
        while True:
            data.set_epoch(epoch)
            loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                                drop_last=True, worker_init_fn=worker_init, persistent_workers=False,
                                generator=torch.Generator().manual_seed(args.seed * 7919 + epoch))
            for b in loader:
                yield b
            epoch += 1

    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16")
    stream = batches()
    acc = {}
    t0, n_img = time.time(), 0
    model.train()
    while step < args.iters and not stop["flag"]:
        if frozen and args.freeze_backbone > 0 and step >= args.freeze_backbone:
            frozen = False
            set_backbone_frozen(False)
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            b = next(stream)
            lq, gt = b["lq"].to(device, non_blocking=True), b["gt"].to(device, non_blocking=True)
            with autocast:
                if dcpt is not None:
                    loss, parts = dcpt(lq, gt, b["label"].to(device))
                else:
                    out = model(lq)
                    if loss_fn is not None:
                        loss, parts = loss_fn(out.float(), gt)
                    else:
                        loss = torch.nn.functional.l1_loss(out.float(), gt)
                        parts = {"l_pix": loss.detach()}
            (loss / args.accum).backward()
            parts["loss"] = loss.detach()
            for k, v in parts.items():
                acc.setdefault(k, []).append(float(v))
            n_img += lq.shape[0]
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
        for o in optimizers:
            o.step()
        for s in schedulers:
            s.step()
        if ema is not None:
            ema.update(model)
        step += 1

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            m = {f"train/{k}": float(np.mean(v)) for k, v in acc.items()}
            m["lr"] = schedulers[0].get_last_lr()[0]
            m["img_per_s"] = n_img / dt
            m["epoch"] = epoch
            m.update(shape_stats(model))
            eta = (args.iters - step) * args.batch * args.accum / max(m["img_per_s"], 1e-6)
            log(f"step {step}/{args.iters} ep {epoch} " + " ".join(f"{k.split('/')[-1]} {v:.4g}" for k, v in m.items()
                if k.startswith("train/")) + f" lr {m['lr']:.2e} {m['img_per_s']:.1f} img/s "
                f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB eta {eta / 3600:.1f}h")
            write_metrics(m)
            acc, t0, n_img = {}, time.time(), 0

        if step % args.val_every == 0 or step == args.iters:
            if dcpt is not None:
                top1, per = dcpt_accuracy(dcpt, sets, args.val_n, device)
                log(f"val dcpt top-1 {top1:.3f} " + " ".join(f"{k} {v:.2f}" for k, v in per.items()))
                write_metrics({"val/dcpt_top1": top1, **{f"val/dcpt_{k}": v for k, v in per.items()}})
            else:
                net = model
                if ema is not None:
                    net = build_model(args.variant, amp_safe=args.precision == "bf16").to(device)
                    net.load_state_dict(ema.state)
                log(f"val ({'ema' if ema else 'model'}, first {args.val_n} images per set):")
                res = evaluate(net, sets, val_names, device=device, limit=args.val_n, log=log)
                write_metrics({f"val/{k}_psnr": v["psnr"] for k, v in res.items()})
                if net is not model:
                    del net
                model.train()
            t0 = time.time()

        if step % args.ckpt_every == 0 or step == args.iters or stop["flag"]:
            checkpoint(last)
        if step % args.keep_every == 0:
            checkpoint(os.path.join(run, f"iter_{step:07d}.pt"))

    checkpoint(last)
    log(f"stopped at step {step}" if stop["flag"] else f"finished {step} iterations")
    metrics_f.close()
    tb.close()


if __name__ == "__main__":
    sys.exit(main())
