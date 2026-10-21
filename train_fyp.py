"""Train AdaIR variants: restoration training, or DCPT pre-training followed by fine-tuning.

  # restoration training from scratch
  python train_fyp.py --name shaped_aio3mix --variant shaped --tasks aio3_mix --iters 100000
  # DCPT pre-training, then restoration fine-tuning from its weights
  python train_fyp.py --name dcpt_shaped --stage dcpt --variant shaped --tasks aio3_mix --iters 20000 --batch 2 --accum 16
  python train_fyp.py --name shaped_dcpt_ft --variant shaped --tasks aio3_mix --init runs/dcpt_shaped/last.pt --iters 100000

First `source /home/minura/Himeth/env.sh` (env + all caches inside Himeth).
Long runs: start detached so they survive session restarts, e.g.
  setsid nohup python train_fyp.py ... > runs/<name>.out 2>&1 &
Ctrl-C / SIGTERM saves runs/<name>/last.pt; running the same command again resumes from it.

Everything for a run goes to runs/<name>/ (gitignored): args.json, train.log, metrics.jsonl, tb/, last.pt,
snapshots iter_XXXXXXX.pt. One "iteration" = one optimiser step = batch x accum samples (x GPUs).

Multi-GPU (restoration stage, no freezing): start with torchrun. --batch / --accum are per GPU, so 2 GPUs x 4 x 4
is the same 32 crops per step as 4 x 8 on one GPU (no BatchNorm, so the gradient is the same mean over 32 crops).
Checkpoints are interchangeable between 1 and N GPUs.
  torchrun --standalone --nproc_per_node 2 train_fyp.py ... --batch 4 --accum 4

Precision: true fp32 by default (TF32 off; --tf32 / --precision bf16 trade accuracy for speed, results.md A12).
Defaults follow AdaIR's train.py where it has one (AdamW lr 2e-4, default weight decay 0.01, warmup then
cosine, 128 crops, effective batch 32 = 4 x 8 accumulation on one 16 GB GPU) and the DCPT code for the DCPT
stage (net AdamW lr 2e-4 wd 0, classifier AdamW lr 5e-5 wd 1e-4, focal loss, L1 on clean images).
"""
import argparse
import contextlib
import datetime
import json
import math
import os
import signal
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="let cuDNN time its conv algorithms and keep the fastest (same fp32 precision, not bitwise "
                        "reproducible); default off")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off (DCPT's configs use 0.999)")
    p.add_argument("--clip", type=float, default=0.0, help="grad-norm clip, 0 = off")
    p.add_argument("--on_nonfinite", choices=("stop", "skip"), default="stop",
                   help="a NaN / inf gradient: stop (save the last good weights) or skip that update (stops after 20 "
                        "in a row); either way it is logged, and a NaN loss also dumps its batch to the run folder")
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
    world = int(os.environ.get("WORLD_SIZE", 1))  # > 1 when started with torchrun
    rank = int(os.environ.get("RANK", 0))
    main_proc = rank == 0  # only process 0 logs, validates and saves
    if world > 1:
        assert args.stage == "restore" and args.freeze_backbone == 0, "multi-GPU: restore stage without freezing only"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # others wait while process 0 validates
    args.world = world
    run = os.path.join(args.runs, args.name)
    os.makedirs(run, exist_ok=True)
    log = Logger(os.path.join(run, "train.log")) if main_proc else (lambda msg: None)
    if args.wd is None:
        args.wd = 0.0 if args.stage == "dcpt" else 0.01
    if main_proc:
        with open(os.path.join(run, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=1)
    log("args: " + json.dumps(vars(args)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.allow_tf32 = args.tf32  # PyTorch's default is True for convs
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
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
    ema = EMA(model, args.ema) if args.ema > 0 and main_proc else None

    data = TrainSet(counts, root=args.data, patch=args.patch, seed=args.seed)
    per_step = args.batch * args.accum * world
    log(f"train set: {len(data)} samples / epoch from {counts}; {args.iters} iters x {per_step} "
        f"= {args.iters * per_step / len(data):.2f} epochs" + (f" ({world} GPUs)" if world > 1 else ""))
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
            ema.state = {k: v.to(device) for k, v in ck["ema"].items()}  # loaded on CPU; the model is on the GPU
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
    # forward / backward go through `net`; `model` stays the plain module (state dicts, EMA, validation)
    # find_unused_parameters: the released AFLB (FreModule) has 15 tensors that never get a gradient: rate_conv
    # (the int() cast of the mask cutoff cuts it off) and conv / score_gen (defined but never called). Their grads
    # stay None, so AdamW skips them exactly as on one GPU.
    net = (DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
           if world > 1 else model)

    tb = SummaryWriter(os.path.join(run, "tb")) if main_proc else None
    metrics_f = open(os.path.join(run, "metrics.jsonl"), "a") if main_proc else None

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
            sampler = None
            if world > 1:  # each GPU takes its own share of the epoch's samples
                sampler = DistributedSampler(data, world, rank, shuffle=True, seed=args.seed * 7919, drop_last=True)
                sampler.set_epoch(epoch)
            loader = DataLoader(data, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                                num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init,
                                persistent_workers=False,  # per-process seed: different crops / noise on each GPU
                                generator=torch.Generator().manual_seed(args.seed * 7919 + epoch + 1000003 * rank))
            for b in loader:
                yield b
            epoch += 1

    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16")
    stream = batches()
    acc = {}
    t0, n_img = time.time(), 0
    model.train()
    state = {"updating": False}
    n_bad, n_dumps = 0, 0  # non-finite updates in a row; NaN-loss batches dumped by this process

    def report(msg):  # every process: other GPUs have no train.log, so print to the shared console
        log(msg) if main_proc else print(f"[rank {rank}] {msg}", flush=True)
    try:
        while step < args.iters and not stop["flag"]:
            if frozen and args.freeze_backbone > 0 and step >= args.freeze_backbone:
                frozen = False
                set_backbone_frozen(False)
            for o in optimizers:
                o.zero_grad(set_to_none=True)
            seen = []  # this step's micro-batches, reported and saved if its gradient is non-finite
            for i in range(args.accum):
                b = next(stream)
                lq, gt = b["lq"].to(device, non_blocking=True), b["gt"].to(device, non_blocking=True)
                # multi-GPU: gradients are averaged across GPUs only on the last micro-batch
                sync = net.no_sync() if world > 1 and i < args.accum - 1 else contextlib.nullcontext()
                with sync:
                    with autocast:
                        if dcpt is not None:
                            loss, parts = dcpt(lq, gt, b["label"].to(device))
                        else:
                            out = net(lq)
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
                seen.append((i, acc["loss"][-1], b["task"], lq, gt))
                if not math.isfinite(acc["loss"][-1]):
                    report(f"NaN / inf loss at step {step + 1}, micro-batch {i}: tasks {dict(Counter(b['task']))}, "
                           f"lq finite {bool(torch.isfinite(lq).all())} "
                           f"[{float(lq.min()):.3g}, {float(lq.max()):.3g}], "
                           f"gt finite {bool(torch.isfinite(gt).all())}")
            # gradients are already averaged across GPUs here, so every process takes the same decision
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            gnorm = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))) if grads else 0.0
            if not math.isfinite(gnorm):
                n_bad += 1
                bad = [n for n, p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                report(f"non-finite gradient at step {step + 1} (norm {gnorm}) in {len(bad)} tensors, e.g. {bad[:5]}; "
                       "this process's micro-batches: "
                       + "; ".join(f"{i} loss {v:.4g} {dict(Counter(t))}" for i, v, t, _, _ in seen))
                if n_dumps < 3:  # for tools/repro_nonfinite.py: the batches and the weights they failed with
                    n_dumps += 1
                    path = os.path.join(run, f"nonfinite_r{rank}_s{step + 1:07d}.pt")
                    save(path, {"step": step + 1, "variant": args.variant, "loss": args.loss,
                                "fft_weight": args.fft_weight, "fft_norm": args.fft_norm,
                                "micro": [{"i": i, "loss": v, "task": t, "lq": a.cpu(), "gt": g.cpu()}
                                          for i, v, t, a, g in seen],
                                "model": {k: v.cpu() for k, v in model.state_dict().items()} if main_proc else None})
                    report(f"saved {path}")
                if args.on_nonfinite == "stop" or n_bad > 20:
                    log(f"stopping before the update: last.pt keeps the weights of step {step}")
                    stop["flag"] = True
                    break
                log(f"update skipped ({n_bad} in a row)")
                continue
            n_bad = 0
            acc.setdefault("gnorm", []).append(gnorm)
            state["updating"] = True
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
            for o in optimizers:
                o.step()
            for s in schedulers:
                s.step()
            if ema is not None:
                ema.update(model)
            step += 1
            state["updating"] = False
            if world > 1:  # a stop signal on any GPU stops all of them at this same step
                f = torch.tensor([float(stop["flag"])], device=device)
                dist.all_reduce(f, op=dist.ReduceOp.MAX)
                stop["flag"] = bool(f.item())

            if main_proc and (step % args.log_every == 0 or step == 1):
                dt = time.time() - t0
                m = {f"train/{k}": float(np.mean(v)) for k, v in acc.items()}  # process 0's share of the batch
                m["lr"] = schedulers[0].get_last_lr()[0]
                m["img_per_s"] = n_img * world / dt
                m["epoch"] = epoch
                m.update(shape_stats(model))
                eta = (args.iters - step) * per_step / max(m["img_per_s"], 1e-6)
                log(f"step {step}/{args.iters} ep {epoch} " + " ".join(f"{k.split('/')[-1]} {v:.4g}" for k, v in m.items()
                    if k.startswith("train/")) + f" lr {m['lr']:.2e} {m['img_per_s']:.1f} img/s "
                    f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB eta {eta / 3600:.1f}h")
                write_metrics(m)
            if step % args.log_every == 0 or step == 1:
                acc, t0, n_img = {}, time.time(), 0

            validate = step % args.val_every == 0 or step == args.iters
            if validate and main_proc:
                if dcpt is not None:
                    top1, per = dcpt_accuracy(dcpt, sets, args.val_n, device)
                    log(f"val dcpt top-1 {top1:.3f} " + " ".join(f"{k} {v:.2f}" for k, v in per.items()))
                    write_metrics({"val/dcpt_top1": top1, **{f"val/dcpt_{k}": v for k, v in per.items()}})
                else:
                    ema_net = model
                    if ema is not None:
                        ema_net = build_model(args.variant, amp_safe=args.precision == "bf16").to(device)
                        ema_net.load_state_dict(ema.state)
                    log(f"val ({'ema' if ema else 'model'}, first {args.val_n} images per set):")
                    res = evaluate(ema_net, sets, val_names, device=device, limit=args.val_n, log=log)
                    write_metrics({f"val/{k}_psnr": v["psnr"] for k, v in res.items()})
                    if ema_net is not model:
                        del ema_net
                    model.train()
                t0 = time.time()

            save_now = step % args.ckpt_every == 0 or step == args.iters or stop["flag"]
            if main_proc and save_now:
                checkpoint(last)
            if main_proc and step % args.keep_every == 0:
                checkpoint(os.path.join(run, f"iter_{step:07d}.pt"))
            if world > 1 and (validate or save_now):
                dist.barrier()
                t0 = time.time()
    except RuntimeError:
        # A signal to the whole process group (e.g. a shutdown) also kills the data-loader workers, and the
        # loader raises that wherever the main process happens to be. If a stop was requested, the model is at
        # the last completed step unless an optimiser update was in progress: then keep the previous checkpoint.
        if not stop["flag"]:
            raise
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # no more dead-worker errors during the final save
        if state["updating"]:
            log(f"stopped during a parameter update; keeping the last saved checkpoint, not step {step}")
            return 3
        log(f"data workers stopped; saving step {step}")

    stream.close()  # shuts the data-loader workers down cleanly
    if main_proc:
        checkpoint(last)
        log(f"stopped at step {step}" if stop["flag"] else f"finished {step} iterations")
        metrics_f.close()
        tb.close()
    if world > 1:
        dist.destroy_process_group()
    return 3 if stop["flag"] else 0  # non-zero when interrupted, so chained stages don't continue


if __name__ == "__main__":
    sys.exit(main())
