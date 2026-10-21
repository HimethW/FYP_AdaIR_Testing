#!/bin/bash
# Baseline on Elgin: the original AdaIR (net/model.py exactly as released, no patch, not even the mask fix),
# fine-tuned from the full released checkpoint ckpt/adair3d.ckpt with plain L1 on aio3_mix. No DCPT, no FFT loss,
# no precautions. Same schedule as stage 2 of ours (scripts/run_ours.sh): 99,730 iters x (4 x 8) = 3,191,360 crops
# = 20.0 epochs, lr 1e-4, wd 0.01, 2% warmup, cosine to 1e-6, EMA 0.999, true fp32, seed 0.
# Decided 2026-10-19/20 (log.md "DECISION"); run on Elgin instead of qbits at the user's request (2026-10-21).
# Re-run this script to resume after an interruption (resumes from runs/baseline_ft/last.pt).
#   setsid nohup bash scripts/run_baseline.sh >> runs/baseline.out 2>&1 &              # GPU 0
#   GPUS=0,1 setsid nohup bash scripts/run_baseline.sh >> runs/baseline.out 2>&1 &     # GPUs 0 + 1 (torchrun)
# With 2 GPUs each takes 4 x 4 crops per step: still 32 per step, the same schedule; checkpoints work either way.
set -e
source /mnt/minura/minura/Himeth/env.sh
cd /mnt/minura/minura/Himeth/AdaIR
GPUS=${GPUS:-0}  # never GPU 2: it carries another user's desktop
case ",$GPUS," in *,2,*) echo "GPU 2 is not ours to use" >&2; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPUS
n=$(awk -F, '{print NF}' <<< "$GPUS")
launch=python
[ "$n" -gt 1 ] && launch="torchrun --standalone --nproc_per_node $n"

$launch train_fyp.py --name baseline_ft --stage restore --variant original --tasks aio3_mix \
  --iters 99730 --batch 4 --accum $((8 / n)) --lr 1e-4 --warmup 0.02 --min_lr 1e-6 \
  --loss l1 --ema 0.999 \
  --init ckpt/adair3d.ckpt \
  --val_every 5000 --val_n 10 --ckpt_every 1000 --keep_every 20000
