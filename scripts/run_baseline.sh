#!/bin/bash
# Baseline on Elgin: the original AdaIR (net/model.py exactly as released, no patch, not even the mask fix),
# fine-tuned from the full released checkpoint ckpt/adair3d.ckpt with plain L1 on aio3_mix. No DCPT, no FFT loss,
# no precautions. Same schedule as stage 2 of ours (scripts/run_ours.sh): 99,730 iters x (4 x 8) = 3,191,360 crops
# = 20.0 epochs, lr 1e-4, wd 0.01, 2% warmup, cosine to 1e-6, EMA 0.999, true fp32, seed 0.
# Decided 2026-10-19/20 (log.md "DECISION"); run on Elgin instead of qbits at the user's request (2026-10-21).
# Re-run this script to resume after an interruption (resumes from runs/baseline_ft/last.pt).
#   setsid nohup bash scripts/run_baseline.sh >> runs/baseline.out 2>&1 &
set -e
source /mnt/minura/minura/Himeth/env.sh
cd /mnt/minura/minura/Himeth/AdaIR
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # GPU 2 carries another user's desktop: never use it

python train_fyp.py --name baseline_ft --stage restore --variant original --tasks aio3_mix \
  --iters 99730 --batch 4 --accum 8 --lr 1e-4 --warmup 0.02 --min_lr 1e-6 \
  --loss l1 --ema 0.999 \
  --init ckpt/adair3d.ckpt \
  --val_every 5000 --val_n 10 --ckpt_every 1000 --keep_every 20000
