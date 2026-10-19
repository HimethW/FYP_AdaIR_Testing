#!/bin/bash
# Our model on qbits: stage 1 = DCPT warm-up with the backbone frozen, stage 2 = full restoration fine-tuning.
# Decided 2026-10-19/20 (log.md "DECISION"). Re-run this script to resume after an interruption: each stage
# resumes from runs/<name>/last.pt, a finished stage returns immediately, and stage 2 only starts after stage 1
# finished (an interrupted stage exits with code 3).
#   setsid nohup bash scripts/run_ours.sh >> runs/ours.out 2>&1 &
set -e
source /home/minura/Himeth/env.sh
cd /home/minura/Himeth/AdaIR

# Stage 1: 1 epoch of aio3_mix (5,000 x 32 = 160,000 crops). Released backbone (+ para2), fresh AFLBs,
# backbone frozen for the whole stage: only the new AFLBs and the DCPT classifier train.
python train_fyp.py --name ours_dcpt --stage dcpt --variant shaped --tasks aio3_mix \
  --iters 5000 --batch 2 --accum 16 --lr 2e-4 \
  --init ckpt/adair3d.ckpt --init_backbone --freeze_backbone -1 \
  --val_every 500 --val_n 10 --ckpt_every 250 --keep_every 100000

# Stage 2: 20 epochs of aio3_mix (99,730 x 32 = 3,191,360 crops = 20.0 x 159,567), everything trains,
# L1 + 0.1 FFT, lr 1e-4, 2% warmup, cosine to 1e-6, EMA 0.999 kept alongside, true fp32.
python train_fyp.py --name ours_ft --stage restore --variant shaped --tasks aio3_mix \
  --iters 99730 --batch 4 --accum 8 --lr 1e-4 --warmup 0.02 --min_lr 1e-6 \
  --loss l1_fft --fft_weight 0.1 --fft_norm backward --ema 0.999 \
  --init runs/ours_dcpt/last.pt \
  --val_every 5000 --val_n 10 --ckpt_every 1000 --keep_every 20000
