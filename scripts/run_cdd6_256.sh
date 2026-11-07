#!/bin/bash
# CDD-11 run, step 3: continue the 128-px fine-tune (runs/cdd6_ft, paused at step 2246 = 10.5 epochs) on 256-px
# crops, so the model sees larger areas of mixed haze (Restormer's progressive patch sizes; results.md A26 / A27).
# Same recipe otherwise (L1 / MS-SSIM / FFT loss with the look-ahead search, EMA 0.999 with warm-up); lr continues
# from where the 128-px cosine was (5e-5) down to 1e-6 over 10 epochs (10.5 + 10 = 20.5 epochs in all; first
# launched with 5, extended to 10 at step ~290 on 2026-11-08, the cosine stretched on resume); batch 1 x 32 = 32
# crops per step as before.
# After every epoch: EMA weights tested on the full CDD-11 test in 256-px tiles (the setting chosen on held-out
# training scenes, results.md A27) -> runs/cdd6_ft256/test/table.md.
# Start detached:  setsid nohup bash scripts/run_cdd6_256.sh >> runs/cdd6.out 2>&1 &   (re-running resumes)
set -e
export PYTHONUTF8=1
source /home/minura/Himeth/env.sh
cd /home/minura/Himeth/AdaIR

python train_fyp.py --name cdd6_ft256 --variant shaped --tasks cdd6 --epochs 10 --init runs/cdd6_ft/last.pt \
  --patch 256 --batch 1 --accum 32 --lr 5e-5 --min_lr 1e-6 --warmup 0.02 --ema 0.999 --ema_warmup \
  --loss mix --mix_a 0.9058 --mix_search --mix_k 20 --mix_pop 3 \
  --val_n 5 --ckpt_every 50 --keep_every 100000 \
  --test_sets cdd6 --test_tf32 --test_workers 8 --test_tile 256
