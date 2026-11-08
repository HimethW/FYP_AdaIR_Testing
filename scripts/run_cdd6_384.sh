#!/bin/bash
# CDD-11 run, step 4: continue the 256-px stage (runs/cdd6_ft256, finished: 10.5 epochs at 128 px + 10 at 256 px)
# on 384-px crops, so the model sees even larger areas of each scene (Restormer's progressive patch sizes; the
# 256-px stage added +0.53 dB, mostly on the haze classes, results.md A28).
# Same recipe otherwise (L1 / MS-SSIM / FFT loss with the look-ahead search, starting from the stage's last a; EMA
# 0.999 with warm-up); a fresh cosine from 3e-5 down to 1e-6 over 5 epochs; batch 1 x 32 = 32 crops per step as before.
# --grad_ckpt: the transformer blocks are recomputed in the backward pass (same gradients, ~30% more compute), so
# 384-px crops fit on the 16 GB GPU.
# After every epoch: EMA weights tested on the full CDD-11 test in TILE-px tiles (512: best on held-out scenes for
# the 256-px model, results.md A29) -> runs/cdd6_ft384/test/table.md.
# Start detached:  setsid nohup bash scripts/run_cdd6_384.sh >> runs/cdd6.out 2>&1 &   (re-running resumes)
set -e
export PYTHONUTF8=1
source /home/minura/Himeth/env.sh
cd /home/minura/Himeth/AdaIR
TILE=512

python train_fyp.py --name cdd6_ft384 --variant shaped --tasks cdd6 --epochs 5 --init runs/cdd6_ft256/last.pt \
  --patch 384 --batch 1 --accum 32 --grad_ckpt --lr 3e-5 --min_lr 1e-6 --warmup 0.02 --ema 0.999 --ema_warmup \
  --loss mix --mix_a 0.816 --mix_search --mix_k 20 --mix_pop 3 \
  --val_n 5 --ckpt_every 50 --keep_every 100000 \
  --test_sets cdd6 --test_tf32 --test_workers 8 --test_tile $TILE
