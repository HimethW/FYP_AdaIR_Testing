#!/bin/bash
# CDD-11 run (6 classes: low, haze, rain, haze+rain, low+haze, low+rain), on qbits' GPU.
#   1. DCPT warm-up of the new frequency blocks: backbone (released AdaIR 5-task, which has seen low light)
#      frozen, 6-class degradation classifier + clean-in / clean-out L1, 2,000 iterations.
#   2. Fine-tune everything for 20 epochs with the L1 / MS-SSIM / FFT loss and the look-ahead loss-mix search,
#      EMA 0.999 with warm-up, min(0.999, (1+t)/(10+t)): a plain 0.999 is still ~80% the starting weights after
#      epoch 1 of this 4,280-step run. After every epoch the weights are kept and the EMA weights are tested on the
#      full CDD-11 test set (OneRestore protocol) -> runs/cdd6_ft/test/epoch_XX.json and runs/cdd6_ft/test/table.md
#      (vs EvoIR / OneRestore).
# Start detached:  setsid nohup bash scripts/run_cdd6.sh >> runs/cdd6.out 2>&1 &
# Running it again resumes whichever stage was interrupted (each stage resumes from its last.pt).
set -e
export PYTHONUTF8=1  # the table uses Δ; a detached shell may have an ASCII locale
source /home/minura/Himeth/env.sh
cd /home/minura/Himeth/AdaIR

if ! grep -q "finished 2000 iterations" runs/cdd6_dcpt/train.log 2>/dev/null; then
  python train_fyp.py --name cdd6_dcpt --stage dcpt --variant shaped --tasks cdd6 --iters 2000 \
    --batch 2 --accum 16 --init ckpt/adair5d.ckpt --init_backbone --freeze_backbone -1 \
    --val_every 500 --val_n 10 --ckpt_every 250
fi

python train_fyp.py --name cdd6_ft --variant shaped --tasks cdd6 --epochs 20 --init runs/cdd6_dcpt/last.pt \
  --batch 4 --accum 8 --lr 1e-4 --min_lr 1e-6 --warmup 0.02 --ema 0.999 --ema_warmup \
  --loss mix --mix_search --mix_k 20 --mix_pop 3 \
  --val_n 5 --ckpt_every 100 --keep_every 100000 \
  --test_sets cdd6 --test_tf32 --test_workers 8
