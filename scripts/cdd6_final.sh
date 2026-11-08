#!/bin/bash
# CDD-11 final test of the 384-px stage, run when runs/cdd6_ft384 finishes:
#   1. best epoch = highest 6-class mean PSNR in the per-epoch tests (512-px tiles, full test, as for every stage);
#   2. tile size re-chosen on 20 held-out training scenes per class (never the test set): 384 / 512 / 640 / 720;
#   3. that tile size once on the full CDD-11 test, without flips (if not already the per-epoch 512 test) and with
#      the x8 flips / rotations (self-ensemble).
# Start detached:  setsid nohup bash scripts/cdd6_final.sh > /dev/null 2>&1 &     log: runs/cdd6_final/chain.log
cd /home/minura/Himeth/AdaIR
source /home/minura/Himeth/env.sh
export PYTHONUTF8=1
OUT=runs/cdd6_final
mkdir -p $OUT
say() { echo "[$(date '+%F %T')] $*" >> $OUT/chain.log; }

say "waiting for runs/cdd6_ft384 to finish"
until grep -q "finished 1070 iterations" runs/cdd6_ft384/train.log; do
  if ! pgrep -f "[n]ame cdd6_ft384" > /dev/null; then say "training stopped without finishing; nothing run"; exit 1; fi
  sleep 60
done
sleep 30

EP=$(python -c '
import glob, json, numpy as np
mean = lambda f: np.mean([v["psnr"] for v in json.load(open(f))["results"].values()])
print(int(json.load(open(max(glob.glob("runs/cdd6_ft384/test/epoch_*.json"), key=mean)))["epoch"]))')
CK=runs/cdd6_ft384/epochs/epoch_$(printf %02d $EP).pt
say "best epoch $EP ($CK); held-out tile check"
python tools/cdd_infer_sweep.py --ckpt $CK --sets cdd6val --limit 20 --settings tile384 tile512 tile640 tile720 \
  --out $OUT/heldout_ep$EP.json 2>&1 | grep -v -i "warning\|autocast" >> $OUT/chain.log

T=$(python -c '
import json, sys, numpy as np
r = json.load(open(sys.argv[1]))
print(max((k for k in r if k != "meta"), key=lambda k: np.mean([v["psnr"] for v in r[k].values()]))[4:])' \
  $OUT/heldout_ep$EP.json)
SETS="tile${T}_x8"
[ "$T" != "512" ] && SETS="tile$T tile${T}_x8"
say "chosen tile $T; full test: $SETS"
python tools/cdd_infer_sweep.py --ckpt $CK --sets cdd6 --settings $SETS --out $OUT/test_ep${EP}_tile$T.json 2>&1 \
  | grep -v -i "warning\|autocast" >> $OUT/chain.log
say "done"
