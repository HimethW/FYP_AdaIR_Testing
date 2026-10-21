#!/bin/bash
# One-time setup of Elgin for the baseline run, entirely inside /mnt/minura/minura/Himeth (user rule, 2026-10-21).
# Resumable: every step checks its own output, so re-run it after an interruption.
#   git clone https://github.com/HimethW/FYP_AdaIR_Testing.git /mnt/minura/minura/Himeth/AdaIR
#   setsid nohup bash /mnt/minura/minura/Himeth/AdaIR/scripts/setup_elgin.sh \
#       >> /mnt/minura/minura/Himeth/setup_elgin.log 2>&1 &
# Steps: 1 Miniforge + the adair env (same pins as qbits, requirements-fyp.txt), 2 datasets from the original
# sources (tools/prepare_data.py), 3 fixed mixture test sets (tools/build_mixtures.py, seed 0), 4 released
# checkpoints (README's Drive folder; adair3d.ckpt checked by sha256), 5 data fingerprint compared with qbits'
# (splits/data_digest_qbits.txt), so the baseline sees exactly the same files as ours.
set -e
H=/mnt/minura/minura/Himeth
REPO=$H/AdaIR
log() { echo "[$(date '+%F %T')] $*"; }

cp "$REPO/env_elgin.sh" "$H/env.sh"
source "$H/env.sh"

# 1. Miniforge (batch mode: does not touch .bashrc) + env
if [ ! -x "$H/miniforge3/bin/conda" ]; then
  log "installing Miniforge into $H/miniforge3"
  curl -fsSL -o "$TMPDIR/Miniforge3.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$TMPDIR/Miniforge3.sh" -b -p "$H/miniforge3"
  rm -f "$TMPDIR/Miniforge3.sh"
fi
source "$H/miniforge3/etc/profile.d/conda.sh"
if [ ! -x "$H/envs/adair/bin/python" ]; then
  log "creating env $H/envs/adair"
  conda create -y -p "$H/envs/adair" python=3.10 unzip  # unzip: tools/prepare_data.py calls it; Elgin has none
fi
conda activate "$H/envs/adair"
if ! python -c "import torch, gdown, skimage, cv2" 2>/dev/null; then
  log "pip install -r requirements-fyp.txt"
  pip install -r "$REPO/requirements-fyp.txt"
fi
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', \
torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0)); \
x = torch.randn(256, 256, device='cuda'); print('gpu ok', float((x @ x).abs().sum()) > 0)"
cd "$REPO"

# 2. datasets (resumable; downloads go to data/_downloads, zips are deleted after extracting)
log "tools/prepare_data.py"
python tools/prepare_data.py

# 3. fixed mixture test sets
if [ ! -f data/test/mix/manifest.json ]; then
  log "tools/build_mixtures.py"
  python tools/build_mixtures.py
fi

# 4. released checkpoints
want=f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb
if [ "$(sha256sum ckpt/adair3d.ckpt 2>/dev/null | cut -c1-64)" != "$want" ]; then
  log "downloading the released checkpoints"
  gdown --folder https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH -O ckpt
  [ "$(sha256sum ckpt/adair3d.ckpt | cut -c1-64)" = "$want" ] || { log "adair3d.ckpt sha256 MISMATCH"; exit 1; }
fi
log "adair3d.ckpt sha256 ok"

# 5. data fingerprint vs qbits
log "data fingerprint"
bash tools/data_digest.sh data > "$H/data_digest_elgin.txt"
if diff splits/data_digest_qbits.txt "$H/data_digest_elgin.txt"; then
  log "data identical to qbits"
else
  log "DATA DIFFERS from qbits (diff above)"
  exit 1
fi
log "setup done"
