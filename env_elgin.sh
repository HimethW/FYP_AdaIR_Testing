# Source this before working on the FYP on Elgin:  source /mnt/minura/minura/Himeth/env.sh
# (scripts/setup_elgin.sh copies this file there.) User rule, 2026-10-21: on Elgin everything of ours lives in
# /mnt/minura/minura/Himeth (the conda install, the env, every download / cache and temp file). Elgin's `/`, which
# holds /home/minura, is shared and 99% full, so HOME is pointed into Himeth too: no tool (conda's
# environments.txt, pip, matplotlib, torch, CUDA, gdown, git) can drop files into the real home folder.
export HIMETH=/mnt/minura/minura/Himeth
export HOME=$HIMETH/home
export XDG_CACHE_HOME=$HIMETH/.cache
export XDG_CONFIG_HOME=$HIMETH/home/.config
export PIP_CACHE_DIR=$HIMETH/.cache/pip
export CONDA_PKGS_DIRS=$HIMETH/.cache/conda_pkgs
export TORCH_HOME=$HIMETH/.cache/torch
export HF_HOME=$HIMETH/.cache/huggingface
export MPLCONFIGDIR=$HIMETH/.cache/matplotlib
export CUDA_CACHE_PATH=$HIMETH/.cache/nv
export TMPDIR=$HIMETH/.tmp
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR"
if [ -f "$HIMETH/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HIMETH/miniforge3/etc/profile.d/conda.sh"
  conda activate "$HIMETH/envs/adair" 2>/dev/null || echo "env not built yet: $HIMETH/envs/adair"
else
  echo "conda not installed yet: run scripts/setup_elgin.sh"
fi
cd "$HIMETH/AdaIR" 2>/dev/null || true
