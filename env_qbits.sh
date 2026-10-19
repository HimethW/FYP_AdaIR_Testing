# Source this before working on the FYP on qbits:  source /home/minura/Himeth/env.sh
# Keeps everything inside /home/minura/Himeth: the Python environment, and every download / cache
# (pip, conda packages, torch, matplotlib, Hugging Face, gdown, CUDA kernel cache, temp files).
export HIMETH=/home/minura/Himeth
export XDG_CACHE_HOME=$HIMETH/.cache
export PIP_CACHE_DIR=$HIMETH/.cache/pip
export CONDA_PKGS_DIRS=$HIMETH/.cache/conda_pkgs
export TORCH_HOME=$HIMETH/.cache/torch
export HF_HOME=$HIMETH/.cache/huggingface
export MPLCONFIGDIR=$HIMETH/.cache/matplotlib
export CUDA_CACHE_PATH=$HIMETH/.cache/nv
export TMPDIR=$HIMETH/.tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
source /home/minura/miniforge3/etc/profile.d/conda.sh
conda activate "$HIMETH/envs/adair" 2>/dev/null || echo "env not built yet: $HIMETH/envs/adair"
cd "$HIMETH/AdaIR" 2>/dev/null || true
