#!/bin/bash
# =============================================================================
# ArXivLens — Cross-Encoder Reranker Training Job (Sol HPC, ASU)
# =============================================================================
#
# USAGE
# -----
#   Submit a fresh run (or resume an interrupted one):
#     sbatch slurm/train_reranker.sh
#
#   The script auto-detects existing checkpoints in $CHECKPOINT_DIR and passes
#   --resume to the trainer if any are found — so re-submitting is always safe.
#
# MONITORING
# ----------
#   Stream live output while the job runs:
#     tail -f /scratch/$USER/mlrag/logs/train_<JOBID>.out
#
#   View metrics in the MLflow UI (run on a login node or forwarded port):
#     mlflow ui --backend-store-uri /scratch/$USER/mlrag/mlruns
#
# NOTES
# -----
#   - Two A100-80GB GPUs; training uses DDP via Accelerate (bf16 mixed precision)
#   - Wall-clock cap is 8 h; end-of-epoch checkpoints let you resume seamlessly
#   - TRANSFORMERS_OFFLINE=1 prevents accidental hub downloads during training
# =============================================================================

#SBATCH --job-name=arxivlens-train
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/mlrag/logs/train_%j.out
#SBATCH --error=/scratch/%u/mlrag/logs/train_%j.err

# Abort immediately on any unset variable, pipeline failure, or non-zero exit.
# This prevents silent partial failures that are hard to diagnose from HPC logs.
set -euo pipefail

# =============================================================================
# 1. Job header — printed first so the log is immediately useful for debugging
# =============================================================================
echo "============================================================"
echo "  ArXivLens reranker training"
echo "  Date     : $(date)"
echo "  Hostname : $(hostname)"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  GPUs     :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
echo "============================================================"

# =============================================================================
# 2. Environment variables — set BEFORE activating the conda env so they are
#    visible to any subprocess the env activation itself might spawn.
# =============================================================================

# Tell HuggingFace where to find pre-downloaded model weights on scratch.
# /scratch is a high-throughput Lustre filesystem; the home quota is too small
# for model caches.
export HF_HOME=/scratch/$USER/hf_cache

# Prevent AutoTokenizer / AutoModel from trying to reach the HF Hub during
# training.  All required weights must be in HF_HOME already (downloaded once
# via `huggingface-cli download` or a setup script).
export TRANSFORMERS_OFFLINE=1

# DataLoader workers fork after the tokenizer is initialized; the fast (Rust)
# tokenizer spawns its own thread pool, which deadlocks when Python's
# multiprocessing forks a process that already holds a thread-pool lock.
# Setting this to false forces the tokenizer to use a single thread, sidestepping
# the deadlock entirely.
export TOKENIZERS_PARALLELISM=false

# Tell MLflow where to write experiment metadata.  Both the trainer (--mlflow-dir)
# and this env var point to the same directory so `mlflow ui` finds everything.
export MLFLOW_TRACKING_URI=/scratch/$USER/mlrag/mlruns

# Cap the number of OpenMP threads per process.  With 2 DDP workers sharing
# 8 CPUs, each gets 4; leaving this unset lets PyTorch/OpenBLAS spawn 8 each,
# causing 16-way oversubscription and degraded throughput.
export OMP_NUM_THREADS=4

# =============================================================================
# 3. Load conda environment
# =============================================================================
module purge                        # start from a clean module state
module load mamba/latest            # makes `source activate` available

# The genai25.09 environment contains accelerate, mlflow, transformers, faiss,
# torch, etc.  Using the absolute path bypasses any .conda/environments.txt
# lookup, which is more robust on shared systems where multiple users may have
# similarly-named envs.
# Conda environment to activate. Defaults to the prebuilt genai25.09 env.
# If Sol provisions a different env, override without editing this file:
#   export ARXIVLENS_ENV=/path/to/your/env  (then sbatch as normal)
CONDA_ENV="${ARXIVLENS_ENV:-/packages/envs/genai25.09}"
# shellcheck disable=SC1091
source activate "$CONDA_ENV"

echo "[env] Conda   : $CONDA_ENV"
echo "[env] Python : $(which python)"
echo "[env] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[env] CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"

# =============================================================================
# 4. Path variables — edit these if you relocate the data or repo
# =============================================================================
REPO_DIR="$HOME/arxivlens"                 # git clone root
SCRATCH="/scratch/$USER/mlrag"             # top-level scratch directory

PAIRS_FILE=$SCRATCH/corpus/pairs.jsonl     # training pairs from build_pairs
VAL_PAIRS_FILE=$SCRATCH/corpus/val_pairs.jsonl  # held-out validation split
CHECKPOINT_DIR=$SCRATCH/checkpoints        # .pt snapshots written here
CONFIG=$REPO_DIR/configs/reranker.yaml     # hyperparameter config

# =============================================================================
# 5. Change into the repo so Python relative imports resolve correctly
# =============================================================================
cd "$REPO_DIR"

# =============================================================================
# 6. Create output directories if they don't already exist
# =============================================================================
mkdir -p "$SCRATCH/logs" "$CHECKPOINT_DIR"

# =============================================================================
# 7. Auto-detect resume
#    Glob for any checkpoint written by _save_checkpoint (pattern:
#    checkpoint_epoch{04d}_step{06d}.pt).  If at least one exists, pass
#    --resume so the trainer reloads the latest one and continues from there.
# =============================================================================
RESUME_FLAG=""
if ls "$CHECKPOINT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 | grep -q .; then
    RESUME_FLAG="--resume"
    echo "[train] Resuming from existing checkpoint in $CHECKPOINT_DIR"
else
    echo "[train] Starting fresh training run"
fi

# =============================================================================
# 8. Launch distributed training via Accelerate
#    --num_processes 2  : one process per A100 (matches --gres=gpu:a100:2)
#    --mixed_precision bf16 : A100s support bf16 natively; faster than fp16 and
#                             numerically more stable (no loss scaling needed)
#    --multi_gpu        : enables DDP across the two processes
#    -m arxivlens.train.train_reranker : invokes the training module's __main__
# =============================================================================
echo "[train] Launching accelerate at $(date)"

accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    --multi_gpu \
    -m arxivlens.train.train_reranker \
    --pairs       "$PAIRS_FILE" \
    --val-pairs   "$VAL_PAIRS_FILE" \
    --config      "$CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --mlflow-dir  "$MLFLOW_TRACKING_URI" \
    $RESUME_FLAG

# =============================================================================
# 9. Footer — report completion and locate the latest checkpoint
# =============================================================================
echo "============================================================"
echo "  Training complete: $(date)"

# Find the last checkpoint (sorted lexicographically = sorted by epoch/step
# because of the zero-padded naming scheme checkpoint_epoch{04d}_step{06d}.pt)
LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
    echo "  Latest checkpoint : $LATEST"
else
    echo "  WARNING: no checkpoint found in $CHECKPOINT_DIR"
fi

echo "  MLflow UI command :"
echo "    mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI"
echo "============================================================"
