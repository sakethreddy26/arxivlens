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
export MLFLOW_TRACKING_URI="${ARXIVLENS_MLFLOW_DIR:-/scratch/$USER/mlrag/mlruns}"

# Cap the number of OpenMP threads per process.  With 2 DDP workers sharing
# 8 CPUs, each gets 4; leaving this unset lets PyTorch/OpenBLAS spawn 8 each,
# causing 16-way oversubscription and degraded throughput.
export OMP_NUM_THREADS=4

# =============================================================================
# 3. Load conda environment
# =============================================================================
# The genai25.09 environment contains accelerate, mlflow, transformers, faiss,
# torch, etc.  Using the absolute path bypasses any .conda/environments.txt
# lookup, which is more robust on shared systems where multiple users may have
# similarly-named envs.
# Conda environment to activate. Defaults to the prebuilt genai25.09 env.
# If Sol provisions a different env, override without editing this file:
#   export ARXIVLENS_ENV=/path/to/your/env  (then sbatch as normal)
CONDA_ENV="${ARXIVLENS_ENV:-/packages/envs/genai25.09}"

if [ "${ARXIVLENS_ENV_READY:-0}" = "1" ]; then
    echo "[env] Reusing environment activated by the parent workflow."
else
    module purge
    module load mamba/latest
    # shellcheck disable=SC1091
    source activate "$CONDA_ENV"
fi

if ! python3 -c 'import torch' >/dev/null 2>&1; then
    echo "[env] ERROR: PyTorch is unavailable in $(command -v python3)." >&2
    echo "[env] Expected the Sol environment at $CONDA_ENV." >&2
    exit 1
fi

echo "[env] Conda   : $CONDA_ENV"
echo "[env] Python : $(command -v python3)"
echo "[env] PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "[env] CUDA   : $(python3 -c 'import torch; print(torch.version.cuda)')"

# =============================================================================
# 4. Path variables — edit these if you relocate the data or repo
# =============================================================================
REPO_DIR="${ARXIVLENS_REPO_DIR:-$HOME/arxivlens}"
SCRATCH="${ARXIVLENS_SCRATCH:-/scratch/$USER/mlrag}"

PAIRS_FILE="${ARXIVLENS_PAIRS_FILE:-$SCRATCH/corpus/pairs.jsonl}"
VAL_PAIRS_FILE="${ARXIVLENS_VAL_PAIRS_FILE:-$SCRATCH/corpus/val_pairs.jsonl}"
CHECKPOINT_DIR="${ARXIVLENS_CHECKPOINT_DIR:-$SCRATCH/checkpoints}"
CONFIG="${ARXIVLENS_CONFIG:-$REPO_DIR/configs/reranker.yaml}"
EVAL_INDEX_PATH="${ARXIVLENS_EVAL_INDEX_PATH:-$SCRATCH/index/index.faiss}"
EVAL_META_PATH="${ARXIVLENS_EVAL_META_PATH:-$SCRATCH/index/meta.jsonl}"
EVAL_PASSAGE_FORMAT="${ARXIVLENS_EVAL_PASSAGE_FORMAT:-abstract}"

case "$EVAL_PASSAGE_FORMAT" in
    abstract|title_abstract) ;;
    *)
        echo "[train] ERROR: ARXIVLENS_EVAL_PASSAGE_FORMAT must be abstract or title_abstract." >&2
        exit 1
        ;;
esac

echo "[paths] repo        : $REPO_DIR"
echo "[paths] config      : $CONFIG"
echo "[paths] pairs       : $PAIRS_FILE"
echo "[paths] val pairs   : $VAL_PAIRS_FILE"
echo "[paths] checkpoints : $CHECKPOINT_DIR"
echo "[paths] mlflow      : $MLFLOW_TRACKING_URI"
echo "[paths] eval index  : $EVAL_INDEX_PATH"
echo "[paths] eval meta   : $EVAL_META_PATH"
echo "[paths] passage fmt : $EVAL_PASSAGE_FORMAT"

# =============================================================================
# 5. Change into the repo so Python relative imports resolve correctly
# =============================================================================
cd "$REPO_DIR"

# The arxivlens package lives under src/ (src-layout) and is NOT pip-installed,
# so put src/ on PYTHONPATH. accelerate launch's worker processes inherit this,
# which is what makes `-m arxivlens.train.train_reranker` resolve.
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

for required_file in "$CONFIG" "$PAIRS_FILE" "$EVAL_INDEX_PATH" "$EVAL_META_PATH"; do
    if [ ! -f "$required_file" ]; then
        echo "[train] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

# =============================================================================
# 6. Create output directories if they don't already exist
# =============================================================================
mkdir -p "$SCRATCH/logs" "$CHECKPOINT_DIR" "$MLFLOW_TRACKING_URI"

# =============================================================================
# 7. Auto-detect resume
#    Glob for any checkpoint written by _save_checkpoint (pattern:
#    checkpoint_epoch{04d}_step{06d}.pt).  If at least one exists, pass
#    --resume so the trainer reloads the latest one and continues from there.
# =============================================================================
RESUME_FLAG=""
RESUME_MODE="${ARXIVLENS_RESUME:-auto}"
HAS_CHECKPOINT=0
if ls "$CHECKPOINT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 | grep -q .; then
    HAS_CHECKPOINT=1
fi

case "$RESUME_MODE" in
    auto)
        if [ "$HAS_CHECKPOINT" -eq 1 ]; then
            RESUME_FLAG="--resume"
            echo "[train] Resuming from existing checkpoint in $CHECKPOINT_DIR"
        else
            echo "[train] Starting fresh training run"
        fi
        ;;
    never)
        if [ "$HAS_CHECKPOINT" -eq 1 ]; then
            echo "[train] ERROR: ARXIVLENS_RESUME=never but checkpoints exist in $CHECKPOINT_DIR" >&2
            echo "[train] Use a new checkpoint directory; existing model files are preserved." >&2
            exit 1
        fi
        echo "[train] Starting fresh training run (resume disabled)"
        ;;
    require)
        if [ "$HAS_CHECKPOINT" -eq 0 ]; then
            echo "[train] ERROR: ARXIVLENS_RESUME=require but no checkpoint exists." >&2
            exit 1
        fi
        RESUME_FLAG="--resume"
        echo "[train] Resume required; loading from $CHECKPOINT_DIR"
        ;;
    *)
        echo "[train] ERROR: ARXIVLENS_RESUME must be auto, never, or require." >&2
        exit 1
        ;;
esac

RUN_SMOKE="${ARXIVLENS_RUN_SMOKE:-0}"
case "$RUN_SMOKE" in
    1)
        echo "[train] Running CPU training smoke gate ..."
        python3 scripts/smoke_test.py
        ;;
    0)
        echo "[train] Skipping smoke gate (ARXIVLENS_RUN_SMOKE=0)"
        ;;
    *)
        echo "[train] ERROR: ARXIVLENS_RUN_SMOKE must be 0 or 1." >&2
        exit 1
        ;;
esac

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
    --eval-index-path "$EVAL_INDEX_PATH" \
    --eval-meta-path "$EVAL_META_PATH" \
    --eval-passage-format "$EVAL_PASSAGE_FORMAT" \
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
