#!/bin/bash
# One-command Sol workflow for the improved listwise reranker.
#
# Usage:
#   sbatch slurm/improve_reranker.sh
#
# The job performs three stages in order:
#   1. Build 1-positive/19-hard-negative query groups if they do not exist.
#   2. Train or resume the isolated listwise_v1 checkpoint directory.
#   3. Evaluate the latest checkpoint with its recorded passage format.
#
# If Sol's wall-time stops training, submit this same command again. Existing
# pairs are reused and training resumes from the latest listwise_v1 checkpoint.

#SBATCH --job-name=arxivlens-improve
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/mlrag/logs/improve_%j.out
#SBATCH --error=/scratch/%u/mlrag/logs/improve_%j.err

set -euo pipefail

REPO_DIR="${ARXIVLENS_REPO_DIR:-$HOME/arxivlens}"
SCRATCH="${ARXIVLENS_SCRATCH:-/scratch/$USER/mlrag}"
PAIRS_FILE="$SCRATCH/corpus/pairs_listwise_v1.jsonl"
VAL_PAIRS_FILE="$SCRATCH/corpus/val_pairs_listwise_v1.jsonl"

cd "$REPO_DIR"

module purge
module load mamba/latest
CONDA_ENV="${ARXIVLENS_ENV:-/packages/envs/genai25.09}"
# shellcheck disable=SC1091
source activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

if [ ! -f "$PAIRS_FILE" ] || [ ! -f "$VAL_PAIRS_FILE" ]; then
    TMP_PAIRS="${PAIRS_FILE}.tmp.${SLURM_JOB_ID}"
    TMP_VAL_PAIRS="${VAL_PAIRS_FILE}.tmp.${SLURM_JOB_ID}"
    trap 'rm -f "$TMP_PAIRS" "$TMP_VAL_PAIRS"' EXIT

    echo "[workflow] Building listwise hard-negative pairs ..."
    python3 scripts/build_pairs.py \
        --input "$SCRATCH/corpus/papers.jsonl" \
        --index "$SCRATCH/index" \
        --output "$TMP_PAIRS" \
        --val-output "$TMP_VAL_PAIRS" \
        --n-hard 19 \
        --n-easy 0 \
        --seed 42

    mv "$TMP_PAIRS" "$PAIRS_FILE"
    mv "$TMP_VAL_PAIRS" "$VAL_PAIRS_FILE"
    trap - EXIT
else
    echo "[workflow] Reusing existing listwise pairs."
fi

export ARXIVLENS_CONFIG="$REPO_DIR/configs/reranker_listwise.yaml"
export ARXIVLENS_PAIRS_FILE="$PAIRS_FILE"
export ARXIVLENS_VAL_PAIRS_FILE="$VAL_PAIRS_FILE"
export ARXIVLENS_CHECKPOINT_DIR="$SCRATCH/checkpoints/listwise_v1"
export ARXIVLENS_MLFLOW_DIR="$SCRATCH/mlruns/listwise_v1"
export ARXIVLENS_EVAL_INDEX_PATH="$SCRATCH/index/index.faiss"
export ARXIVLENS_EVAL_META_PATH="$SCRATCH/index/meta.jsonl"
export ARXIVLENS_EVAL_PASSAGE_FORMAT=abstract
export ARXIVLENS_RESUME=auto

echo "[workflow] Starting or resuming listwise training ..."
bash slurm/train_reranker.sh

export EVAL_PASSAGE_FORMAT=checkpoint
export EVAL_RESULTS_DIR="$SCRATCH/results/listwise_v1"
unset EVAL_CHECKPOINT

echo "[workflow] Training finished; evaluating latest checkpoint ..."
bash slurm/eval_reranker.sh

echo "[workflow] Complete. Metrics are above and in $EVAL_RESULTS_DIR."
