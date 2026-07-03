#!/bin/bash
# =============================================================================
# ArXivLens — Cross-Encoder Reranker Evaluation Job (Sol HPC, ASU)
# =============================================================================
#
# USAGE
# -----
#   sbatch slurm/eval_reranker.sh
#
# REQUIREMENTS
# ------------
#   At least one checkpoint must exist in $CHECKPOINT_DIR.  Run the training
#   job first:
#     sbatch slurm/train_reranker.sh
#
# OUTPUT METRICS
# --------------
#   Prints to the .out log file:
#     nDCG@5     — normalized Discounted Cumulative Gain at rank 5
#     nDCG@10    — normalized Discounted Cumulative Gain at rank 10
#     MRR        — Mean Reciprocal Rank
#     Recall@1   — hit-rate at rank 1 (1.0 if the single relevant item is #1)
#     Recall@5   — fraction of relevant items found in the top 5
#     Recall@10  — fraction of relevant items found in the top 10
#
# MONITORING
# ----------
#   tail -f /scratch/spate472/mlrag/logs/eval_<JOBID>.out
# =============================================================================

#SBATCH --job-name=arxivlens-eval
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/spate472/mlrag/logs/eval_%j.out
#SBATCH --error=/scratch/spate472/mlrag/logs/eval_%j.err

# Abort immediately on any unset variable, pipeline failure, or non-zero exit.
set -euo pipefail

# =============================================================================
# 1. Job header
# =============================================================================
echo "============================================================"
echo "  ArXivLens reranker evaluation"
echo "  Date     : $(date)"
echo "  Hostname : $(hostname)"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  GPUs     :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
echo "============================================================"

# =============================================================================
# 2. Environment variables (same rationale as train_reranker.sh)
# =============================================================================
export HF_HOME=/scratch/spate472/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI=/scratch/spate472/mlrag/mlruns
export OMP_NUM_THREADS=4

# =============================================================================
# 3. Load conda environment
# =============================================================================
module purge
module load mamba/latest
# shellcheck disable=SC1091
source activate /packages/envs/genai25.09

echo "[env] Python : $(which python)"
echo "[env] PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# =============================================================================
# 4. Path variables
# =============================================================================
REPO_DIR=/home/spate472/arxivlens
SCRATCH=/scratch/spate472/mlrag

# Two candidate pair sources:
#   VAL_PAIRS_FILE — an explicit held-out split, if one was ever emitted.
#   PAIRS_FILE     — the full training pairs, from which the Python heredoc
#                    auto-splits the same held-out fraction the trainer uses.
# Nothing in the project is guaranteed to create val_pairs.jsonl, so we hand
# both paths to Python and let it decide (mirroring train_reranker.py's logic).
VAL_PAIRS_FILE=$SCRATCH/corpus/val_pairs.jsonl
PAIRS_FILE=$SCRATCH/corpus/pairs.jsonl
CHECKPOINT_DIR=$SCRATCH/checkpoints

# Fail fast only if NEITHER pair source exists — otherwise the Python heredoc
# below picks whichever is available (explicit val > auto-split from training).
if [ ! -f "$VAL_PAIRS_FILE" ] && [ ! -f "$PAIRS_FILE" ]; then
    echo "[eval] ERROR: neither $VAL_PAIRS_FILE nor $PAIRS_FILE exists." >&2
    echo "[eval] Build the training pairs first (scripts/build_pairs.py)." >&2
    exit 1
fi

# Export so the Python heredoc can read them from os.environ.
export REPO_DIR
export VAL_PAIRS_FILE
export PAIRS_FILE
export CHECKPOINT_DIR

# =============================================================================
# 5. Change into the repo (keeps relative imports consistent with training)
# =============================================================================
cd "$REPO_DIR"

# =============================================================================
# 6. Find the latest checkpoint
#    Checkpoints are named checkpoint_epoch{04d}_step{06d}.pt.
#    Zero-padding means lexicographic order == numeric order, so `ls -t`
#    (by mtime) is used here as a belt-and-suspenders alternative that also
#    handles the edge case where multiple jobs raced to write checkpoints in
#    the same second (mtime then falls back to the zero-padded name, which is
#    still correct).
# =============================================================================
LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)

if [ -z "$LATEST_CKPT" ]; then
    echo "[eval] ERROR: No checkpoint found in $CHECKPOINT_DIR. Train first." >&2
    exit 1
fi

echo "[eval] Using checkpoint: $LATEST_CKPT"

# Export for the Python heredoc
export LATEST_CKPT

# =============================================================================
# 7. Run evaluation inline via Python heredoc
#    The heredoc is delimited by 'PYEOF' (single-quoted) so the shell does NOT
#    expand $variables inside it — all substitution happens in Python via
#    os.environ, which is exactly what we want (Python controls the types).
# =============================================================================
echo "[eval] Starting evaluation at $(date)"

python - <<'PYEOF'
import os
import sys
import json
import torch
from pathlib import Path

# Insert repo src/ tree so arxivlens.* imports resolve without requiring an
# editable install (the genai25.09 env may not have this package installed).
repo_dir = os.environ["REPO_DIR"]
sys.path.insert(0, os.path.join(repo_dir, "src"))

from arxivlens.model.transformer import TransformerConfig
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.data.dataset import PairDataset, collate_fn
from arxivlens.train.eval import evaluate_rankings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# val_fraction / seed mirror configs/reranker.yaml (val_fraction: 0.1, seed: 42)
# so the auto-split held-out set below matches the trainer's val split exactly.
VAL_FRACTION = 0.1
SEED = 42

ckpt_path = os.environ["LATEST_CKPT"]
val_pairs = os.environ["VAL_PAIRS_FILE"]
train_pairs = os.environ.get("PAIRS_FILE", "")

print(f"[eval] Loading checkpoint: {ckpt_path}")

# Load on CPU first: avoids CUDA memory double-booking when we later call
# model.to(device).  map_location="cpu" is always safe with DDP checkpoints
# because _save_checkpoint calls accelerator.unwrap_model before saving,
# so the state dict is plain module weights without DDP prefixes.
state = torch.load(ckpt_path, map_location="cpu")
cfg = state["config"]

# Rebuild the model architecture from the config stored inside the checkpoint,
# so the eval job does not need to read reranker.yaml separately.
m = cfg["model"]
config = TransformerConfig(
    vocab_size=m["vocab_size"],
    d_model=m["d_model"],
    n_heads=m["n_heads"],
    n_layers=m["n_layers"],
    d_ff=m["d_ff"],
    max_len=m["max_len"],
    dropout=m.get("dropout", 0.1),
)

tokenizer_name = cfg["training"]["tokenizer_name"]
print(f"[eval] Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = CrossEncoderReranker(config, tokenizer=tokenizer)
model.load_state_dict(state["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[eval] Running on device: {device}")
model = model.to(device)

# Build the validation dataset from the same JSONL format used during training.
# Determine the eval set exactly as train_reranker.py does:
#   explicit val_pairs.jsonl > auto-split held-out fraction from training pairs.
max_input_length = m.get("max_input_length", 256)
if os.path.exists(val_pairs):
    # An explicit held-out split exists — use it directly (original behaviour).
    print(f"[eval] Loading val pairs from: {val_pairs}")
    dataset = PairDataset(val_pairs, tokenizer, max_length=max_input_length)
    eval_source = val_pairs
elif train_pairs and os.path.exists(train_pairs):
    # No explicit split: hold out the last VAL_FRACTION of the training pairs
    # via a seeded random_split, mirroring train_reranker.py so the eval set
    # matches the trainer's val set 1:1.
    print(f"[eval] No val_pairs file; auto-splitting from: {train_pairs}")
    full_dataset = PairDataset(train_pairs, tokenizer, max_length=max_input_length)
    n_val = max(1, int(len(full_dataset) * VAL_FRACTION))
    n_train = len(full_dataset) - n_val
    print(
        f"[eval] auto-splitting: {n_train} train / {n_val} val "
        f"(val_fraction={VAL_FRACTION}, seed={SEED})"
    )
    generator = torch.Generator().manual_seed(SEED)
    _train_dataset, dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    eval_source = f"{train_pairs} (auto-split val_fraction={VAL_FRACTION})"
else:
    # Neither source exists — the bash guard above should have caught this, but
    # fail loudly here too so a misconfigured env never yields empty metrics.
    print(
        "[eval] ERROR: no eval data — neither VAL_PAIRS_FILE nor PAIRS_FILE "
        "exists. Expected val_pairs.jsonl or pairs.jsonl at "
        "/scratch/spate472/mlrag/corpus/. Build them with scripts/build_pairs.py.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[eval] Val dataset size: {len(dataset)} pairs")

# batch_size=64 is safe on a single 80 GB A100 for max_len=256 sequences.
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

# Score every (query, passage) pair and GROUP by query_id across the whole
# loader — mirroring _run_eval in train_reranker.py — so each query becomes a
# genuine multi-candidate ranking rather than a degenerate 1-candidate query.
# The grouped (scores, labels) tuples are fed to evaluate_rankings for
# macro-averaged nDCG, MRR, and Recall metrics.
groups = {}
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        # model returns (B,) raw logits; convert to float32 on CPU for metrics.
        logits = model(ids, mask).cpu().float()
        for qid, s, l in zip(batch["query_ids"], logits.tolist(), batch["labels"].tolist()):
            bucket = groups.setdefault(qid, ([], []))
            bucket[0].append(s)
            bucket[1].append(l)

        if (batch_idx + 1) % 20 == 0:
            print(f"[eval] Scored {(batch_idx + 1) * 64} / {len(dataset)} pairs ...")

# Skip empty groups; feed one (scores, labels) tuple per query_id.
all_queries = [(s, l) for s, l in groups.values() if s]
metrics = evaluate_rankings(all_queries)

print()
print("=== ArXivLens Reranker Evaluation ===")
for k, v in metrics.items():
    print(f"  {k:15s}: {v:.4f}")
print()
print(f"Checkpoint : {ckpt_path}")
print(f"Val pairs  : {eval_source}")
print(f"N queries  : {len(all_queries)}")
PYEOF

# =============================================================================
# 8. Footer
# =============================================================================
echo "============================================================"
echo "  Evaluation complete: $(date)"
echo "============================================================"
