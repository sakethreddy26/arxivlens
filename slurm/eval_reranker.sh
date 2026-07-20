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
# EVAL PROTOCOL
# -------------
#   Retrieve-then-rerank (matches the production pipeline): for each held-out
#   query the FAISS retriever surfaces EVAL_NUM_CANDIDATES (~50) real corpus
#   candidates, the cross-encoder reranks them, and the query's OWN paper is the
#   single positive. This desaturates recall@{5,10} (grouping a query against
#   only its ~5 synthetic pairs pins them at 1.0). Requires a FAISS index.
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
#   Plus two interpretability counts:
#     N queries              — total held-out queries scored
#     gold missed retrieval  — queries where the gold paper was NOT retrieved
#                              (the retriever recall ceiling — no reranker can
#                              recover these, so they cap every metric)
#
# MONITORING
# ----------
#   tail -f /scratch/$USER/mlrag/logs/eval_<JOBID>.out
# =============================================================================

#SBATCH --job-name=arxivlens-eval
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/mlrag/logs/eval_%j.out
#SBATCH --error=/scratch/%u/mlrag/logs/eval_%j.err

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
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI=/scratch/$USER/mlrag/mlruns
export OMP_NUM_THREADS=4

# =============================================================================
# 3. Load conda environment
# =============================================================================
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

# =============================================================================
# 4. Path variables
# =============================================================================
REPO_DIR="${ARXIVLENS_REPO_DIR:-$HOME/arxivlens}"
SCRATCH="${ARXIVLENS_SCRATCH:-/scratch/$USER/mlrag}"

# Two candidate pair sources:
#   VAL_PAIRS_FILE — an explicit held-out split, if one was ever emitted.
#   PAIRS_FILE     — the full training pairs, from which the Python heredoc
#                    auto-splits the same held-out fraction the trainer uses.
# Nothing in the project is guaranteed to create val_pairs.jsonl, so we hand
# both paths to Python and let it decide (mirroring train_reranker.py's logic).
VAL_PAIRS_FILE="${ARXIVLENS_VAL_PAIRS_FILE:-$SCRATCH/corpus/val_pairs.jsonl}"
PAIRS_FILE="${ARXIVLENS_PAIRS_FILE:-$SCRATCH/corpus/pairs.jsonl}"
CHECKPOINT_DIR="${ARXIVLENS_CHECKPOINT_DIR:-$SCRATCH/checkpoints}"

# Retrieve-then-rerank eval config.
#   EVAL_NUM_CANDIDATES — candidates the FAISS retriever surfaces per query
#                         before the cross-encoder reranks them. ~50 is the
#                         eval breadth that DESATURATES recall@{5,10} (grouping
#                         a query against only its ~5 synthetic pairs pins them
#                         at 1.0).
#   EVAL_INDEX_PATH     — FAISS FlatIP index over the corpus (built offline).
#   EVAL_META_PATH      — meta.jsonl in FAISS row order (paper_id/title/abstract).
# The index/meta live under the Sol scratch layout /scratch/$USER/mlrag/index/
# (same tree as $SCRATCH above); override any of these before sbatch if needed.
EVAL_NUM_CANDIDATES="${EVAL_NUM_CANDIDATES:-50}"
EVAL_INDEX_PATH="${EVAL_INDEX_PATH:-$SCRATCH/index/index.faiss}"
EVAL_META_PATH="${EVAL_META_PATH:-$SCRATCH/index/meta.jsonl}"
EVAL_PASSAGE_FORMAT="${EVAL_PASSAGE_FORMAT:-checkpoint}"

case "$EVAL_PASSAGE_FORMAT" in
    checkpoint|abstract|title_abstract) ;;
    *)
        echo "[eval] ERROR: EVAL_PASSAGE_FORMAT must be checkpoint, abstract, or title_abstract." >&2
        exit 1
        ;;
esac

# Durable provenance artifact directory. The Python heredoc writes one JSON per
# job here (results/eval_<SLURM_JOB_ID>.json) recording exactly which run/
# checkpoint produced which numbers, so README/MODEL_CARD figures stay traceable.
# Lives under the same Sol scratch tree ($SCRATCH) as the index/checkpoints.
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$SCRATCH/results}"

# Fail fast only if NEITHER pair source exists — otherwise the Python heredoc
# below picks whichever is available (explicit val > auto-split from training).
if [ ! -f "$VAL_PAIRS_FILE" ] && [ ! -f "$PAIRS_FILE" ]; then
    echo "[eval] ERROR: neither $VAL_PAIRS_FILE nor $PAIRS_FILE exists." >&2
    echo "[eval] Build the training pairs first (scripts/build_pairs.py)." >&2
    exit 1
fi

# The retrieve-then-rerank harness cannot run without a FAISS index + meta.
# Fail loudly here (rather than letting Python fall back to a degenerate,
# saturated group-by-query_id eval) so the user knows to build the index first.
if [ ! -f "$EVAL_INDEX_PATH" ]; then
    echo "[eval] ERROR: FAISS index not found at $EVAL_INDEX_PATH." >&2
    echo "[eval] Build the retrieval index first, or set EVAL_INDEX_PATH." >&2
    exit 1
fi
if [ ! -f "$EVAL_META_PATH" ]; then
    echo "[eval] ERROR: retrieval meta not found at $EVAL_META_PATH." >&2
    echo "[eval] Build the retrieval index first, or set EVAL_META_PATH." >&2
    exit 1
fi

# Export so the Python heredoc can read them from os.environ.
export REPO_DIR
export VAL_PAIRS_FILE
export PAIRS_FILE
export CHECKPOINT_DIR
export EVAL_NUM_CANDIDATES
export EVAL_INDEX_PATH
export EVAL_META_PATH
export EVAL_RESULTS_DIR
export EVAL_PASSAGE_FORMAT

# =============================================================================
# 5. Change into the repo (keeps relative imports consistent with training)
# =============================================================================
cd "$REPO_DIR"

# The arxivlens package lives under src/ (src-layout) and is NOT pip-installed,
# so put src/ on PYTHONPATH. accelerate launch's worker processes inherit this,
# which is what makes `-m arxivlens.train.train_reranker` resolve.
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# =============================================================================
# 6. Find the latest checkpoint
#    Checkpoints are named checkpoint_epoch{04d}_step{06d}.pt.
#    Zero-padding means lexicographic order == numeric order, so `ls -t`
#    (by mtime) is used here as a belt-and-suspenders alternative that also
#    handles the edge case where multiple jobs raced to write checkpoints in
#    the same second (mtime then falls back to the zero-padded name, which is
#    still correct).
# =============================================================================
if [ -n "${EVAL_CHECKPOINT:-}" ]; then
    LATEST_CKPT="$EVAL_CHECKPOINT"
else
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
fi

if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
    echo "[eval] ERROR: checkpoint not found: ${LATEST_CKPT:-<none>}" >&2
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

python3 - <<'PYEOF'
import os
import sys
import json
import datetime
import torch
from pathlib import Path

# Insert repo src/ tree so arxivlens.* imports resolve without requiring an
# editable install (the genai25.09 env may not have this package installed).
repo_dir = os.environ["REPO_DIR"]
sys.path.insert(0, os.path.join(repo_dir, "src"))

from arxivlens.model.transformer import TransformerConfig
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.data.dataset import PairDataset, group_split_indices
from arxivlens.retrieve.index import FaissRetriever
from arxivlens.train.eval import (
    build_retrieval_eval_queries,
    evaluate_rankings,
    rank_diagnostics,
)
from transformers import AutoTokenizer

ckpt_path = os.environ["LATEST_CKPT"]
val_pairs = os.environ["VAL_PAIRS_FILE"]
train_pairs = os.environ.get("PAIRS_FILE", "")
num_candidates = int(os.environ["EVAL_NUM_CANDIDATES"])
index_path = os.environ["EVAL_INDEX_PATH"]
meta_path = os.environ["EVAL_META_PATH"]
requested_passage_format = os.environ["EVAL_PASSAGE_FORMAT"]

print(f"[eval] Loading checkpoint: {ckpt_path}")

# Load on CPU first: avoids CUDA memory double-booking when we later call
# model.to(device).  map_location="cpu" is always safe with DDP checkpoints
# because _save_checkpoint calls accelerator.unwrap_model before saving,
# so the state dict is plain module weights without DDP prefixes.
state = torch.load(ckpt_path, map_location="cpu")
cfg = state["config"]
passage_format = requested_passage_format
if passage_format == "checkpoint":
    passage_format = str(
        cfg.get("training", {}).get("eval_passage_format", "title_abstract")
    )

# val_fraction / seed are read from the checkpoint config so the auto-split
# held-out set below matches the trainer's actual val split exactly, falling
# back to the configs/reranker.yaml defaults (val_fraction: 0.1, seed: 42) only
# if the checkpoint predates recording them. Only the auto-split branch uses
# these; the explicit val_pairs.jsonl branch is unaffected.
VAL_FRACTION = float(cfg.get("training", {}).get("val_fraction", 0.1))
SEED = int(cfg.get("training", {}).get("seed", 42))

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

# -------------------------------------------------------------------------- #
# Determine the held-out eval set exactly as train_reranker.py does:
#   explicit val_pairs.jsonl > auto-split held-out fraction from training pairs.
# We reconstruct one eval RECORD per held-out query group from the label==1
# pair (the schema built by data/pairs.py):
#   id       <- query_id (the paper's real arXiv id)
#   title    <- query   (the paper's title, used as the retrieval query)
#   abstract <- passage (the positive pair's passage == the paper's own abstract)
# These records feed build_retrieval_eval_queries, which retrieves real corpus
# candidates and reranks them — the desaturated retrieve-then-rerank protocol,
# NOT the degenerate group-by-query_id scoring this script used to do.
# -------------------------------------------------------------------------- #
max_input_length = m.get("max_input_length", 256)


def _read_pairs(path):
    """Load a pairs.jsonl file as a list of dicts (query_id, query, passage, label)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _records_from_pairs(pairs):
    """Reconstruct held-out eval records from the label==1 pair of each query group.

    One record per query_id, in first-seen order. The positive pair carries the
    paper's own abstract (data/pairs.py), so it is the authoritative source for
    the abstract; negatives are ignored here (they are re-derived by retrieval).

    Records whose reconstructed qid is a bare positional index (qid.isdigit())
    are DROPPED, not kept: build_retrieval_eval_queries raises ValueError on such
    ids because a digit-only id means the real arXiv id was lost during
    reconstruction and label-matching against retrieved paper_id would be
    meaningless. Dropping them (and counting them) keeps one malformed group from
    aborting the whole eval job. Returns (records, n_skipped).
    """
    records = {}
    order = []
    n_skipped = 0
    for row in pairs:
        if int(row.get("label", 0)) != 1:
            continue
        qid = str(row.get("query_id"))
        if qid in records:
            continue  # first positive per group wins
        if qid.isdigit():
            # Unscoreable: real id lost during reconstruction. Skip, don't crash.
            n_skipped += 1
            continue
        records[qid] = {
            "id": qid,
            "title": row.get("query", ""),
            "abstract": row.get("passage", ""),
        }
        order.append(qid)
    return [records[qid] for qid in order], n_skipped


if os.path.exists(val_pairs):
    # An explicit held-out split exists — use every query group in it directly.
    print(f"[eval] Loading val pairs from: {val_pairs}")
    eval_records, n_skipped_positional = _records_from_pairs(_read_pairs(val_pairs))
    eval_source = val_pairs
elif train_pairs and os.path.exists(train_pairs):
    # No explicit split: hold out VAL_FRACTION of the QUERY GROUPS via the
    # shared seeded group_split_indices helper, mirroring train_reranker.py so
    # the eval set matches the trainer's val set 1:1 (whole query groups, never
    # orphaned candidates). We reuse PairDataset only for its query_ids() view;
    # the records themselves are reconstructed from the raw pairs.
    print(f"[eval] No val_pairs file; auto-splitting from: {train_pairs}")
    full_dataset = PairDataset(train_pairs, tokenizer, max_length=max_input_length)
    _train_idx, val_idx = group_split_indices(
        full_dataset.query_ids(), VAL_FRACTION, SEED
    )
    all_pairs = _read_pairs(train_pairs)
    val_pairs_rows = [all_pairs[i] for i in val_idx]
    eval_records, n_skipped_positional = _records_from_pairs(val_pairs_rows)
    print(
        f"[eval] auto-splitting by query group: "
        f"{len(full_dataset) - len(val_idx)} train / {len(val_idx)} val pairs "
        f"(val_fraction={VAL_FRACTION} of query groups, seed={SEED})"
    )
    eval_source = f"{train_pairs} (auto-split val_fraction={VAL_FRACTION})"
else:
    # Neither source exists — the bash guard above should have caught this, but
    # fail loudly here too so a misconfigured env never yields empty metrics.
    print(
        "[eval] ERROR: no eval data — neither VAL_PAIRS_FILE nor PAIRS_FILE "
        f"exists. Expected {val_pairs} or {train_pairs}. "
        "Build them with scripts/build_pairs.py.",
        file=sys.stderr,
    )
    sys.exit(1)

if not eval_records:
    print(
        "[eval] ERROR: reconstructed 0 held-out eval records (no label==1 pairs "
        f"in {eval_source}). Regenerate pairs with scripts/build_pairs.py.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[eval] Held-out eval records: {len(eval_records)} queries")

# Build the FAISS retriever over the corpus. Existence of index/meta was
# guarded in bash; construction still fails loudly if faiss/meta are malformed.
print(f"[eval] Loading FAISS index: {index_path}")
print(f"[eval] Loading retrieval meta: {meta_path}")
retriever = FaissRetriever(index_path=index_path, meta_path=meta_path)

# Retrieve-then-rerank: for each held-out query, pull num_candidates real
# candidates from FAISS, rerank them with the cross-encoder, and label the
# query's own paper as the single positive. This is the desaturated protocol.
print(f"[eval] Retrieving {num_candidates} candidates/query and reranking ...")
print(f"[eval] Passage format: {passage_format}")
reranker_queries, retrieval_queries = build_retrieval_eval_queries(
    eval_records=eval_records,
    retriever=retriever,
    reranker=model,
    num_candidates=num_candidates,
    with_retrieval_baseline=True,
    passage_format=passage_format,
)

# Reranker metrics (existing var name kept) plus a retrieval-only baseline over
# the IDENTICAL candidate sets, so the from-scratch cross-encoder has a
# head-to-head comparison (closes the G5 retrieval-only-baseline blocker).
metrics = evaluate_rankings(reranker_queries)
baseline_metrics = evaluate_rankings(retrieval_queries)
diagnostics = rank_diagnostics(reranker_queries, retrieval_queries)
for row, record in zip(diagnostics["per_query"], eval_records):
    row["query_id"] = str(record.get("id", ""))
    row["title"] = str(record.get("title", ""))
diag_summary = diagnostics["summary"]
hard_metrics = diagnostics["hard_metrics"]

# Interpretability: a query whose gold paper never made it into the retrieved
# candidate set has an all-zero label vector — no reranker can recover it, so it
# caps every metric. Counting these separates reranker quality from the
# retriever recall ceiling.
n_scored = sum(1 for s, _ in reranker_queries if s)
n_missed = sum(1 for s, l in reranker_queries if s and 1.0 not in l)

print()
print("=== ArXivLens Reranker Evaluation (retrieve-then-rerank) ===")
print(f"{'metric':17s} {'retrieval-only':16s} {'+ reranker'}")
for k in metrics:
    print(f"{k:17s} {baseline_metrics[k]:<16.4f} {metrics[k]:.4f}")
print()
print("=== Hard-only subset (FAISS gold rank > 1) ===")
print(f"{'metric':17s} {'retrieval-only':16s} {'+ reranker'}")
for k in hard_metrics["reranker"]:
    print(
        f"{k:17s} "
        f"{hard_metrics['retrieval_only'][k]:<16.4f} "
        f"{hard_metrics['reranker'][k]:.4f}"
    )
print()
print("=== Rank movement diagnostics ===")
print(f"Hard queries        : {diag_summary['n_hard_queries']}")
print(f"Improved            : {diag_summary['improved']}")
print(f"Same                : {diag_summary['same']}")
print(f"Worsened            : {diag_summary['worsened']}")
print(f"Mean rank delta     : {diag_summary['mean_rank_delta']:.4f}")
print(f"Median rank delta   : {diag_summary['median_rank_delta']:.4f}")
print()
print(f"Checkpoint          : {ckpt_path}")
print(f"Eval source         : {eval_source}")
print(f"Index / meta        : {index_path} | {meta_path}")
print(f"Candidates/query    : {num_candidates}")
print(f"Passage format      : {passage_format}")
print(f"N queries           : {n_scored}")
print(f"Gold missed retrieval: {n_missed}  (retriever recall ceiling)")
print(
    f"Skipped (positional id): {n_skipped_positional}  "
    "(unscoreable — real id lost in reconstruction, dropped not scored)"
)

# -------------------------------------------------------------------------- #
# Durable provenance artifact.
# Everything above prints to the .out log only, which is ephemeral and easy to
# mis-transcribe into README/MODEL_CARD. Write one JSON file per job recording
# EXACTLY which run/checkpoint produced which numbers, so published figures stay
# traceable and can never be silently mismatched. Purely additive — all stdout
# above is preserved.
# -------------------------------------------------------------------------- #
job_id = os.environ.get("SLURM_JOB_ID", "")
results_dir = os.environ.get("EVAL_RESULTS_DIR", "results")
os.makedirs(results_dir, exist_ok=True)
# Fall back to "local" when run outside slurm so this never crashes / collides.
out_path = os.path.join(results_dir, f"eval_{job_id or 'local'}.json")

report = {
    "checkpoint": ckpt_path,
    "slurm_job_id": job_id,
    "eval_source": eval_source,
    "index_path": index_path,
    "meta_path": meta_path,
    "num_candidates": num_candidates,
    "passage_format": passage_format,
    "n_queries": n_scored,
    "gold_missed_retrieval": n_missed,
    "skipped_positional_id": n_skipped_positional,
    "metrics": {
        "reranker": metrics,
        "retrieval_only": baseline_metrics,
    },
    "rank_diagnostics": diagnostics,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)

print(f"[eval] Wrote results artifact: {out_path}")
PYEOF

# =============================================================================
# 8. Footer
# =============================================================================
echo "============================================================"
echo "  Evaluation complete: $(date)"
echo "============================================================"
