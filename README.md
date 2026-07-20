# ArXivLens

[![CI](https://github.com/sakethreddy26/ArXivLens/actions/workflows/ci.yml/badge.svg)](https://github.com/sakethreddy26/ArXivLens/actions/workflows/ci.yml)

Semantic search over ArXiv ML papers — dense retrieval with a FAISS bi-encoder index, reranked by a from-scratch cross-encoder transformer, with an attention "lens" into why a paper ranked where it did.

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────┐
│  Bi-Encoder (sentence-transformers)│
│  Encode query → 384-d vector      │
└─────────────────┬───────────────┘
                  │  cosine search
                  ▼
┌─────────────────────────────────┐
│  FAISS Index (flat IP / IVF)    │
│  ~100 k–1 M paper abstracts     │
│  → top-K candidates (K = 50)   │
└─────────────────┬───────────────┘
                  │ (query, passage) pairs
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Cross-Encoder Reranker  (from scratch)                  │
│  [CLS] query [SEP] passage [SEP]                        │
│  → TransformerEncoder (6 layers, 8 heads, d_model=512) │
│  → [CLS] hidden → Linear(512,1) → relevance logit      │
└────────────────────────────┬────────────────────────────┘
                             │  ranked results + attention weights
                             ▼
                     FastAPI  /search  /explain
                     (attention heatmap via /explain)
```

**Stage 1 — Dense retrieval.** `sentence-transformers/all-MiniLM-L6-v2` encodes every abstract offline into a 384-d vector stored in a FAISS flat-IP index. At query time the query is embedded with the same model and the top-K (default 50) nearest neighbours are returned in milliseconds — no cross-attention, pure dot-product similarity.

**Stage 2 — Cross-encoder reranking.** Each of the K candidates is paired with the query as `[CLS] query [SEP] passage [SEP]` and fed through a from-scratch `TransformerEncoder` (6 layers, 8 heads, d_model=512, d_ff=2048, max_len=512, max_input_length=256). The `[CLS]` hidden state is projected to a scalar relevance logit; candidates are re-sorted by that logit. Because query and passage tokens attend to each other jointly, the reranker captures fine-grained term interactions the bi-encoder misses.

**Stage 3 — Attention lens.** The same forward pass that scores a pair also returns per-layer, per-head attention weights. `POST /explain` slices the query-token × passage-token sub-block and averages over layers and heads to produce a 2-D heatmap. See `notebooks/attention_demo.ipynb` for an interactive version.

---

## Bi-encoder vs. Cross-encoder

| Property | Bi-encoder (FAISS stage) | Cross-encoder (reranker) |
|---|---|---|
| Query sees passage? | No — encoded independently | Yes — joint attention |
| Latency | Sub-millisecond (dot product) | O(K) forward passes |
| Scalable to corpus? | Yes — vectors precomputed | No — must run per pair |
| Ranking quality | Good recall, coarser ranking | Higher precision |
| Role in pipeline | Retrieve top-K candidates | Rerank the K candidates |

Neither alone is sufficient: the bi-encoder cannot afford to look at every paper deeply; the cross-encoder cannot afford to look at every paper at all. Retrieve-then-rerank gets the speed of the first and the accuracy of the second.

---

## Quick start — Docker

```bash
docker build -t arxivlens .
docker run -p 8000:8000 \
  -e INDEX_PATH=/data/index.faiss \
  -e META_PATH=/data/meta.jsonl \
  -e CHECKPOINT=/data/checkpoint.pt \
  -v /path/to/your/data:/data \
  arxivlens
# Then: curl http://localhost:8000/health
```

`INDEX_PATH` and `META_PATH` are required; `CHECKPOINT` is optional (the service runs with an untrained reranker if omitted). When a `CHECKPOINT` is supplied, the served reranker's architecture is read from the config stored inside the checkpoint — not a hardcoded arch — so any trained checkpoint loads without shape mismatch. Additional environment variables:

| Variable | Default | Description |
|---|---|---|
| `TOKENIZER` | `bert-base-uncased` | HuggingFace tokenizer name |
| `RETRIEVE_K` | `50` | FAISS candidate cap before reranking |

The Docker image uses CPU-only PyTorch. For production throughput, run on a GPU host with the full requirements installed directly (see `requirements.txt`).

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; returns retriever and reranker class names |
| `POST` | `/search` | Retrieve and rerank papers for a free-text query |
| `POST` | `/explain` | Return cross-encoder attention weights for a (query, paper_id) pair |

Interactive docs are available at `http://localhost:8000/docs` once the service is running.

**Search example:**

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms in transformers", "top_n": 5}' | python -m json.tool
```

Response shape:

```json
{
  "query": "attention mechanisms in transformers",
  "retrieval_count": 50,
  "results": [
    {
      "paper_id": "1706.03762",
      "title": "Attention Is All You Need",
      "abstract": "...",
      "score": 3.142,
      "rank": 1
    }
  ]
}
```

`score` is a raw cross-encoder logit (higher = more relevant). `rank` is 1-based.

**Explain example:**

```bash
curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms in transformers", "paper_id": "1706.03762"}' \
  | python -m json.tool
```

Returns `attention_weights` — a `query_tokens × passage_tokens` matrix averaged over all layers and heads — along with `tokens` (token strings for axis labels) and the `score` logit.

---

## Results

Head-to-head on the same held-out queries, 50 candidates per query:

| Stage | nDCG@5 | nDCG@10 | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| FAISS retrieval-only | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ |
| + from-scratch reranker | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ |

The numbers are produced by a single Sol eval job (`slurm/eval_reranker.sh`) and recorded with full provenance — checkpoint, SLURM job id, candidate count, metrics, and missed-gold count — in `results/eval_<JOBID>.json`.

---

## Attention visualization

`notebooks/attention_demo.ipynb` runs a trained reranker on an example (query, passage) pair and renders the query × passage attention heatmap averaged over layers and heads. Attention weight patterns are useful for hypothesis generation but are not causal explanations — see Jain & Wallace, "Attention is not Explanation" (NAACL 2019) for why high attention on a token does not imply that token is decisive for the score.

---

## Pipeline on Sol (ASU HPC)

The full end-to-end run, in order. The repo (reranker, training loop, serving code) is what lives in git; the corpus, FAISS index, training pairs, and checkpoints all live on Sol scratch under `/scratch/spate472/mlrag/` and are **never committed**.

### 0. Prerequisites — data + index on Sol

Before anything else, the corpus and FAISS index must already exist on scratch:

```
/scratch/spate472/mlrag/
├── corpus/papers.jsonl        # one paper per line: {id, title, abstract, ...}
└── index/
    ├── index.faiss            # FAISS flat-IP embedding index
    └── meta.jsonl             # paper id per FAISS row, in row order
```

These are produced on Sol by two build steps run there, where faiss and the GPU embedder are available: fetching the ArXiv abstracts, then embedding them and building the FAISS flat-IP index. **Those two build scripts are not committed to this repo** — they live on Sol scratch and are an acknowledged reproducibility gap. A reviewer cloning the repo cannot regenerate the corpus or index from it; the steps below assume `corpus/papers.jsonl`, `index/index.faiss`, and `index/meta.jsonl` already exist on scratch. Nothing downstream works until these files are in place.

### 1. Clone + set up

```bash
git clone https://github.com/sakethreddy26/ArXivLens.git /home/spate472/arxivlens
cd /home/spate472/arxivlens
```

`/home/spate472/arxivlens` is the `REPO_DIR` the SLURM scripts expect — clone there or edit the `REPO_DIR` variable at the top of both `slurm/*.sh`. The core dependencies (torch, accelerate, mlflow, transformers, faiss) all ship in the prebuilt `genai25.09` mamba env the jobs activate, so **no pip install is needed** for training or eval.

### 2. Build training pairs

Run `scripts/build_pairs.py` on Sol (this is where faiss and the index live), writing its output to the scratch path the train job reads:

```bash
python scripts/build_pairs.py \
  --input  /scratch/spate472/mlrag/corpus/papers.jsonl \
  --index  /scratch/spate472/mlrag/index \
  --output /scratch/spate472/mlrag/corpus/pairs.jsonl \
  --n-hard 2 --n-easy 2 --seed 0
```

This reads the corpus, reconstructs each paper's embedding from the FAISS index to pull hard negatives (nearest neighbours), and emits `{query, passage, label}` pairs at a 1 positive : 2 hard : 2 easy-random negative ratio.

> **Critical:** the `--output` path must be `/scratch/spate472/mlrag/corpus/pairs.jsonl`, because `slurm/train_reranker.sh` reads `PAIRS_FILE` from exactly there. The script's repo-relative default (`corpus/pairs.jsonl`) will *not* be found by the job.

**Validation split.** By default both the training and eval jobs auto-split a 10% held-out set (`val_fraction: 0.1`, `seed: 42`) from `pairs.jsonl`, so the val split is consistent across the two with no extra step. If you'd rather materialise an explicit validation file, pass `--val-output` and `build_pairs.py` will write a deterministic (seeded) 10% split to it:

```bash
python scripts/build_pairs.py \
  --input      /scratch/spate472/mlrag/corpus/papers.jsonl \
  --index      /scratch/spate472/mlrag/index \
  --output     /scratch/spate472/mlrag/corpus/pairs.jsonl \
  --val-output /scratch/spate472/mlrag/corpus/val_pairs.jsonl \
  --n-hard 2 --n-easy 2 --seed 0
```

When `val_pairs.jsonl` is present at `/scratch/spate472/mlrag/corpus/`, `slurm/eval_reranker.sh` uses it directly; otherwise the eval job falls back to the same 10% auto-split from `pairs.jsonl`.

### 3. Train

```bash
sbatch slurm/train_reranker.sh
```

2×A100-80GB, DDP via Accelerate, bf16 mixed precision, 8 h wall clock. Training is BCEWithLogitsLoss on the (query, passage, label) pairs; AdamW with linear warmup then constant LR (optionally cosine decay via `training.lr_schedule`). The job auto-detects existing checkpoints in `/scratch/spate472/mlrag/checkpoints/` and passes `--resume` to reload the latest one — so re-submitting after a wall-clock preemption picks up where it left off and is always safe.

### 4. Monitor

```bash
# Stream the live job log
tail -f /scratch/spate472/mlrag/logs/train_<JOBID>.out

# Inspect metrics (train loss, nDCG@10, MRR) in the MLflow UI
mlflow ui --backend-store-uri /scratch/spate472/mlrag/mlruns
```

### 5. Evaluate

```bash
sbatch slurm/eval_reranker.sh
```

1×A100. Loads the latest checkpoint from `/scratch/spate472/mlrag/checkpoints/` automatically and prints nDCG@5, nDCG@10, MRR, and Recall@{1,5,10} to the eval log:

```bash
tail -f /scratch/spate472/mlrag/logs/eval_<JOBID>.out
```

### 6. Serve (optional)

Point the FastAPI service at the trained checkpoint plus the FAISS index by setting `CHECKPOINT`, `INDEX_PATH`, and `META_PATH` — see [Quick start — Docker](#quick-start--docker) for the exact env vars and run command.

---

## Repo structure

```
ArXivLens/
├── src/arxivlens/
│   ├── model/
│   │   ├── transformer.py      # From-scratch TransformerEncoder (the core)
│   │   ├── reranker.py         # CrossEncoderReranker + TokenizerLike protocol
│   │   └── attention.py        # Attention extraction + query×passage slicing
│   ├── retrieve/
│   │   ├── index.py            # FaissRetriever (MiniLM + FAISS flat-IP)
│   │   └── pipeline.py         # RetrieveReranker, RetrieverLike/RerankerLike protocols
│   ├── serve/
│   │   ├── api.py              # FastAPI app: /health /search /explain
│   │   └── schemas.py          # Pydantic request/response models
│   ├── data/
│   │   ├── pairs.py            # (query, passage, label) pair building
│   │   └── dataset.py          # PairDataset + collate_fn
│   └── train/
│       ├── train_reranker.py   # Training loop (Accelerate + MLflow + checkpoint/resume)
│       └── eval.py             # nDCG, MRR, Recall metric computation
├── slurm/
│   ├── train_reranker.sh       # SLURM job: 2×A100, DDP bf16
│   └── eval_reranker.sh        # SLURM job: 1×A100 eval
├── notebooks/
│   └── attention_demo.ipynb    # Query×passage attention heatmap
├── tests/                      # pytest suite (CPU-friendly, no GPU required)
├── configs/                    # reranker.yaml hyperparameters
├── scripts/
│   └── build_pairs.py          # Build training pairs from FAISS hard negatives
├── Dockerfile                  # Two-stage build (CPU-only runtime)
├── pyproject.toml
└── requirements.txt
```

---

## Limitations

- The reranker is trained on synthetic pairs built from FAISS hard negatives. No human relevance labels have been collected yet, so ranking quality against real user queries is unverified.
- The model is small (d_model=512, 6 layers, ~34.5 M parameters) relative to production rerankers. Performance numbers will remain placeholder until the Sol eval job completes.
- Attention weights are an interpretability aid, not a causal explanation. Token salience measured by attention does not reliably predict counterfactual importance (Jain & Wallace 2019).
- The Docker image uses CPU-only PyTorch. Serving latency at scale requires a GPU host; the cross-encoder scores each candidate with a separate forward pass (O(K) per query).

---
