# Model Card — ArXivLens CrossEncoderReranker

**Model type:** Cross-encoder relevance reranker (from-scratch transformer)
**Domain:** ArXiv ML paper abstracts
**Stage:** Stage-2 reranker in a retrieve-then-rerank pipeline
**Version:** 0.1.0

---

## 1. Model Details

### Architecture

| Component | Value |
|---|---|
| Class | `CrossEncoderReranker` (`src/arxivlens/model/reranker.py`) |
| Encoder body | `TransformerEncoder` (`src/arxivlens/model/transformer.py`) |
| Vocabulary | WordPiece, bert-base-uncased — 30,522 tokens |
| Input format | `[CLS] query [SEP] passage [SEP]`, max 256 tokens (`max_input_length`) |
| Layers | 6 pre-norm `EncoderLayer` blocks |
| Attention heads | 8 (d_head = 64) |
| d_model | 512 |
| d_ff | 2,048 |
| Dropout | 0.1 (attention weights, FFN activations, residual branches) |
| Positional encoding | Sinusoidal, non-trainable (max_len=512) |
| Output head | `Linear(512, 1)` on `[CLS]` hidden state → relevance logit |
| Parameter count | ~34.5 M (derived analytically from `configs/reranker.yaml`; verify via `sum(p.numel() for p in model.parameters())`) |

The encoder is written from scratch in pure PyTorch. No pretrained transformer weights are used — only the WordPiece tokenizer is borrowed from `bert-base-uncased` via `AutoTokenizer`.

### Input and Output

**Input** — a query string and a passage string, concatenated by the tokenizer:

```python
from transformers import AutoTokenizer
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.model.transformer import TransformerConfig

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = TransformerConfig(vocab_size=30522, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=512)
model = CrossEncoderReranker(config, tokenizer=tokenizer)

# Score one query against multiple passages
scores = model.score("transformer architectures for NLP", passages)
# scores: 1-D tensor of len(passages) relevance logits; higher = more relevant
ranked = sorted(enumerate(passages), key=lambda x: scores[x[0]].item(), reverse=True)
```

**Output** — a raw logit (unbounded scalar, higher = more relevant). Apply `torch.sigmoid` to convert to a probability. Training uses `BCEWithLogitsLoss` directly on the logit.

---

## 2. Intended Use

**Primary use case:** Stage-2 reranker in the ArXivLens retrieve-then-rerank pipeline. A FAISS bi-encoder retrieves up to 50 candidate papers; this model reranks them by scoring each `(query, abstract)` pair with a full forward pass.

When served via `src/arxivlens/serve/api.py`, the reranker's architecture is read from the config stored inside the checkpoint (not a hardcoded arch), so any trained checkpoint loads without a shape mismatch.

**Intended users:** Researchers and students querying the ArXiv ML corpus through the ArXivLens API or notebooks.

**Out-of-scope uses:**

- General-purpose reranking outside the ArXiv ML domain.
- Production serving without GPU — CPU-only inference is supported but will be slow at retrieval scale.
- Tasks requiring calibrated probabilities (the output is a logit, not a calibrated probability).
- Any application requiring human-verified relevance judgments as a quality guarantee.

---

## 3. Training Data

**Corpus:** ArXiv ML paper abstracts (titles + abstracts in `corpus/pairs.jsonl`, gitignored).

**Pair generation** (`src/arxivlens/data/pairs.py`): Synthetic labels are produced by `build_pairs` with no human annotation.

| Pair type | Construction | Label |
|---|---|---|
| Positive | Paper title → its own abstract | 1 |
| Hard negative | FAISS ANN neighbors of the paper (topically close, not the same paper) | 0 |
| Easy negative | Random other papers from the corpus | 0 |

Default ratio: 1 positive : 2 hard negatives : 2 easy negatives per query (`n_hard=2`, `n_easy=2`).

**Limitation of synthetic labels:** A hard negative might be genuinely relevant to the title-query yet be labeled 0 because it is not the source paper. There are no human judgments to correct this. Evaluation numbers should be read as estimates of ranking ability under the same synthetic protocol, not as claims against human relevance.

---

## 4. Training Procedure

### Hardware and distributed setup

| Setting | Value |
|---|---|
| Cluster | ASU Sol HPC (`slurm/train_reranker.sh`) |
| GPUs | 2 × A100-80GB |
| Distributed | DDP via Hugging Face Accelerate |
| Precision | bf16 mixed precision (A100 native); fp32 fallback on other hardware |
| Wall-clock limit | 8 hours per SLURM job (auto-resume supported) |

### Optimizer and loss

| Hyperparameter | Value (from `configs/reranker.yaml`) |
|---|---|
| Loss | `BCEWithLogitsLoss` (binary cross-entropy per pair) |
| Optimizer | AdamW |
| Learning rate | 2e-4 (base) |
| LR schedule | Linear warmup for 200 steps, then constant (`lr_schedule: constant`) |
| Gradient clipping | Max global grad norm 1.0 |
| Batch size | 128 |
| Epochs | 8 |
| Seed | 42 |

### Checkpointing

Checkpoints are saved every 500 optimizer steps and at the end of every epoch, to `checkpoints/checkpoint_epoch{EEEE}_step{SSSSSS}.pt`. Zero-padded filenames ensure lexicographic sort equals numeric sort; `--resume` reloads the latest automatically.

### Experiment tracking

MLflow logs `train/loss`, `train/lr`, and validation metrics every 500 steps to `mlruns/` (experiment: `arxivlens-reranker`). To launch the UI:

```bash
mlflow ui --backend-store-uri /scratch/spate472/mlrag/mlruns
```

### Launching training

```bash
# Fresh run
python -m arxivlens.train.train_reranker

# Override pairs file and resume a previous run
python -m arxivlens.train.train_reranker \
    --pairs corpus/pairs.jsonl \
    --resume

# On Sol (handles DDP, bf16, auto-resume)
sbatch slurm/train_reranker.sh
```

---

## 5. Evaluation

Metrics are computed by `src/arxivlens/train/eval.py` (entry point: `evaluate_rankings`). All metrics are **macro-averaged** across queries; ties are broken by stable sort (original input order).

| Metric | Definition |
|---|---|
| nDCG@5 | Normalized Discounted Cumulative Gain at rank 5 |
| nDCG@10 | Normalized Discounted Cumulative Gain at rank 10 |
| MRR | Mean Reciprocal Rank of the first relevant result |
| Recall@1 | Fraction of queries where the relevant paper is rank 1 |
| Recall@5 | Fraction of queries where the relevant paper is in the top 5 |
| Recall@10 | Fraction of queries where the relevant paper is in the top 10 |

**Results.** Head-to-head on the same held-out queries, 50 candidates per query:

| Stage | nDCG@5 | nDCG@10 | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| FAISS retrieval-only | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ |
| + from-scratch reranker | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ | _pending eval job_ |

The numbers are produced by a single Sol eval job (`slurm/eval_reranker.sh`) and recorded with full provenance — checkpoint, SLURM job id, candidate count, metrics, and missed-gold count — in `results/eval_<JOBID>.json`.

Note: All eval numbers reflect the synthetic label protocol described in Section 3. Because hard negatives may be genuinely relevant, these numbers underestimate real-world ranking quality.

---

## 6. Attention Visualization

The `POST /explain` API endpoint returns per-layer, per-head attention weights for a specific `(query, paper_id)` pair:

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "graph neural networks", "paper_id": "2301.00001"}'
```

The response includes `tokens`, `attention_weights` (shape: `query_tokens × passage_tokens`, averaged over layers and heads), and the relevance `score`.

For interactive exploration, `notebooks/attention_demo.ipynb` renders a query-rows × passage-columns heatmap — the sub-block bounded by the first and second `[SEP]` tokens, excluding padding.

**Important caveat:** Attention weights show which tokens each head attended to, but they are not a faithful explanation of why the model assigned a particular score. Jain & Wallace (NAACL 2019) demonstrate that attention distributions can be manipulated without changing model outputs and do not reliably identify which inputs drove the prediction. Treat the heatmap as an exploratory diagnostic, not a causal explanation.

---

## 7. Limitations

- **Synthetic training data only.** No human relevance judgments were used. The positive/negative labeling is a proxy, not ground truth.
- **Small model.** d_model=512 with 6 layers (~34.5 M parameters) is smaller than production rerankers (e.g., BERT-base at d_model=768, 12 layers). Ranking quality will likely fall below commercial alternatives.
- **Domain-specific.** Trained exclusively on ArXiv ML abstracts. Performance on other domains (e.g., biomedical, legal, code) is untested and not expected to transfer without fine-tuning.
- **CPU inference is slow.** The Docker image runs CPU-only; scoring 50 candidates per query is feasible but not production-grade without a GPU.
- **No bias or fairness evaluation.** The model has not been audited for differential ranking by author demographics, institutional affiliation, or writing style.
- **Attention is not explanation.** See Section 6.

---

## 8. Ethical Considerations

- **No sensitive personal data.** Training data consists entirely of publicly available ArXiv paper titles and abstracts.
- **Output is a ranked list, not a decision.** The model surfaces papers for human review; it does not autonomously accept, reject, or act on any submission.
- **Human oversight recommended.** Rankings should be treated as a starting point for exploration. High-stakes use cases (e.g., literature reviews for grant applications or systematic reviews) warrant independent verification of results.
- **Ranking artifacts.** The model may amplify patterns in the ArXiv corpus (e.g., favoring papers with longer, more structured abstracts) without those patterns correlating with actual relevance. Users should verify that top-ranked results are appropriate for their query.

---

## 9. Citation

If you use ArXivLens in your work, please cite:

```bibtex
@misc{arxivlens2026,
  author       = {Saket Reddy Pate},
  title        = {{ArXivLens}: Semantic Search over {ArXiv} {ML} Papers
                  with a From-Scratch Cross-Encoder Reranker},
  year         = {2026},
  howpublished = {\url{https://github.com/sakethreddy26/arxivlens}},
  note         = {Model card and source code at \texttt{MODEL\_CARD.md}}
}
```
