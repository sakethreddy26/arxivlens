"""Training script for the ArXivLens cross-encoder reranker.

Pipeline overview
-----------------
This script wires together every component built in earlier phases:

    argparse CLI
        -> load reranker.yaml
        -> build TransformerConfig + CrossEncoderReranker
        -> build PairDataset (train + optional val)
        -> Accelerator (bf16 mixed precision on A100; CPU fallback)
        -> AdamW + linear warmup then constant LR (optional cosine decay)
        -> pointwise BCE or query-group listwise training loop
        -> MLflow metric logging
        -> checkpoint save/resume

Loss choices
------------
``bce`` preserves the original pointwise binary-classification objective.
``listwise`` batches complete query groups and minimizes a softmax cross-entropy
that raises the single gold passage above every negative in its candidate list.
The latter matches the actual ranking task and is the recommended Sol setup.

Checkpointing for Sol
---------------------
Sol HPC jobs have an 8-hour wall-clock cap. ``_save_checkpoint`` snapshots the
full training state (model weights, optimiser, scheduler, epoch, step, raw
config) so a new job can pick up exactly where the previous one left off via
``--resume``. Checkpoints live in ``checkpoint_dir`` (never committed to git).

Usage examples
--------------
    # Fresh training run with defaults from configs/reranker.yaml
    python -m arxivlens.train.train_reranker

    # Override pairs path and resume a previous run
    python -m arxivlens.train.train_reranker \
        --pairs corpus/pairs.jsonl \
        --resume

    # Non-default config
    python -m arxivlens.train.train_reranker --config configs/my_reranker.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from arxivlens.data.dataset import (
    PairDataset,
    QueryGroupDataset,
    collate_fn,
    collate_query_groups,
    group_split_indices,
)
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.model.transformer import TransformerConfig
from arxivlens.train.eval import build_retrieval_eval_queries, evaluate_rankings
from arxivlens.train.losses import listwise_softmax_loss


def _mlflow_safe(name: str) -> str:
    """MLflow metric names may not contain '@'; map it to '_at_' (e.g.
    'val/ndcg@5' -> 'val/ndcg_at_5'). Other chars are already MLflow-legal."""
    return name.replace("@", "_at_")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return the top-level dict.

    Kept separate so tests can monkeypatch or pass alternative configs without
    touching the filesystem.
    """
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class _Namespace:
    """Thin wrapper that makes a nested dict accessible as ``obj.key``.

    We don't pull in pydantic or dataclasses-json to keep dependencies lean;
    this small helper is enough to give readable attribute access throughout the
    training loop.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            # Recursively wrap nested dicts so ``cfg.training.lr`` works.
            setattr(self, key, _Namespace(value) if isinstance(value, dict) else value)

    def as_dict(self) -> dict[str, Any]:
        """Flatten back to a plain dict for MLflow param logging."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _Namespace):
                for subkey, subval in value.as_dict().items():
                    out[f"{key}.{subkey}"] = subval
            else:
                out[key] = value
        return out

    def as_dict_nested(self) -> dict[str, Any]:
        """Reconstruct the original NESTED dict (mirrors the raw YAML layout).

        Used as the checkpoint ``config`` when a caller (e.g. the smoke test)
        does not supply the raw ``cfg_dict``. Nested because eval_reranker.sh
        reads ``state["config"]["model"]["vocab_size"]`` etc.
        """
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = value.as_dict_nested() if isinstance(value, _Namespace) else value
        return out


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    step: int,
    checkpoint_dir: str | Path,
    cfg_dict: dict[str, Any],
    batch_in_epoch: int = 0,
) -> None:
    """Serialize training state to ``checkpoint_dir/checkpoint_epoch{e}_step{s}.pt``.

    Only the main process writes the file; other ranks (in multi-GPU runs) just
    wait. The saved state includes the raw YAML config so the checkpoint is
    self-documenting — useful months later when you forget which hyperparams
    produced which checkpoint.

    Args:
        accelerator: wraps DDP/FSDP; used to unwrap the model and guard writes.
        model: the (potentially DDP-wrapped) reranker.
        optimizer: AdamW, mid-training state.
        scheduler: LambdaLR warmup scheduler.
        epoch: epoch to RESUME INTO (0-based). Step checkpoints pass the
            epoch currently in progress (resume re-enters it and skips ahead
            to ``batch_in_epoch``); epoch-end checkpoints pass ``epoch + 1``
            (with ``batch_in_epoch=0``) so resume starts the NEXT epoch
            instead of silently retraining the completed one.
        step: global optimizer step count.
        checkpoint_dir: directory to write into; created if absent.
        cfg_dict: raw YAML config dict for provenance.
        batch_in_epoch: index of the NEXT unseen batch within ``epoch`` (i.e.
            ``batch_idx + 1`` at step-checkpoint time; ``0`` at an epoch-end
            checkpoint). Resume skips batches with ``batch_idx <
            batch_in_epoch``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}_step{step:06d}.pt"

    # accelerator.unwrap_model strips DDP/FSDP wrappers so we always save the
    # raw module state dict, which is loadable on any topology.
    unwrapped = accelerator.unwrap_model(model)

    state: dict[str, Any] = {
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "batch_in_epoch": batch_in_epoch,  # next unseen batch within `epoch`
        # batch_in_epoch is a PER-RANK batch index: under DDP the loader is
        # sharded, so the same index means different data at a different
        # world size. Record num_processes so resume can refuse a mismatch.
        "num_processes": accelerator.num_processes,
        "config": cfg_dict,  # raw YAML dict for reproducibility
    }

    if accelerator.is_main_process:
        torch.save(state, path)
        accelerator.print(f"[checkpoint] saved to {path}")


def _load_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the most recent checkpoint path, or None if the directory is empty.

    Checkpoints are named ``checkpoint_epoch{e:04d}_step{s:06d}.pt``. Zero-padding
    is applied to both epoch (4 digits) and step (6 digits), so lexicographic sort
    is always identical to numeric sort — no edge cases regardless of run length.
    """
    checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint_epoch*_step*.pt"))
    if not checkpoints:
        return None
    return checkpoints[-1]  # last alphabetically = highest step


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_eval(
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Run the held-out val set through the model and return ranking metrics.

    Candidates are grouped by ``query_id`` across the WHOLE val loader: every
    (query, passage) pair carrying the same ``query_id`` is one candidate of a
    single ranking, so ``evaluate_rankings`` sees a genuine multi-candidate
    ranking per query rather than a degenerate 1-candidate-per-query view. This
    survives a shuffled val loader because grouping is by key, not by batch, and
    handles a variable number of candidates per query (a query_id's candidates
    may even be split across train/val by the pair-level shuffle in
    ``build_pairs.py``).

    Args:
        model: the reranker (may be DDP-wrapped; forward still works).
        val_loader: dataloader over the held-out pairs; batches carry
            ``query_ids`` (a list[str]) from ``collate_fn``.
        accelerator: used only to move tensors; eval always runs on main process.

    Returns:
        Dict with keys from ``evaluate_rankings``:
        ``ndcg@5``, ``ndcg@10``, ``mrr``, ``recall@1``, ``recall@5``, ``recall@10``.
    """
    model.eval()
    # query_id -> (scores, labels) accumulated across every batch.
    groups: dict[str, tuple[list[float], list[float]]] = {}

    try:
        with torch.no_grad():
            for batch in val_loader:
                # val_loader is a plain (un-prepared) DataLoader over the FULL
                # val set — accelerate must not shard it, or this main-rank-only
                # eval would score just 1/N of the queries with groups truncated
                # at shard boundaries. Batches therefore arrive on CPU and are
                # moved to the right device here.
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                logits = model(input_ids, attention_mask)  # (B,)
                scores = logits.cpu().float().tolist()
                labels = batch["labels"].cpu().float().tolist()
                query_ids = batch["query_ids"]
                for qid, s, l in zip(query_ids, scores, labels):
                    bucket = groups.setdefault(qid, ([], []))
                    bucket[0].append(s)
                    bucket[1].append(l)
    finally:
        model.train()

    # Skip empty groups (evaluate_rankings also skips empties, but be explicit).
    all_queries = [(s, l) for s, l in groups.values() if s]
    return evaluate_rankings(all_queries)


# ---------------------------------------------------------------------------
# Retrieve-then-rerank final eval (FAISS)
# ---------------------------------------------------------------------------

def _raw_val_records(val_dataset: Any) -> list[dict[str, Any]]:
    """Return the raw parsed pairs backing ``val_dataset``, in val order.

    Handles both val-dataset shapes the trainer produces:

    * a :class:`~torch.utils.data.Subset` over the full ``PairDataset`` (the
      seeded group-wise auto-split), whose ``.indices`` select the held-out
      rows of ``subset.dataset._lines``;
    * a plain ``PairDataset`` loaded from an explicit ``val_pairs_file``.

    Parses the stored JSON lines directly (no tokenization) so this is cheap
    and independent of ``__getitem__``.
    """
    if isinstance(val_dataset, Subset):
        base = val_dataset.dataset
        indices = list(val_dataset.indices)
    else:
        base = val_dataset
        indices = range(len(getattr(base, "_lines", [])))

    lines = getattr(base, "_lines", None)
    if lines is None:
        return []
    return [json.loads(lines[i]) for i in indices]


def _reconstruct_eval_records(val_dataset: Any) -> list[dict[str, Any]]:
    """Rebuild one held-out eval record per query from the val pairs.

    Each query_id group in ``pairs.jsonl`` carries exactly one label-1 pair
    (the paper's own title/abstract). We reconstruct, per group, a record with:

    * ``id``       = the pair's ``query_id`` (the paper's real id, per
      ``build_pairs``; a bare positional-index fallback is rejected downstream
      by ``build_retrieval_eval_queries``);
    * ``title``    = the pair's ``query`` (the paper title used as the query);
    * ``abstract`` = the label-1 pair's ``passage`` (the paper's own abstract).

    Only groups whose positive (label 1) pair is present in the held-out split
    yield a record — a group cannot be scored as retrieve-then-rerank without
    its gold abstract. Preserves first-seen group order for reproducibility.
    Does NOT read papers.jsonl (design decision O2).
    """
    records: dict[str, dict[str, Any]] = {}
    for pair in _raw_val_records(val_dataset):
        if int(pair.get("label", 0)) != 1:
            continue
        qid = str(pair.get("query_id", ""))
        if not qid or qid in records:
            continue
        records[qid] = {
            "id": qid,
            "title": "" if pair.get("query") is None else str(pair["query"]),
            "abstract": "" if pair.get("passage") is None else str(pair["passage"]),
        }
    return list(records.values())


def _try_build_faiss_retriever(cfg: _Namespace, accelerator: Accelerator) -> Any | None:
    """Build a ``FaissRetriever`` for final eval, or return ``None`` to fall back.

    Graceful degradation (design decision O4): the index/meta files are absent
    in CI, CPU smoke tests, and most local runs, and ``faiss`` /
    ``sentence-transformers`` may not be installed at all. Any of those cases
    must NOT break training — we log the reason and return ``None`` so the
    caller uses the existing grouped-by-query_id eval instead.

    All heavy imports happen lazily inside ``FaissRetriever.__init__``; this
    function only touches the filesystem and catches everything.
    """
    index_path = Path(str(getattr(cfg.training, "eval_index_path", "index/index.faiss")))
    meta_path = Path(str(getattr(cfg.training, "eval_meta_path", "index/meta.jsonl")))

    if not index_path.exists() or not meta_path.exists():
        accelerator.print(
            f"[final eval] FAISS index/meta not found "
            f"(index={index_path}, meta={meta_path}); "
            "falling back to grouped-by-query_id eval."
        )
        return None

    try:
        # Lazy import so a missing faiss/sentence-transformers install never
        # breaks module import or CPU-only runs.
        from arxivlens.retrieve.index import FaissRetriever  # noqa: PLC0415

        retriever = FaissRetriever(index_path, meta_path)
    except Exception as exc:  # noqa: BLE001 — any failure => graceful fallback
        accelerator.print(
            f"[final eval] could not build FaissRetriever ({type(exc).__name__}: {exc}); "
            "falling back to grouped-by-query_id eval."
        )
        return None

    accelerator.print(
        f"[final eval] using FAISS retrieve-then-rerank eval "
        f"(index={index_path}, meta={meta_path})."
    )
    return retriever


def _run_faiss_eval(
    model: nn.Module,
    retriever: Any,
    val_dataset: Any,
    accelerator: Accelerator,
    num_candidates: int,
    passage_format: str = "title_abstract",
) -> dict[str, float]:
    """Final retrieve-then-rerank eval over the held-out queries.

    Reconstructs eval records from the val pairs, retrieves ``num_candidates``
    real candidates per query, reranks them with the (unwrapped) cross-encoder,
    and aggregates ranking metrics. Runs on the calling process only (the
    caller guards this to rank 0); the reranker's ``score`` handles device
    placement via the model's own parameters.
    """
    eval_records = _reconstruct_eval_records(val_dataset)
    accelerator.print(
        f"[final eval] reconstructed {len(eval_records)} held-out query records "
        f"for FAISS eval (num_candidates={num_candidates})."
    )
    # Unwrap so we call the plain CrossEncoderReranker.score (DDP wrappers do
    # not expose it), matching how checkpoints store the unwrapped module.
    reranker = accelerator.unwrap_model(model)
    was_training = reranker.training
    reranker.eval()
    try:
        queries = build_retrieval_eval_queries(
            eval_records,
            retriever,
            reranker,
            num_candidates,
            passage_format=passage_format,
        )
    finally:
        if was_training:
            reranker.train()
    return evaluate_rankings(queries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface.

    All arguments override the corresponding key in ``reranker.yaml``; the YAML
    provides defaults so the script can be launched with zero flags in the common
    case.
    """
    p = argparse.ArgumentParser(
        description="Train the ArXivLens cross-encoder reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reranker.yaml"),
        help="Path to reranker.yaml (hyperparameters + paths).",
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override config training.pairs_file.",
    )
    p.add_argument(
        "--val-pairs",
        type=Path,
        default=None,
        dest="val_pairs",
        metavar="PATH",
        help="Override config training.val_pairs_file. "
             "If omitted and the config file does not exist, "
             "a fraction of --pairs is held out (val_fraction).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        dest="checkpoint_dir",
        metavar="PATH",
        help="Override config training.checkpoint_dir.",
    )
    p.add_argument(
        "--mlflow-dir",
        type=Path,
        default=None,
        dest="mlflow_dir",
        metavar="PATH",
        help="Override config training.mlflow_dir.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in checkpoint_dir.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override config training.n_epochs.",
    )
    p.add_argument(
        "--eval-index-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override config training.eval_index_path.",
    )
    p.add_argument(
        "--eval-meta-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override config training.eval_meta_path.",
    )
    p.add_argument(
        "--eval-passage-format",
        choices=("abstract", "title_abstract"),
        default=None,
        help="Override config training.eval_passage_format.",
    )
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_training(
    cfg: _Namespace,
    tokenizer: Any,
    accelerator: Accelerator | None = None,
    resume: bool = False,
    cfg_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build model/data/optimizer from ``cfg`` and run the full training loop.

    Extracted from :func:`main` so callers (notably the CPU smoke test) can drive
    the REAL training loop with an injected offline tokenizer — no HuggingFace
    download, no CLI. ``main`` is a thin wrapper that parses args, builds the
    accelerator + tokenizer, and delegates here.

    Args:
        cfg: parsed ``_Namespace`` config (already merged with CLI overrides).
        tokenizer: any object satisfying the ``TokenizerLike`` protocol.
        accelerator: an ``Accelerator``; created here (bf16 with fp32 fallback)
            when ``None``.
        resume: when True, load the latest checkpoint in ``checkpoint_dir`` and
            continue from the exact saved epoch / batch_in_epoch / global_step.
        cfg_dict: raw YAML dict stored inside checkpoints for provenance;
            defaults to ``cfg.as_dict()`` when omitted.

    Returns:
        A small log dict with keys ``step_losses`` (list[float], per optimizer
        step), ``trained_batch_indices`` (dict[int, list[int]]: epoch ->
        batch_idx values actually trained), ``global_step`` (final step count),
        and ``final_metrics`` (the last eval metrics dict, or ``None``).
    """
    if cfg_dict is None:
        cfg_dict = cfg.as_dict_nested()

    # Convenient local aliases to keep loop code readable.
    pairs_file: str = cfg.training.pairs_file
    val_pairs_file: str = getattr(cfg.training, "val_pairs_file", "")
    checkpoint_dir: str = cfg.training.checkpoint_dir
    mlflow_dir: str = cfg.training.mlflow_dir
    lr: float = float(cfg.training.learning_rate)
    batch_size: int = int(cfg.training.batch_size)
    n_epochs: int = int(cfg.training.n_epochs)
    warmup_steps: int = int(cfg.training.warmup_steps)
    checkpoint_every: int = int(cfg.training.checkpoint_every_steps)
    eval_every: int = int(cfg.training.eval_every_steps)
    val_fraction: float = float(cfg.training.val_fraction)
    seed: int = int(cfg.training.seed)
    max_input_length: int = int(cfg.model.max_input_length)
    # Gradient clipping: getattr fallback keeps old configs working (default 1.0).
    grad_clip: float = float(getattr(cfg.training, "grad_clip", 1.0))
    # LR schedule shape after warmup: "constant" (default, unchanged numerics)
    # or "cosine" (decay to 0 over the full run). getattr keeps old configs OK.
    lr_schedule: str = str(getattr(cfg.training, "lr_schedule", "constant"))
    # Retrieve-then-rerank final-eval breadth. getattr fallback keeps old
    # configs (and the CPU smoke test's config) working; a YAML null would
    # slip past a plain getattr, so coerce with an explicit default guard.
    eval_num_candidates: int = int(getattr(cfg.training, "eval_num_candidates", 50) or 50)
    eval_passage_format: str = str(
        getattr(cfg.training, "eval_passage_format", "title_abstract")
    )
    if eval_passage_format not in {"abstract", "title_abstract"}:
        raise ValueError(
            "training.eval_passage_format must be 'abstract' or "
            f"'title_abstract', got {eval_passage_format!r}"
        )
    loss_type: str = str(getattr(cfg.training, "loss_type", "bce")).lower()
    if loss_type not in {"bce", "listwise"}:
        raise ValueError(
            f"training.loss_type must be 'bce' or 'listwise', got {loss_type!r}"
        )
    queries_per_batch: int = int(getattr(cfg.training, "queries_per_batch", 4))
    if queries_per_batch <= 0:
        raise ValueError("training.queries_per_batch must be positive")
    bce_aux_weight: float = float(getattr(cfg.training, "bce_aux_weight", 0.0))
    if bce_aux_weight < 0:
        raise ValueError("training.bce_aux_weight cannot be negative")
    weight_decay: float = float(getattr(cfg.training, "weight_decay", 0.01))

    # ------------------------------------------------------------------
    # 2. Reproducibility seeds
    # ------------------------------------------------------------------
    # Seed all sources of randomness before creating datasets or models.
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # numpy not strictly required; seed it only when available

    # ------------------------------------------------------------------
    # 3. Accelerator — bf16 on A100, silent fallback to fp32 elsewhere
    # ------------------------------------------------------------------
    if accelerator is None:
        try:
            accelerator = Accelerator(mixed_precision="bf16")
        except (ValueError, RuntimeError):
            # bf16 is not supported on all hardware (e.g., CPU-only or older GPUs).
            # Fall back to fp32 so the script also runs in test/local environments.
            accelerator = Accelerator(mixed_precision="no")

    accelerator.print(
        f"[accelerate] device={accelerator.device}, "
        f"mixed_precision={accelerator.mixed_precision}, "
        f"num_processes={accelerator.num_processes}"
    )

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    transformer_config = TransformerConfig(
        vocab_size=int(cfg.model.vocab_size),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ff=int(cfg.model.d_ff),
        max_len=int(cfg.model.max_len),
        dropout=float(cfg.model.dropout),
    )
    model = CrossEncoderReranker(
        transformer_config,
        tokenizer=tokenizer,
        max_length=max_input_length,
    )
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"[model] {n_params:,} parameters")

    # ------------------------------------------------------------------
    # 6. Datasets and dataloaders
    # ------------------------------------------------------------------
    accelerator.print(f"[data] loading training pairs from {pairs_file} ...")
    full_dataset = PairDataset(pairs_file, tokenizer, max_length=max_input_length)

    # Determine val dataset: explicit file > auto-split from training set.
    val_path = Path(val_pairs_file) if val_pairs_file else None
    if val_path is not None and val_path.exists():
        accelerator.print(f"[data] loading val pairs from {val_path} ...")
        val_dataset = PairDataset(str(val_path), tokenizer, max_length=max_input_length)
        train_dataset = full_dataset
    else:
        # Hold out val_fraction of the QUERY GROUPS for monitoring. A
        # pair-level random_split would orphan groups across the train/val
        # boundary (forced 0.0 for no-positive groups, trivial 1.0 for
        # positive-only singletons), biasing every ranking metric toward the
        # label ratio. group_split_indices assigns WHOLE query_id groups to
        # val and is seeded by `seed` alone so the split is reproducible
        # (the eval_reranker.sh mirror depends on that).
        train_idx, val_idx = group_split_indices(
            full_dataset.query_ids(), val_fraction, seed
        )
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        accelerator.print(
            f"[data] auto-splitting by query group: {len(train_idx)} train / "
            f"{len(val_idx)} val pairs (val_fraction={val_fraction} of query "
            "groups)"
        )

    # A dedicated generator drives the train shuffle so we can reseed it
    # deterministically at the top of each epoch (manual_seed(seed + epoch)).
    # That reproducible per-epoch order is what makes mid-epoch resume correct:
    # the resumed run replays the SAME shuffled order and skips already-seen
    # batches. This generator is SEPARATE from the group_split_indices seed
    # above, which stays `seed` alone so the val split is reproducible (the
    # eval_reranker.sh mirror depends on that).
    train_gen = torch.Generator()
    if loss_type == "listwise":
        grouped_train_dataset = QueryGroupDataset(train_dataset)
        n_train_groups = len(grouped_train_dataset)
        group_sizes = grouped_train_dataset.group_sizes()
        train_loader = DataLoader(
            grouped_train_dataset,
            batch_size=queries_per_batch,
            shuffle=True,
            generator=train_gen,
            collate_fn=collate_query_groups,
            # Remove a final underfilled query batch. Accelerate still shards
            # only whole batches, so no query candidate list is ever split.
            drop_last=n_train_groups >= queries_per_batch,
        )
        accelerator.print(
            f"[data] listwise groups={n_train_groups}, "
            f"queries_per_batch={queries_per_batch}, "
            f"candidates/query={min(group_sizes)}..{max(group_sizes)}"
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,           # shuffle every epoch for better gradient diversity
            generator=train_gen,    # reseeded per epoch -> reproducible resume
            collate_fn=collate_fn,
            drop_last=False,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # stable order for reproducible eval numbers
        collate_fn=collate_fn,
    )

    accelerator.print(
        f"[data] {len(train_dataset)} train pairs, "
        f"{len(val_dataset)} val pairs, "
        f"{len(train_loader)} unsharded train batches / epoch"
    )

    # ------------------------------------------------------------------
    # 7. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Two schedule shapes share the same linear warmup (0 -> lr over
    # warmup_steps):
    #   "constant" — hold at lr after warmup (warmup-then-CONSTANT, NOT decay).
    #                Simple and effective for short fine-tuning runs; keeps the
    #                default numerics identical to before this option existed.
    #   "cosine"   — after warmup, cosine-decay the scale from 1 -> 0 over the
    #                remaining steps of the full run (total_steps below).
    # With Accelerate's default split_batches=False, DDP ranks receive
    # different complete batches. Scheduler steps follow the per-rank length.
    steps_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes)
    total_steps = n_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        """Return the LR scale factor at ``step`` (multiplied by base lr)."""
        if step < warmup_steps:
            # Avoid division by zero when warmup_steps == 0.
            return step / max(1, warmup_steps)
        if lr_schedule == "cosine":
            # Cosine decay from 1.0 at end-of-warmup down to 0.0 at total_steps.
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0  # "constant": full base lr after warmup

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 8. Accelerate wrapping (handles device placement, DDP, mixed prec.)
    # ------------------------------------------------------------------
    # val_loader is deliberately NOT prepared: accelerator.prepare would shard
    # it across ranks, so the main-rank-only eval in _run_eval would see just
    # ~1/N of the val set with query groups truncated at shard boundaries.
    # It stays a plain DataLoader over the FULL val set; _run_eval moves each
    # batch to accelerator.device itself.
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # ------------------------------------------------------------------
    # 9. Optional resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    resume_batch = 0

    if resume:
        latest = _load_latest_checkpoint(checkpoint_dir)
        if latest is None:
            accelerator.print(
                f"[resume] no checkpoints found in {checkpoint_dir}; "
                "starting from scratch"
            )
        else:
            accelerator.print(f"[resume] loading checkpoint {latest} ...")
            # Load on CPU first so we don't duplicate tensors on GPU before
            # moving them; map_location='cpu' is safe with accelerate.
            state = torch.load(latest, map_location="cpu")
            saved_training = state.get("config", {}).get("training", {})
            saved_loss_type = str(saved_training.get("loss_type", "bce")).lower()
            if saved_loss_type != loss_type:
                raise RuntimeError(
                    f"checkpoint uses loss_type={saved_loss_type!r}, but this "
                    f"run requests {loss_type!r}; use a separate checkpoint directory"
                )
            if loss_type == "listwise":
                saved_queries_per_batch = int(
                    saved_training.get("queries_per_batch", queries_per_batch)
                )
                if saved_queries_per_batch != queries_per_batch:
                    raise RuntimeError(
                        "checkpoint uses queries_per_batch="
                        f"{saved_queries_per_batch}, but this run requests "
                        f"{queries_per_batch}; resume would change "
                        "batch_in_epoch semantics"
                    )
            # Restore model weights into the (possibly DDP-wrapped) model.
            accelerator.unwrap_model(model).load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            # Step checkpoints store the epoch IN PROGRESS (resume re-enters
            # it and skips to batch_in_epoch); epoch-end checkpoints store
            # epoch+1 with batch_in_epoch=0 (resume starts the next epoch).
            # Old checkpoints predate batch_in_epoch, so fall back to 0.
            start_epoch = int(state["epoch"])
            resume_batch = int(state.get("batch_in_epoch", 0))
            global_step = int(state["global_step"])
            # batch_in_epoch counts PER-RANK sharded batches, so a mid-epoch
            # resume at a different world size would silently skip the wrong
            # data. Refuse loudly instead of corrupting the run.
            ckpt_world = state.get("num_processes")
            if (
                ckpt_world is not None
                and int(ckpt_world) != accelerator.num_processes
                and resume_batch > 0
            ):
                raise RuntimeError(
                    f"checkpoint {latest} was saved mid-epoch with "
                    f"num_processes={ckpt_world} but this run has "
                    f"num_processes={accelerator.num_processes}; "
                    "batch_in_epoch is a per-rank index, so resuming at a "
                    "different world size would skip the wrong batches. "
                    "Resume with the same GPU count, or restart from an "
                    "epoch-end checkpoint (batch_in_epoch=0)."
                )
            accelerator.print(
                f"[resume] restored epoch={start_epoch}, "
                f"batch_in_epoch={resume_batch}, step={global_step}"
            )

    # ------------------------------------------------------------------
    # 10. MLflow experiment setup (main process only)
    # ------------------------------------------------------------------
    # Guard both calls to the main process so DDP ranks don't race to create
    # the same experiment concurrently, which can cause MLflow backend errors.
    if accelerator.is_main_process:
        mlflow.set_tracking_uri(str(mlflow_dir))
        mlflow.set_experiment("arxivlens-reranker")

    # Flatten the config to a single-level dict for MLflow params.
    flat_cfg = _Namespace(cfg_dict).as_dict()

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    # Only the main process opens an MLflow run; worker ranks use a no-op
    # context manager so the indented training code is identical for all ranks.
    # Log object returned to the caller (smoke test inspects these).
    step_losses: list[float] = []
    trained_batch_indices: dict[int, list[int]] = {}
    final_metrics: dict[str, float] | None = None

    _mlflow_ctx = mlflow.start_run() if accelerator.is_main_process else contextlib.nullcontext()
    with _mlflow_ctx:
        if accelerator.is_main_process:
            # Log the full config as MLflow params for experiment tracking.
            # MLflow param values must be strings; cast everything.
            mlflow.log_params({str(k): str(v) for k, v in flat_cfg.items()})

        for epoch in range(start_epoch, n_epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            # Reseed the shuffle generator so every epoch has a REPRODUCIBLE
            # order — the invariant mid-epoch resume relies on: a resumed run
            # replays this exact order and skips already-trained batches.
            train_gen.manual_seed(seed + epoch)
            trained_batch_indices.setdefault(epoch, [])

            for batch_idx, batch in enumerate(train_loader):
                # --- Mid-epoch resume skip ---
                # MUST be the first statement in the body: skipped batches must
                # not touch the optimizer, global_step, checkpointing, or eval.
                if epoch == start_epoch and batch_idx < resume_batch:
                    continue

                optimizer.zero_grad()

                # Forward pass: (B,) relevance logits.
                logits = model(batch["input_ids"], batch["attention_mask"])
                bce_loss = criterion(logits, batch["labels"])
                if loss_type == "listwise":
                    ranking_loss = listwise_softmax_loss(
                        logits, batch["labels"], batch["query_ids"]
                    )
                    loss = ranking_loss + bce_aux_weight * bce_loss
                else:
                    ranking_loss = None
                    loss = bce_loss

                # Backward through accelerator so mixed-precision scaling is
                # handled correctly for both bf16 and fp32 modes.
                accelerator.backward(loss)
                # Clip gradients before the optimizer step to prevent occasional
                # loss spikes from destabilizing training. Guarded so grad_clip
                # <= 0 disables clipping entirely.
                if grad_clip and grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_loss_sum += loss.item()
                epoch_steps += 1
                step_losses.append(loss.item())
                trained_batch_indices[epoch].append(batch_idx)

                # --- Per-step MLflow logging ---
                if accelerator.is_main_process:
                    # Current LR is the base lr multiplied by the scheduler scale.
                    current_lr = scheduler.get_last_lr()[0]
                    train_metrics = {
                        "train/loss": loss.item(),
                        "train/bce_loss": bce_loss.item(),
                        "train/lr": current_lr,
                    }
                    if ranking_loss is not None:
                        train_metrics["train/listwise_loss"] = ranking_loss.item()
                    mlflow.log_metrics(train_metrics, step=global_step)

                # --- Periodic checkpoint ---
                if global_step % checkpoint_every == 0:
                    _save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        checkpoint_dir,
                        cfg_dict,
                        batch_in_epoch=batch_idx + 1,  # next unseen batch
                    )

                # --- Periodic held-out eval ---
                # Eval runs on the MAIN process only: the val set is small, so
                # gathering across ranks (accelerator.gather) buys nothing and
                # would just add sync overhead. Non-main ranks skip eval and then
                # rejoin at wait_for_everyone() below, so no rank races ahead into
                # the next checkpoint write while main is still evaluating.
                if (
                    global_step % eval_every == 0
                    and val_loader is not None
                    and accelerator.is_main_process
                ):
                    metrics = _run_eval(model, val_loader, accelerator)
                    mlflow.log_metrics(
                        {_mlflow_safe(f"val/{k}"): v for k, v in metrics.items()},
                        step=global_step,
                    )
                    # Print a quick summary so Sol logs show progress.
                    ndcg10 = metrics.get("ndcg@10", float("nan"))
                    mrr_val = metrics.get("mrr", float("nan"))
                    accelerator.print(
                        f"[eval] step={global_step} "
                        f"ndcg@10={ndcg10:.4f} mrr={mrr_val:.4f}"
                    )
                if global_step % eval_every == 0 and val_loader is not None:
                    # Barrier so non-main ranks wait for main's eval to finish
                    # before anyone proceeds to the next checkpoint write.
                    accelerator.wait_for_everyone()

            # After the first (possibly partial) epoch, resume skipping is done.
            resume_batch = 0

            # --- End-of-epoch summary ---
            avg_loss = epoch_loss_sum / max(1, epoch_steps)
            accelerator.print(
                f"[epoch {epoch}] avg_train_loss={avg_loss:.4f} "
                f"steps={epoch_steps} global_step={global_step}"
            )
            if accelerator.is_main_process:
                mlflow.log_metrics(
                    {"train/epoch_loss": avg_loss},
                    step=global_step,
                )

            # Checkpoint at the end of every epoch. Save epoch + 1 with
            # batch_in_epoch=0 so resume starts the NEXT epoch: storing the
            # completed epoch as-is would make `range(start_epoch, ...)`
            # re-enter it and silently retrain the whole epoch (pushing
            # global_step and the cosine schedule past total_steps). The +1
            # also gives this file a different name from a step checkpoint
            # written at the last batch of the epoch, so it no longer
            # overwrites that valid resume point.
            _save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                checkpoint_dir,
                cfg_dict,
                batch_in_epoch=0,
            )

        accelerator.print("[done] training complete.")

        # Final eval over the full val set — main process only (see the
        # periodic-eval rationale above; the val set is small).
        #
        # Design decision O1: run the FAISS retrieve-then-rerank eval ONLY here
        # at FINAL eval, never on the periodic monitoring steps above (which keep
        # the cheap grouped-by-query_id eval). Design decision O4: if no index is
        # available (CI/CPU/local), fall back to the same grouped eval so nothing
        # breaks — final_metrics stays identical to the previous behaviour.
        if val_loader is not None and accelerator.is_main_process:
            retriever = _try_build_faiss_retriever(cfg, accelerator)
            if retriever is not None:
                try:
                    final_metrics = _run_faiss_eval(
                        model,
                        retriever,
                        val_dataset,
                        accelerator,
                        eval_num_candidates,
                        eval_passage_format,
                    )
                except Exception as exc:  # noqa: BLE001 — never fail the run on eval
                    accelerator.print(
                        f"[final eval] FAISS eval failed "
                        f"({type(exc).__name__}: {exc}); "
                        "falling back to grouped-by-query_id eval."
                    )
                    final_metrics = _run_eval(model, val_loader, accelerator)
            else:
                final_metrics = _run_eval(model, val_loader, accelerator)
            mlflow.log_metrics(
                {
                    _mlflow_safe(f"val/final_{k}"): v
                    for k, v in final_metrics.items()
                },
                step=global_step,
            )
            accelerator.print(
                "[final eval] "
                + "  ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
            )
        if val_loader is not None:
            # Barrier so non-main ranks don't exit the MLflow run context (and
            # the function) before main finishes the final eval + logging.
            accelerator.wait_for_everyone()

    return {
        "step_losses": step_losses,
        "trained_batch_indices": trained_batch_indices,
        "global_step": global_step,
        "final_metrics": final_metrics,
    }


def main() -> None:
    """Parse args, build the accelerator + tokenizer, then run training."""
    args = _build_parser().parse_args()

    # ------------------------------------------------------------------
    # 1. Load and merge config
    # ------------------------------------------------------------------
    cfg_dict = _load_yaml(args.config)
    cfg = _Namespace(cfg_dict)

    # CLI flags override the YAML where provided.
    if args.pairs is not None:
        cfg.training.pairs_file = str(args.pairs)
    if args.val_pairs is not None:
        cfg.training.val_pairs_file = str(args.val_pairs)
    if args.checkpoint_dir is not None:
        cfg.training.checkpoint_dir = str(args.checkpoint_dir)
    if args.mlflow_dir is not None:
        cfg.training.mlflow_dir = str(args.mlflow_dir)
    if args.epochs is not None:
        cfg.training.n_epochs = args.epochs
    if args.eval_index_path is not None:
        cfg.training.eval_index_path = str(args.eval_index_path)
    if args.eval_meta_path is not None:
        cfg.training.eval_meta_path = str(args.eval_meta_path)
    if args.eval_passage_format is not None:
        cfg.training.eval_passage_format = args.eval_passage_format

    # ------------------------------------------------------------------
    # 2. Accelerator — bf16 on A100, silent fallback to fp32 elsewhere
    # ------------------------------------------------------------------
    try:
        accelerator = Accelerator(mixed_precision="bf16")
    except (ValueError, RuntimeError):
        accelerator = Accelerator(mixed_precision="no")

    # ------------------------------------------------------------------
    # 3. Tokenizer
    # ------------------------------------------------------------------
    # AutoTokenizer.from_pretrained fetches bert-base-uncased from the HuggingFace
    # hub (or HF_HOME cache on Sol). The tokenizer is injected into the model so
    # the reranker never imports transformers directly — it only needs the protocol.
    tokenizer_name: str = cfg.training.tokenizer_name
    accelerator.print(f"[tokenizer] loading {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Store resolved CLI overrides so checkpoint provenance describes the
    # actual paths and hyperparameters used by this run.
    resolved_cfg_dict = cfg.as_dict_nested()
    run_training(
        cfg,
        tokenizer,
        accelerator=accelerator,
        resume=args.resume,
        cfg_dict=resolved_cfg_dict,
    )


if __name__ == "__main__":
    main()
