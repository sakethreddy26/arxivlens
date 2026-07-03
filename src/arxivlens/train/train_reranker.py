"""Training script for the ArXivLens cross-encoder reranker.

Pipeline overview
-----------------
This script wires together every component built in earlier phases:

    argparse CLI
        -> load reranker.yaml
        -> build TransformerConfig + CrossEncoderReranker
        -> build PairDataset (train + optional val)
        -> Accelerator (bf16 mixed precision on A100; CPU fallback)
        -> AdamW + linear warmup scheduler
        -> BCE training loop
        -> MLflow metric logging
        -> checkpoint save/resume

Loss choice — BCEWithLogitsLoss
--------------------------------
We use binary cross-entropy rather than a margin/pairwise ranking loss because:
  1. Each (query, passage) pair is an *independent* binary classification task:
     label 1 = relevant, label 0 = irrelevant.
  2. BCE is numerically stable: ``BCEWithLogitsLoss`` fuses the sigmoid with the
     log so there is never an explicit sigmoid followed by a log-of-small-number.
  3. Margin ranking loss would require constructing (positive, negative) pairs
     *within every batch*, coupling the data pipeline to batch structure. That
     complicates the dataloader and gives no clear accuracy benefit given our
     fixed 1:4 positive:negative ratio (from ``build_pairs``), where BCE already
     sees four times as many negatives as positives and calibrates accordingly.

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
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from arxivlens.data.dataset import PairDataset, collate_fn
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.model.transformer import TransformerConfig
from arxivlens.train.eval import evaluate_rankings


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
        epoch: current epoch index (0-based).
        step: global optimizer step count.
        checkpoint_dir: directory to write into; created if absent.
        cfg_dict: raw YAML config dict for provenance.
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
                logits = model(batch["input_ids"], batch["attention_mask"])  # (B,)
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
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args, build everything, run the training loop."""
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
    tokenizer_name: str = cfg.training.tokenizer_name
    max_input_length: int = int(cfg.model.max_input_length)

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
    # 4. Tokenizer
    # ------------------------------------------------------------------
    # AutoTokenizer.from_pretrained fetches bert-base-uncased from the HuggingFace
    # hub (or HF_HOME cache on Sol). The tokenizer is injected into the model so
    # the reranker never imports transformers directly — it only needs the protocol.
    accelerator.print(f"[tokenizer] loading {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        # Hold out val_fraction of the training set for monitoring.
        n_val = max(1, int(len(full_dataset) * val_fraction))
        n_train = len(full_dataset) - n_val
        accelerator.print(
            f"[data] auto-splitting: {n_train} train / {n_val} val "
            f"(val_fraction={val_fraction})"
        )
        # Use a seeded generator so the split is reproducible.
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle every epoch for better gradient diversity
        collate_fn=collate_fn,
        drop_last=False,        # keep partial last batch; BCE handles any size
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
        f"{len(train_loader)} train batches / epoch"
    )

    # ------------------------------------------------------------------
    # 7. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    # BCE is appropriate because each (query, passage) pair is an independent
    # binary classification; see module docstring for the full justification.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Linear warmup: ramp from 0 -> lr over warmup_steps, then hold at lr.
    # A constant post-warmup rate is simple and works well for short fine-tuning
    # runs. Cosine decay would help for longer schedules but adds a hyperparameter.
    def lr_lambda(step: int) -> float:
        """Return the LR scale factor at ``step`` (multiplied by base lr)."""
        if step < warmup_steps:
            # Avoid division by zero when warmup_steps == 0.
            return step / max(1, warmup_steps)
        return 1.0  # full base lr after warmup

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 8. Accelerate wrapping (handles device placement, DDP, mixed prec.)
    # ------------------------------------------------------------------
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ------------------------------------------------------------------
    # 9. Optional resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume:
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
            # Restore model weights into the (possibly DDP-wrapped) model.
            accelerator.unwrap_model(model).load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = int(state["epoch"])      # resume at the next epoch
            global_step = int(state["global_step"])
            accelerator.print(
                f"[resume] restored epoch={start_epoch}, step={global_step}"
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

            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass: (B,) relevance logits.
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])

                # Backward through accelerator so mixed-precision scaling is
                # handled correctly for both bf16 and fp32 modes.
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_loss_sum += loss.item()
                epoch_steps += 1

                # --- Per-step MLflow logging ---
                if accelerator.is_main_process:
                    # Current LR is the base lr multiplied by the scheduler scale.
                    current_lr = scheduler.get_last_lr()[0]
                    mlflow.log_metrics(
                        {
                            "train/loss": loss.item(),
                            "train/lr": current_lr,
                        },
                        step=global_step,
                    )

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
                    )

                # --- Periodic held-out eval ---
                if global_step % eval_every == 0 and val_loader is not None:
                    metrics = _run_eval(model, val_loader, accelerator)
                    if accelerator.is_main_process:
                        mlflow.log_metrics(
                            {f"val/{k}": v for k, v in metrics.items()},
                            step=global_step,
                        )
                        # Print a quick summary so Sol logs show progress.
                        ndcg10 = metrics.get("ndcg@10", float("nan"))
                        mrr_val = metrics.get("mrr", float("nan"))
                        accelerator.print(
                            f"[eval] step={global_step} "
                            f"ndcg@10={ndcg10:.4f} mrr={mrr_val:.4f}"
                        )

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

            # Checkpoint at the end of every epoch regardless of step count,
            # so Sol jobs that hit the wall-clock cap mid-epoch don't lose the
            # partial progress — but also so a complete epoch is always saved.
            _save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch + 1,  # epoch+1 so resume skips this completed epoch
                global_step,
                checkpoint_dir,
                cfg_dict,
            )

        accelerator.print("[done] training complete.")

        # Final eval over the full val set.
        if val_loader is not None:
            final_metrics = _run_eval(model, val_loader, accelerator)
            if accelerator.is_main_process:
                mlflow.log_metrics(
                    {f"val/final_{k}": v for k, v in final_metrics.items()},
                    step=global_step,
                )
                accelerator.print(
                    "[final eval] "
                    + "  ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
                )


if __name__ == "__main__":
    main()
