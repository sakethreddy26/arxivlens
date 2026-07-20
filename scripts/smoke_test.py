"""CPU-only smoke test for the ArXivLens reranker training loop.

This is the end-to-end GATE for the training code: it drives the REAL
``run_training`` entry point (the same function ``main`` calls) on a tiny
CPU-only config with an offline stub tokenizer, and asserts six properties
that together prove the loop, checkpointing, resume (mid-epoch AND
epoch-boundary), and eval are all wired correctly:

    (a) loss decreases  -- the model actually learns on the toy data.
    (b) a checkpoint file is written.
    (c) mid-epoch resume is EXACT -- a resumed run continues from the saved
        global_step and batch, retraining no batch it already trained and
        skipping none it hadn't.
    (d) eval is NON-degenerate -- at least one query group has >1 candidate and
        MRR is a real value in (0, 1].
    (e) epoch-boundary resume -- resuming from an epoch-END checkpoint
        (batch_in_epoch == 0) starts the NEXT epoch and never silently
        retrains the completed one.
    (f) listwise mode -- complete query groups train end-to-end, produce finite
        losses, and save the requested ranking objective in checkpoint provenance.

No network, no GPU, no HuggingFace download: the tokenizer is the deterministic
``StubTokenizer`` from ``tests/test_dataset.py`` and everything runs in a temp
directory that is cleaned up on exit.

Run it directly::

    python scripts/smoke_test.py

Exit code is 0 only when all five assertions pass.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Make ``src/`` importable when run as a plain script (no install required),
# mirroring scripts/build_pairs.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# tests/ holds the offline StubTokenizer we reuse here.
_TESTS = _REPO_ROOT / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import torch  # noqa: E402

# Neutralize MLflow so the smoke test never touches a tracking backend: some
# MLflow versions reject metric names containing '@' (e.g. "val/ndcg@5"), which
# is irrelevant to what this test verifies (the training/resume/eval logic).
# This is confined to the smoke test and does not alter training behaviour.
import mlflow as _mlflow  # noqa: E402


def _mlflow_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class _NullRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_mlflow.set_tracking_uri = _mlflow_noop
_mlflow.set_experiment = _mlflow_noop
_mlflow.log_params = _mlflow_noop
_mlflow.log_metrics = _mlflow_noop
_mlflow.start_run = lambda *a, **k: _NullRun()

from arxivlens.data.pairs import build_pairs  # noqa: E402
from arxivlens.train.train_reranker import (  # noqa: E402
    _load_latest_checkpoint,
    _Namespace,
    run_training,
)
from test_dataset import StubTokenizer  # noqa: E402  (offline, no HF download)


# --------------------------------------------------------------------------- #
# Toy data                                                                     #
# --------------------------------------------------------------------------- #

def _make_records(n: int = 12) -> list[dict[str, str]]:
    """Build ``n`` fake corpus records with distinct-ish title/abstract text.

    Each record carries an ``id`` so ``build_pairs`` uses it as the ``query_id``
    (grouping the positive with its negatives under one ranking).
    """
    topics = [
        "attention transformer", "graph neural network", "diffusion model",
        "reinforcement learning", "contrastive representation", "language model",
        "convolutional vision", "bayesian inference", "optimization gradient",
        "retrieval augmentation", "speech recognition", "federated learning",
    ]
    records: list[dict[str, str]] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        records.append(
            {
                "id": f"p{i}",
                "title": f"{topic} paper number {i}",
                "abstract": f"this abstract studies {topic} with method {i} on data {i}",
            }
        )
    return records


def _fake_neighbor_fn(n_records: int):
    """Deterministic neighbour lookup: return a rotating set of other indices.

    Gives each query several hard negatives so every ``query_id`` ends up with
    MULTIPLE candidates -- required for the non-degenerate eval assertion.
    """

    def neighbor_fn(query_index: int, k: int) -> list[int]:
        out: list[int] = []
        j = query_index
        while len(out) < k:
            j = (j + 1) % n_records
            if j != query_index and j not in out:
                out.append(j)
            if len(out) >= n_records - 1:
                break
        return out

    return neighbor_fn


def _write_pairs(path: Path) -> int:
    """Write ~50 pairs (from 12 records, 1:4 pos:neg) to ``path``; return count."""
    records = _make_records(12)
    neighbor_fn = _fake_neighbor_fn(len(records))
    rng = random.Random(0)
    pairs = list(
        build_pairs(records, neighbor_fn, rng, n_hard=2, n_easy=2)
    )
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")
    return len(pairs)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

def _make_cfg(
    tmp: Path,
    pairs_file: Path,
    vocab_size: int,
    n_epochs: int = 2,
) -> _Namespace:
    """Build the tiny CPU config as a nested dict wrapped in ``_Namespace``."""
    cfg_dict: dict[str, Any] = {
        "model": {
            "vocab_size": vocab_size,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 64,
            "max_len": 64,
            "dropout": 0.1,
            "max_input_length": 64,
        },
        "training": {
            "pairs_file": str(pairs_file),
            "val_pairs_file": "",  # force auto-split so we always have val data
            "checkpoint_dir": str(tmp / "checkpoints"),
            # MLflow needs a valid tracking URI; a bare Windows path with
            # backslashes is rejected, so hand it an explicit file:// URI.
            "mlflow_dir": (tmp / "mlruns").as_uri(),
            "learning_rate": 1.0e-3,
            "batch_size": 4,
            "n_epochs": n_epochs,
            "warmup_steps": 2,
            "lr_schedule": "constant",
            "grad_clip": 1.0,
            "checkpoint_every_steps": 3,
            "eval_every_steps": 3,
            "val_fraction": 0.2,
            "seed": 42,
            "tokenizer_name": "stub",
        },
    }
    return _Namespace(cfg_dict)


# --------------------------------------------------------------------------- #
# Assertions                                                                   #
# --------------------------------------------------------------------------- #

def _report(name: str, ok: bool, detail: str) -> bool:
    """Print a single PASS/FAIL line and return ``ok`` unchanged."""
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arxivlens_smoke_"))
    results: list[bool] = []
    try:
        tokenizer = StubTokenizer()
        vocab_size = StubTokenizer._vocab_size
        pairs_file = tmp / "pairs.jsonl"
        n_pairs = _write_pairs(pairs_file)
        print(f"[setup] wrote {n_pairs} pairs to {pairs_file}")

        checkpoint_dir = tmp / "checkpoints"

        # ---------------------------------------------------------------- #
        # Full fresh run (fresh accelerator each call; None -> auto CPU fp32)
        # ---------------------------------------------------------------- #
        cfg = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=2)
        log = run_training(cfg, tokenizer, accelerator=None, resume=False)

        step_losses = log["step_losses"]
        trained = log["trained_batch_indices"]
        final_metrics = log["final_metrics"]

        # (a) loss decreases: mean(last third) < mean(first third).
        n = len(step_losses)
        third = max(1, n // 3)
        first_third = step_losses[:third]
        last_third = step_losses[-third:]
        mean_first = sum(first_third) / len(first_third)
        mean_last = sum(last_third) / len(last_third)
        results.append(_report(
            "(a) loss decreases",
            mean_last < mean_first,
            f"mean(first third)={mean_first:.4f} -> mean(last third)={mean_last:.4f} "
            f"over {n} steps",
        ))

        # (b) a checkpoint file was written.
        ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch*_step*.pt"))
        results.append(_report(
            "(b) checkpoint written",
            len(ckpts) > 0,
            f"{len(ckpts)} checkpoint file(s), e.g. "
            f"{ckpts[0].name if ckpts else '<none>'}",
        ))

        # ---------------------------------------------------------------- #
        # (c) mid-epoch resume correctness.
        # Re-run from scratch into a FRESH checkpoint dir, then truncate the
        # checkpoint history to the first MID-EPOCH step-checkpoint (one with
        # batch_in_epoch > 0) and resume from it. The resumed run must continue
        # from the saved step, re-enter epoch 0 at the saved batch, and never
        # retrain a batch it already trained.
        # ---------------------------------------------------------------- #
        resume_dir = tmp / "resume_ckpts"
        cfg_c = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=2)
        cfg_c.training.checkpoint_dir = str(resume_dir)
        log_pre = run_training(cfg_c, tokenizer, accelerator=None, resume=False)

        # Find the first mid-epoch checkpoint (epoch 0, batch_in_epoch > 0).
        all_ckpts = sorted(resume_dir.glob("checkpoint_epoch*_step*.pt"))
        chosen: Path | None = None
        chosen_state: dict[str, Any] | None = None
        for ck in all_ckpts:
            st = torch.load(ck, map_location="cpu")
            if int(st["epoch"]) == 0 and int(st.get("batch_in_epoch", 0)) > 0:
                chosen = ck
                chosen_state = st
                break

        if chosen is None or chosen_state is None:
            results.append(_report(
                "(c) resume correctness", False,
                "no mid-epoch (batch_in_epoch>0) checkpoint was produced",
            ))
        else:
            saved_step = int(chosen_state["global_step"])
            saved_batch = int(chosen_state["batch_in_epoch"])
            # Batches epoch 0 trained BEFORE the kill: 0 .. saved_batch-1.
            pre_kill_epoch0 = list(range(saved_batch))

            # Remove every checkpoint AFTER the chosen one so resume loads it.
            for ck in all_ckpts:
                if ck != chosen and ck.name > chosen.name:
                    ck.unlink()

            # Resume: fresh accelerator, same config, resume=True.
            cfg_r = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=2)
            cfg_r.training.checkpoint_dir = str(resume_dir)
            log_post = run_training(cfg_r, tokenizer, accelerator=None, resume=True)

            post_epoch0 = log_post["trained_batch_indices"].get(0, [])
            first_trained = post_epoch0[0] if post_epoch0 else None

            continues = log_post["global_step"] > saved_step
            first_batch_ok = first_trained == saved_batch
            no_overlap = set(pre_kill_epoch0).isdisjoint(set(post_epoch0))

            ok_c = continues and first_batch_ok and no_overlap
            results.append(_report(
                "(c) resume correctness",
                ok_c,
                f"saved step={saved_step} batch={saved_batch}; resumed "
                f"first trained batch={first_trained}, final step="
                f"{log_post['global_step']}, "
                f"continues={continues} first_batch_match={first_batch_ok} "
                f"no_overlap={no_overlap}",
            ))

        # ---------------------------------------------------------------- #
        # (d) non-degenerate eval: reconstruct grouping to confirm a
        # multi-candidate query exists, and MRR is a genuine value in (0, 1].
        # ---------------------------------------------------------------- #
        multi = _has_multi_candidate_query(pairs_file, cfg)
        mrr = float(final_metrics["mrr"]) if final_metrics else float("nan")
        ok_d = multi and (0.0 < mrr <= 1.0)
        results.append(_report(
            "(d) non-degenerate eval",
            ok_d,
            f"multi_candidate_query={multi}, mrr={mrr:.4f} "
            f"(want 0 < mrr <= 1)",
        ))

        # ---------------------------------------------------------------- #
        # (e) epoch-boundary resume: a 1-epoch run's LATEST checkpoint is the
        # epoch-end one (stored epoch=1, batch_in_epoch=0). Resuming it with
        # n_epochs=2 must train ONLY epoch 1 -- zero batches of epoch 0
        # retrained -- and advance global_step by exactly one epoch's batches.
        # This is the path assertion (c) structurally cannot cover (it
        # filters for batch_in_epoch > 0).
        # ---------------------------------------------------------------- #
        end_dir = tmp / "epochend_ckpts"
        cfg_e = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=1)
        cfg_e.training.checkpoint_dir = str(end_dir)
        log_e_pre = run_training(cfg_e, tokenizer, accelerator=None, resume=False)
        pre_step = int(log_e_pre["global_step"])

        latest_e = _load_latest_checkpoint(end_dir)
        st_e = torch.load(latest_e, map_location="cpu") if latest_e else {}
        is_epoch_end = (
            int(st_e.get("batch_in_epoch", -1)) == 0
            and int(st_e.get("epoch", -1)) == 1
        )

        cfg_e2 = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=2)
        cfg_e2.training.checkpoint_dir = str(end_dir)
        log_e_post = run_training(cfg_e2, tokenizer, accelerator=None, resume=True)

        trained_e = log_e_post["trained_batch_indices"]
        retrained_epoch0 = trained_e.get(0, [])
        epoch1_batches = trained_e.get(1, [])
        step_ok = (
            int(log_e_post["global_step"]) == pre_step + len(epoch1_batches)
        )
        ok_e = (
            is_epoch_end
            and not retrained_epoch0
            and len(epoch1_batches) > 0
            and epoch1_batches[0] == 0
            and step_ok
        )
        results.append(_report(
            "(e) epoch-boundary resume",
            ok_e,
            f"latest_ckpt_is_epoch_end={is_epoch_end}, "
            f"epoch0_batches_retrained={len(retrained_epoch0)} (want 0), "
            f"epoch1_batches={len(epoch1_batches)} starting at "
            f"{epoch1_batches[0] if epoch1_batches else '<none>'}, "
            f"step {pre_step} -> {log_e_post['global_step']} "
            f"(exact_advance={step_ok})",
        ))

        # ---------------------------------------------------------------- #
        # (f) Ranking-aligned listwise mode: this is the objective used by
        # the improved Sol experiment. Exercise the real grouped dataloader,
        # loss, optimizer, checkpoint, and eval path before an A100 job starts.
        # ---------------------------------------------------------------- #
        listwise_dir = tmp / "listwise_ckpts"
        cfg_l = _make_cfg(tmp, pairs_file, vocab_size, n_epochs=1)
        cfg_l.training.checkpoint_dir = str(listwise_dir)
        cfg_l.training.loss_type = "listwise"
        cfg_l.training.queries_per_batch = 2
        cfg_l.training.bce_aux_weight = 0.1
        cfg_l.training.weight_decay = 0.01
        log_l = run_training(cfg_l, tokenizer, accelerator=None, resume=False)

        listwise_ckpt = _load_latest_checkpoint(listwise_dir)
        listwise_state = (
            torch.load(listwise_ckpt, map_location="cpu") if listwise_ckpt else {}
        )
        listwise_losses = log_l["step_losses"]
        finite_losses = bool(listwise_losses) and all(
            torch.isfinite(torch.tensor(value)).item() for value in listwise_losses
        )
        saved_loss_type = (
            listwise_state.get("config", {})
            .get("training", {})
            .get("loss_type")
        )
        ok_f = (
            finite_losses
            and listwise_ckpt is not None
            and saved_loss_type == "listwise"
            and int(log_l["global_step"]) > 0
        )
        results.append(_report(
            "(f) listwise training path",
            ok_f,
            f"steps={log_l['global_step']}, finite_losses={finite_losses}, "
            f"checkpoint={listwise_ckpt.name if listwise_ckpt else '<none>'}, "
            f"saved_loss_type={saved_loss_type!r}",
        ))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = all(results) and len(results) == 6
    print()
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
    return 0 if passed else 1


def _has_multi_candidate_query(pairs_file: Path, cfg: _Namespace) -> bool:
    """Return True if the auto-split val set has >=1 query group with >1 pair.

    Reconstructs the SAME seeded group-wise auto-split ``run_training``
    performs (via the shared ``group_split_indices`` helper) so we verify
    eval saw a genuine multi-candidate ranking, not the degenerate
    one-candidate-per-query view the earlier fix removed.
    """
    from arxivlens.data.dataset import PairDataset, group_split_indices

    seed = int(cfg.training.seed)
    val_fraction = float(cfg.training.val_fraction)
    max_len = int(cfg.model.max_input_length)
    tokenizer = StubTokenizer()

    full = PairDataset(str(pairs_file), tokenizer, max_length=max_len)
    qids = full.query_ids()
    _train_idx, val_idx = group_split_indices(qids, val_fraction, seed)

    counts: dict[str, int] = {}
    for idx in val_idx:
        counts[qids[idx]] = counts.get(qids[idx], 0) + 1
    return any(c > 1 for c in counts.values())


if __name__ == "__main__":
    raise SystemExit(main())
