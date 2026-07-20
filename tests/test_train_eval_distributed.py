"""Regression tests for main-rank validation with a distributed wrapper."""

from __future__ import annotations

import torch
from torch import nn

from arxivlens.train.train_reranker import _run_eval


class _Scorer(nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        return input_ids[:, 0].float()


class _WrapperThatMustNotRun:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("main-only eval called the distributed wrapper")


class _FakeAccelerator:
    device = torch.device("cpu")

    def __init__(self, unwrapped: nn.Module) -> None:
        self.unwrapped = unwrapped
        self.unwrap_calls = 0

    def unwrap_model(self, _model):
        self.unwrap_calls += 1
        return self.unwrapped


def test_run_eval_uses_unwrapped_model_for_main_only_validation() -> None:
    scorer = _Scorer()
    scorer.train()
    accelerator = _FakeAccelerator(scorer)
    batch = {
        "input_ids": torch.tensor([[2], [1]], dtype=torch.long),
        "attention_mask": torch.ones((2, 1), dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0]),
        "query_ids": ["q0", "q0"],
    }

    metrics = _run_eval(_WrapperThatMustNotRun(), [batch], accelerator)

    assert accelerator.unwrap_calls == 1
    assert scorer.training is True
    assert metrics["mrr"] == 1.0
