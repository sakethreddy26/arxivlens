"""CPU tests for query-group ranking losses."""

from __future__ import annotations

import pytest
import torch

from arxivlens.train.losses import listwise_softmax_loss


def test_listwise_loss_rewards_gold_at_top() -> None:
    labels = torch.tensor([1.0, 0.0, 0.0])
    query_ids = ["q0", "q0", "q0"]

    good = listwise_softmax_loss(torch.tensor([4.0, 0.0, -1.0]), labels, query_ids)
    bad = listwise_softmax_loss(torch.tensor([-1.0, 4.0, 0.0]), labels, query_ids)

    assert good.item() < bad.item()


def test_listwise_loss_averages_groups_and_backpropagates() -> None:
    logits = torch.tensor([2.0, 0.0, -1.0, 1.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    loss = listwise_softmax_loss(logits, labels, ["q0", "q0", "q1", "q1"])

    expected = (
        torch.logsumexp(logits[:2], dim=0)
        - logits[0]
        + torch.logsumexp(logits[2:], dim=0)
        - logits[3]
    ) / 2
    assert loss.item() == pytest.approx(expected.item())

    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 4


def test_listwise_loss_rejects_malformed_group() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        listwise_softmax_loss(
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            ["q0", "q0"],
        )
