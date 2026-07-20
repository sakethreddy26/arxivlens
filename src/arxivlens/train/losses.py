"""Ranking losses used by the cross-encoder trainer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def listwise_softmax_loss(
    logits: Tensor,
    labels: Tensor,
    query_ids: Sequence[str],
) -> Tensor:
    """Cross-entropy over each query's complete candidate list.

    Each query group must contain exactly one positive. For group scores ``s``
    and gold index ``g``, the loss is ``logsumexp(s) - s[g]``. Minimizing it
    directly raises the gold passage above every negative instead of treating
    pairs as unrelated binary classifications.
    """
    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("logits and labels must both be 1-D")
    if logits.shape != labels.shape or logits.numel() != len(query_ids):
        raise ValueError("logits, labels, and query_ids must describe the same candidates")

    grouped_indices: dict[str, list[int]] = {}
    for index, query_id in enumerate(query_ids):
        grouped_indices.setdefault(str(query_id), []).append(index)

    losses: list[Tensor] = []
    for query_id, indices in grouped_indices.items():
        index_tensor = torch.tensor(indices, device=logits.device, dtype=torch.long)
        group_logits = logits.index_select(0, index_tensor)
        group_labels = labels.index_select(0, index_tensor)
        positive_indices = torch.nonzero(group_labels > 0.5, as_tuple=False).flatten()
        if positive_indices.numel() != 1:
            raise ValueError(
                f"query group {query_id!r} has {positive_indices.numel()} positives; "
                "listwise loss requires exactly one"
            )
        if group_logits.numel() < 2:
            raise ValueError(f"query group {query_id!r} has no negatives")
        gold_score = group_logits[positive_indices[0]]
        losses.append(torch.logsumexp(group_logits, dim=0) - gold_score)

    if not losses:
        raise ValueError("listwise loss received no query groups")
    return torch.stack(losses).mean()
