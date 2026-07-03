"""Ranking-quality metrics for the reranker held-out split.

Purpose
-------
After the from-scratch cross-encoder scores a set of candidate
passages for each query, we need to say *how good* that ranking is. This module
implements the standard information-retrieval metrics the README results table
reports — **nDCG@{5,10}, MRR, Recall@{1,5,10}** — as pure, dependency-light
functions (numpy only). Everything here is deterministic and CPU-only so the
same predictions always yield the same numbers.

What "relevant" means
---------------------
Under the synthetic labeling scheme, a candidate is **relevant** when its
ground-truth ``label == 1`` (the paper's true title/abstract pair); every other
candidate is a negative (``label == 0``). Binary relevance is the primary mode.
Graded relevance (integer gains > 1) is also supported by the DCG-based metrics
for completeness, and documented per function.

Conventions (be precise — this is read by reviewers)
----------------------------------------------------
* **Ranks are 1-based.** The top-scoring item is at rank 1.
* **DCG uses a log2 discount:** ``DCG@k = sum_{i=1..k} gain_i / log2(i + 1)``,
  so the rank-1 item is divided by ``log2(2) = 1`` (no discount) and rank 2 by
  ``log2(3)``. ``gain_i`` is the relevance label of the item at rank ``i``.
* **nDCG = DCG@k / IDCG@k**, where IDCG@k is the DCG of the ideal ordering
  (labels sorted descending). If ``IDCG@k == 0`` (no relevant items in the
  query at all) nDCG is defined here as ``0.0``.
* **Tie-breaking:** when two candidates share a predicted score, we sort by
  score descending using a **stable** sort (``numpy.argsort(kind="stable")`` on
  the negated scores). Stability means ties keep their *original input order*,
  so metrics are fully reproducible and never depend on hash/sort nondeterminism.
  Callers who want tie-break-by-label should order their input accordingly.

Input contract
--------------
The convenience entry point :func:`evaluate_rankings` takes an iterable of
per-query ``(scores, labels)`` tuples:

* ``scores``  — 1-D array-like of predicted relevance scores (higher = more
  relevant), one per candidate. These are the reranker's raw outputs; they need
  not be sorted.
* ``labels``  — 1-D array-like of the same length, the ground-truth relevance
  of each candidate *in the same order as* ``scores`` (``1`` = relevant, ``0``
  = irrelevant; integers > 1 allowed for graded relevance).

``scores[j]`` and ``labels[j]`` describe the **same** candidate ``j``. The two
arrays are zipped by position; there is no separate id column. Each query is
scored independently and the per-query metrics are averaged (macro-average)
across all queries. Queries with no candidates are skipped.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "dcg_at_k",
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "evaluate_rankings",
]

ArrayLike = Sequence[float] | np.ndarray


def _order_by_score(scores: ArrayLike) -> np.ndarray:
    """Return indices that sort ``scores`` descending, breaking ties stably.

    A stable sort on the negated scores means equal scores retain their
    original input order, making every downstream metric deterministic.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    # np.argsort is ascending; negate to get descending. kind="stable" keeps
    # the input order among equal (tied) scores.
    return np.argsort(-scores, kind="stable")


def _gains_in_predicted_order(scores: ArrayLike, labels: ArrayLike) -> np.ndarray:
    """Relevance labels reordered by descending predicted score (rank order)."""
    labels = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"scores and labels must be the same length, got "
            f"{scores.shape[0]} and {labels.shape[0]}"
        )
    order = _order_by_score(scores)
    return labels[order]


def _dcg(gains: np.ndarray, k: int) -> float:
    """DCG over an array of gains already in rank order (1-based ranks).

    ``DCG@k = sum_{i=1..k} gain_i / log2(i + 1)``. The rank-1 gain is undiscounted
    because ``log2(1 + 1) == 1``.
    """
    gains = gains[:k]
    if gains.size == 0:
        return 0.0
    ranks = np.arange(1, gains.size + 1)  # 1-based ranks
    discounts = np.log2(ranks + 1.0)
    return float(np.sum(gains / discounts))


def dcg_at_k(scores: ArrayLike, labels: ArrayLike, k: int) -> float:
    """Discounted Cumulative Gain at rank ``k`` for a single query.

    Candidates are ranked by descending ``scores`` (stable tie-break), then
    ``DCG@k = sum_{i=1..k} label_i / log2(i + 1)`` over the top ``k``.

    Args:
        scores: predicted relevance scores, one per candidate.
        labels: ground-truth relevance, aligned to ``scores`` by position.
        k: cutoff rank (1-based). ``k`` larger than the candidate count simply
            sums over all candidates.

    Returns:
        The DCG value (``0.0`` for an empty query).
    """
    gains = _gains_in_predicted_order(scores, labels)
    return _dcg(gains, k)


def ndcg_at_k(scores: ArrayLike, labels: ArrayLike, k: int) -> float:
    """Normalized DCG at rank ``k`` for a single query.

    ``nDCG@k = DCG@k / IDCG@k`` where IDCG@k is the DCG of the *ideal* ranking
    (labels sorted descending). Supports binary relevance (labels in {0, 1}) and
    graded relevance (integer gains > 1). If the query contains no relevant
    items (``IDCG@k == 0``) the result is ``0.0`` by definition.

    A perfect ranking — every relevant item above every irrelevant one — gives
    ``nDCG@k == 1.0``.

    Args:
        scores: predicted relevance scores, one per candidate.
        labels: ground-truth relevance, aligned to ``scores`` by position.
        k: cutoff rank (1-based).

    Returns:
        nDCG in ``[0.0, 1.0]``.
    """
    gains = _gains_in_predicted_order(scores, labels)
    dcg = _dcg(gains, k)
    # Ideal DCG: same gains sorted descending (the best achievable ordering).
    ideal_gains = np.sort(gains)[::-1]
    idcg = _dcg(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(scores: ArrayLike, labels: ArrayLike) -> float:
    """Reciprocal rank of the first relevant item for a single query.

    Ranks candidates by descending ``scores`` (stable tie-break) and returns
    ``1 / rank`` of the first item with ``label >= 1`` (1-based rank). If the
    first relevant item is at rank 3 the value is ``1/3``. If no candidate is
    relevant, returns ``0.0``.

    (The "mean" in MRR is applied across queries by :func:`evaluate_rankings`;
    this function returns the per-query reciprocal rank.)

    Args:
        scores: predicted relevance scores, one per candidate.
        labels: ground-truth relevance, aligned to ``scores`` by position.

    Returns:
        Reciprocal rank in ``[0.0, 1.0]``.
    """
    gains = _gains_in_predicted_order(scores, labels)
    relevant = np.nonzero(gains >= 1.0)[0]
    if relevant.size == 0:
        return 0.0
    first_rank = int(relevant[0]) + 1  # 0-based index -> 1-based rank
    return 1.0 / first_rank


def recall_at_k(scores: ArrayLike, labels: ArrayLike, k: int) -> float:
    """Recall at rank ``k`` for a single query.

    ``Recall@k = (# relevant items in the top k) / (total # relevant items)``.
    With exactly one relevant item per query — the common case under the section 5
    synthetic protocol — this reduces to **hit-rate@k**: 1.0 if the single
    relevant item is in the top ``k``, else 0.0.

    Candidates are ranked by descending ``scores`` (stable tie-break). ``k``
    larger than the candidate count is fine — it just considers all candidates.
    If the query has no relevant items, returns ``0.0``.

    Args:
        scores: predicted relevance scores, one per candidate.
        labels: ground-truth relevance, aligned to ``scores`` by position.
        k: cutoff rank (1-based).

    Returns:
        Recall in ``[0.0, 1.0]``.
    """
    gains = _gains_in_predicted_order(scores, labels)
    total_relevant = int(np.count_nonzero(gains >= 1.0))
    if total_relevant == 0:
        return 0.0
    hits = int(np.count_nonzero(gains[:k] >= 1.0))
    return hits / total_relevant


# Metric cutoffs the README results table reports.
_NDCG_KS = (5, 10)
_RECALL_KS = (1, 5, 10)


def evaluate_rankings(
    queries: Iterable[tuple[ArrayLike, ArrayLike]],
) -> dict[str, float]:
    """Aggregate ranking metrics over a held-out set of queries.

    Input contract (see also the module docstring): ``queries`` is an iterable
    of ``(scores, labels)`` tuples, one tuple per query, where ``scores`` and
    ``labels`` are equal-length array-likes aligned by candidate position
    (``scores[j]``/``labels[j]`` describe the same candidate; ``label == 1``
    means relevant). Predicted scores need not be pre-sorted.

    Each query is scored independently and the results are **macro-averaged**
    (unweighted mean) across all non-empty queries. Empty queries (no
    candidates) are skipped; if every query is empty, all metrics are ``0.0``.

    Returns:
        A dict with keys ``ndcg@5``, ``ndcg@10``, ``mrr``, ``recall@1``,
        ``recall@5``, ``recall@10``. The trainer logs these to MLflow and
        the README tabulates them.
    """
    keys = (
        [f"ndcg@{k}" for k in _NDCG_KS]
        + ["mrr"]
        + [f"recall@{k}" for k in _RECALL_KS]
    )
    sums = {key: 0.0 for key in keys}
    n_queries = 0

    for scores, labels in queries:
        scores_arr = np.asarray(scores, dtype=np.float64).ravel()
        if scores_arr.size == 0:
            continue  # nothing to rank; skip rather than divide by zero
        n_queries += 1
        for k in _NDCG_KS:
            sums[f"ndcg@{k}"] += ndcg_at_k(scores, labels, k)
        sums["mrr"] += mrr(scores, labels)
        for k in _RECALL_KS:
            sums[f"recall@{k}"] += recall_at_k(scores, labels, k)

    if n_queries == 0:
        return {key: 0.0 for key in keys}
    return {key: sums[key] / n_queries for key in keys}
