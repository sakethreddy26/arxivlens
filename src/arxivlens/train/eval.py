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

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "dcg_at_k",
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "evaluate_rankings",
    "build_retrieval_eval_queries",
    "rank_diagnostics",
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


def _first_relevant_rank(scores: ArrayLike, labels: ArrayLike) -> int | None:
    """Return the 1-based rank of the first relevant item, or ``None``."""
    gains = _gains_in_predicted_order(scores, labels)
    relevant = np.nonzero(gains >= 1.0)[0]
    if relevant.size == 0:
        return None
    return int(relevant[0]) + 1


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


def rank_diagnostics(
    reranker_queries: Sequence[tuple[ArrayLike, ArrayLike]],
    retrieval_queries: Sequence[tuple[ArrayLike, ArrayLike]],
) -> dict[str, Any]:
    """Summarize how reranking moves the gold item relative to FAISS order.

    The regular metrics say whether each ranking is good. This diagnostic says
    whether the reranker actually changed the gold paper's rank, which matters
    when the retrieval baseline is already saturated near rank 1.
    """
    if len(reranker_queries) != len(retrieval_queries):
        raise ValueError(
            "reranker_queries and retrieval_queries must be aligned, got "
            f"{len(reranker_queries)} and {len(retrieval_queries)}"
        )

    rows: list[dict[str, Any]] = []
    hard_reranker: list[tuple[ArrayLike, ArrayLike]] = []
    hard_retrieval: list[tuple[ArrayLike, ArrayLike]] = []
    improved = same = worsened = gold_missed = empty = 0
    deltas: list[int] = []

    for query_index, ((rr_scores, rr_labels), (rt_scores, rt_labels)) in enumerate(
        zip(reranker_queries, retrieval_queries)
    ):
        if len(rr_scores) == 0:
            empty += 1
            rows.append(
                {
                    "query_index": query_index,
                    "candidate_count": 0,
                    "gold_retrieved": False,
                    "faiss_rank": None,
                    "reranker_rank": None,
                    "rank_delta": None,
                    "outcome": "empty",
                    "hard_query": False,
                }
            )
            continue

        faiss_rank = _first_relevant_rank(rt_scores, rt_labels)
        reranker_rank = _first_relevant_rank(rr_scores, rr_labels)
        if faiss_rank is None or reranker_rank is None:
            gold_missed += 1
            outcome = "missed"
            rank_delta = None
            hard_query = False
        else:
            rank_delta = faiss_rank - reranker_rank
            deltas.append(rank_delta)
            hard_query = faiss_rank > 1
            if hard_query:
                hard_reranker.append((rr_scores, rr_labels))
                hard_retrieval.append((rt_scores, rt_labels))
            if rank_delta > 0:
                improved += 1
                outcome = "improved"
            elif rank_delta < 0:
                worsened += 1
                outcome = "worsened"
            else:
                same += 1
                outcome = "same"

        rows.append(
            {
                "query_index": query_index,
                "candidate_count": len(rr_scores),
                "gold_retrieved": faiss_rank is not None,
                "faiss_rank": faiss_rank,
                "reranker_rank": reranker_rank,
                "rank_delta": rank_delta,
                "outcome": outcome,
                "hard_query": hard_query,
            }
        )

    total_nonempty = len(reranker_queries) - empty
    return {
        "summary": {
            "n_queries": len(reranker_queries),
            "n_nonempty": total_nonempty,
            "n_empty": empty,
            "n_gold_missed": gold_missed,
            "n_hard_queries": len(hard_reranker),
            "improved": improved,
            "same": same,
            "worsened": worsened,
            "mean_rank_delta": float(np.mean(deltas)) if deltas else 0.0,
            "median_rank_delta": float(np.median(deltas)) if deltas else 0.0,
        },
        "hard_metrics": {
            "reranker": evaluate_rankings(hard_reranker),
            "retrieval_only": evaluate_rankings(hard_retrieval),
        },
        "per_query": rows,
    }


# --------------------------------------------------------------------------- #
# Retrieve-then-rerank candidate assembly (dependency-injected, faiss-free)    #
# --------------------------------------------------------------------------- #


def _normalize_id(paper_id: Any) -> str:
    """Canonicalize a paper id for equality comparison.

    Retrieval meta (``FaissRetriever``) sources the id from ``paper_id`` /
    ``id`` / ``arxiv_id`` while the held-out records carry ``id``; both are
    coerced to ``str`` here and stripped of surrounding whitespace so the two
    sides compare on a level field.
    """
    return str(paper_id).strip()


def _looks_like_positional_index(value: str) -> bool:
    """True if ``value`` looks like a bare positional index, not a real id.

    ``pairs.py`` falls back to ``str(positional_index)`` when a record has no
    ``id`` field, so a query id of e.g. ``"7"`` means the reconstruction lost
    the real arXiv id and label matching against ``paper_id`` would be
    meaningless. Real arXiv ids always contain a non-digit (``.`` or ``/``).
    """
    return value.isdigit()


def _candidate_passage(candidate: Mapping[str, Any], passage_format: str) -> str:
    """Build candidate text using an explicit, validated evaluation format."""
    abstract = "" if candidate.get("abstract") is None else str(candidate["abstract"])
    if passage_format == "abstract":
        return abstract.strip()
    if passage_format == "title_abstract":
        title = "" if candidate.get("title") is None else str(candidate["title"])
        return f"{title} {abstract}".strip()
    raise ValueError(
        "passage_format must be 'abstract' or 'title_abstract', "
        f"got {passage_format!r}"
    )


def build_retrieval_eval_queries(
    eval_records: Iterable[Mapping[str, Any]],
    retriever: Any,
    reranker: Any,
    num_candidates: int,
    *,
    with_retrieval_baseline: bool = False,
    passage_format: str = "title_abstract",
) -> (
    list[tuple[list[float], list[float]]]
    | tuple[
        list[tuple[list[float], list[float]]],
        list[tuple[list[float], list[float]]],
    ]
):
    """Assemble ``(scores, labels)`` per query via real retrieve-then-rerank.

    This is the non-degenerate replacement for scoring a query against only its
    own ~5 synthetic pairs (which saturates recall@k at 1.0). For each held-out
    record it retrieves ``num_candidates`` real candidates from the FAISS
    corpus, reranks them with the cross-encoder, and labels the query's own
    paper as the single positive — so the metrics measure whether the reranker
    can float the gold paper to the top of a realistic ~50-candidate list.

    Dependency injection (faiss-free): ``retriever`` and ``reranker`` are passed
    in, so this module never imports ``faiss``/``torch`` at top level and stays
    importable in CI. In production pass a ``FaissRetriever`` and a
    ``CrossEncoderReranker``; tests pass small fakes.

    Per record the flow is:

    * ``candidates = retriever.retrieve(query=title, k=num_candidates)`` — a
      list of dicts, each with at least ``paper_id``, ``title``, ``abstract``,
      in decreasing similarity order.
    * ``passages = ["{title} {abstract}".strip()]`` built from **each
      candidate's own** title/abstract (matching ``retrieve/pipeline.py``), NOT
      the query's.
    * ``scores = reranker.score(query=title, passages=passages)`` — a length-N
      array-like (a torch tensor in production) of relevance logits.
    * ``labels[j] = 1.0`` iff the candidate's normalized ``paper_id`` equals the
      record's normalized ``id``, else ``0.0``.

    Labeling scheme: exactly one positive per query — the query's own paper.
    Every other retrieved candidate is a negative (label 0), consistent with the
    synthetic single-relevant protocol used elsewhere.

    Missed-retrieval contract: if the gold paper is NOT among the retrieved
    candidates, the tuple is still emitted with an **all-zero** label vector.
    That is a non-empty ranking, so ``evaluate_rankings`` scores it as ``0.0``
    for every metric and includes it in the macro-average — the correct
    retrieve-then-rerank penalty (the pipeline genuinely failed to surface the
    answer). Only queries where the retriever returns *no* candidates at all are
    skipped (they contribute an empty ranking that ``evaluate_rankings`` drops).

    Args:
        eval_records: held-out records, each a mapping with ``id``, ``title``,
            ``abstract``. Reconstructed from ``pairs.jsonl`` upstream
            (query_id -> id, query -> title, the label-1 passage -> abstract).
        retriever: object with ``retrieve(query: str, k: int) -> list[dict]``.
        reranker: object with ``score(query: str, passages: Sequence[str])``
            returning a length-``len(passages)`` array-like of scores.
        num_candidates: candidates to retrieve per query (the eval breadth).
        with_retrieval_baseline: keyword-only. When ``False`` (default) the
            return type and behavior are unchanged (a single list). When
            ``True`` a retrieval-only baseline is emitted alongside the reranker
            queries so the two can be compared head-to-head over the IDENTICAL
            held-out queries and candidate sets.
        passage_format: ``"title_abstract"`` mirrors the original serving
            pipeline. ``"abstract"`` matches the current synthetic training
            pairs and is the recommended format for the listwise Sol run.

    Returns:
        When ``with_retrieval_baseline`` is ``False``: one ``(scores, labels)``
        tuple per query, both plain ``list[float]`` of equal length, ready to
        hand to :func:`evaluate_rankings`.

        When ``with_retrieval_baseline`` is ``True``: a 2-tuple
        ``(reranker_out, retrieval_out)`` of two such lists, aligned
        index-for-index (same query, same candidate set, same labels — only the
        score vector differs). The retrieval-only scores encode the FAISS rank
        order: candidate ``j`` (0-based) gets score ``len(candidates) - j``, a
        strictly-descending vector that reproduces the retriever's ranking under
        the stable-sort metric. Because those scores come from the SAME single
        retrieval pass (not a second ``retrieve()`` call), the baseline is
        guaranteed to share the reranker's exact candidate sets.

    Raises:
        ValueError: if a record's ``id`` looks like a bare positional index
            rather than a real paper id (its real id was lost during
            reconstruction, making label matching meaningless).
    """
    if passage_format not in {"abstract", "title_abstract"}:
        raise ValueError(
            "passage_format must be 'abstract' or 'title_abstract', "
            f"got {passage_format!r}"
        )

    out: list[tuple[list[float], list[float]]] = []
    retrieval_out: list[tuple[list[float], list[float]]] = []

    for record in eval_records:
        title = "" if record.get("title") is None else str(record["title"])
        gold_id = _normalize_id(record.get("id", ""))

        if not gold_id or _looks_like_positional_index(gold_id):
            raise ValueError(
                f"eval record id {gold_id!r} looks like a positional index, not "
                "a real paper id; label matching against retrieved paper_id "
                "would be meaningless. Ensure pairs reconstruction preserved the "
                "record's real id."
            )

        candidates = retriever.retrieve(query=title, k=num_candidates)
        if not candidates:
            # Empty ranking; evaluate_rankings skips it. Nothing to score.
            out.append(([], []))
            retrieval_out.append(([], []))
            continue

        # Use each candidate's own text. The explicit format lets one run the
        # deployed title+abstract path and the training-aligned abstract path
        # without changing candidate sets or labels.
        passages = [
            _candidate_passage(candidate, passage_format) for candidate in candidates
        ]
        scores = reranker.score(query=title, passages=passages)

        # Coerce whatever the reranker returned (torch tensor, ndarray, list)
        # into a flat list of floats without importing torch here.
        #
        # reranker.score may return a torch tensor (possibly on GPU), an ndarray,
        # or a list. Move a GPU tensor to host WITHOUT importing torch by duck-typing
        # the tensor API: .detach() drops autograd, .cpu() copies device->host. numpy
        # cannot read a cuda tensor directly (raises TypeError), so this is required
        # for the real CrossEncoderReranker on an A100; ndarrays/lists lack these
        # methods and pass straight through.
        if hasattr(scores, "detach"):
            scores = scores.detach()
        if hasattr(scores, "cpu"):
            scores = scores.cpu()
        scores_list = [float(s) for s in np.asarray(scores, dtype=np.float64).ravel()]

        labels = [
            1.0 if _normalize_id(c.get("paper_id", "")) == gold_id else 0.0
            for c in candidates
        ]

        out.append((scores_list, labels))

        # Retrieval-only baseline over the SAME candidates in the SAME order.
        # retriever.retrieve returns candidates in decreasing similarity order
        # (rank 1 first), so a strictly-descending score vector (N, N-1, ..., 1)
        # reproduces the FAISS ranking exactly under evaluate_rankings' stable
        # sort. No stored similarity scores needed — rank order is sufficient.
        retrieval_scores = [float(len(candidates) - j) for j in range(len(candidates))]
        retrieval_out.append((retrieval_scores, labels))

    if with_retrieval_baseline:
        return out, retrieval_out
    return out
