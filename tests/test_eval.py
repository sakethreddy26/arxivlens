"""CPU-only tests for ranking metrics.

Hand-computable cases with expected values worked out in the comments, so a
reader can verify the formulas (log2 discount, 1-based ranks) by inspection.
All floats are asserted with a tolerance.
"""

import math

import pytest

from arxivlens.train.eval import (
    build_retrieval_eval_queries,
    dcg_at_k,
    evaluate_rankings,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

TOL = 1e-9


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    # Relevant items already ranked above irrelevant -> nDCG == 1.0.
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert ndcg_at_k(scores, labels, 4) == pytest.approx(1.0, abs=TOL)
    assert ndcg_at_k(scores, labels, 2) == pytest.approx(1.0, abs=TOL)


def test_ndcg_known_misranking():
    # Predicted order puts an irrelevant item first: rank-order labels = [0, 1, 1].
    #   DCG@3  = 0/log2(2) + 1/log2(3) + 1/log2(4)
    #          = 0 + 0.6309297535714574 + 0.5 = 1.1309297535714573
    #   IDCG@3 = 1/log2(2) + 1/log2(3) + 0    = 1 + 0.6309297535714574 = 1.6309297535714573
    #   nDCG@3 = 1.1309297535714573 / 1.6309297535714573 = 0.6934264036172708
    scores = [0.9, 0.5, 0.4]  # ranks item0 (label 0) first
    labels = [0, 1, 1]
    expected = 1.1309297535714573 / 1.6309297535714573
    assert ndcg_at_k(scores, labels, 3) == pytest.approx(expected, abs=TOL)
    assert ndcg_at_k(scores, labels, 3) == pytest.approx(0.6934264036172708, abs=TOL)


def test_ndcg_zero_relevant_is_zero():
    # No relevant items -> IDCG == 0 -> defined as 0.0 (no crash).
    assert ndcg_at_k([0.3, 0.2, 0.1], [0, 0, 0], 3) == pytest.approx(0.0, abs=TOL)


def test_ndcg_graded_relevance():
    # Graded gains: rank-order = [3, 2, 1] (already ideal -> nDCG == 1.0).
    scores = [0.9, 0.5, 0.1]
    labels = [3, 2, 1]
    assert ndcg_at_k(scores, labels, 3) == pytest.approx(1.0, abs=TOL)

    # Swap so predicted order is [2, 3, 1] -> non-ideal.
    #   DCG@3  = 2/1 + 3/log2(3) + 1/log2(4) = 2 + 1.8927892607143721 + 0.5 = 4.392789...
    #   IDCG@3 = 3/1 + 2/log2(3) + 1/log2(4) = 3 + 1.2618595071429148 + 0.5 = 4.761859...
    scores2 = [0.9, 0.95, 0.1]  # item1 (gain 3) now ranks above item0 (gain 2)
    dcg = 2.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(scores2, labels, 3) == pytest.approx(dcg / idcg, abs=TOL)


def test_dcg_matches_manual():
    # rank-order labels [1, 0, 1]: DCG@3 = 1/1 + 0 + 1/log2(4) = 1.5
    assert dcg_at_k([0.9, 0.8, 0.7], [1, 0, 1], 3) == pytest.approx(1.5, abs=TOL)


# --------------------------------------------------------------------------
# MRR
# --------------------------------------------------------------------------
def test_mrr_first_relevant_at_rank_one():
    assert mrr([0.9, 0.1, 0.2], [1, 0, 0]) == pytest.approx(1.0, abs=TOL)


def test_mrr_first_relevant_at_rank_three():
    # Highest two scores are irrelevant; first relevant is at rank 3 -> 1/3.
    assert mrr([0.9, 0.8, 0.7], [0, 0, 1]) == pytest.approx(1.0 / 3.0, abs=TOL)


def test_mrr_no_relevant_is_zero():
    assert mrr([0.9, 0.8], [0, 0]) == pytest.approx(0.0, abs=TOL)


# --------------------------------------------------------------------------
# Recall@k
# --------------------------------------------------------------------------
def test_recall_hit_rate_single_relevant():
    # One relevant item at rank 2.
    scores = [0.9, 0.8, 0.1]
    labels = [0, 1, 0]
    assert recall_at_k(scores, labels, 1) == pytest.approx(0.0, abs=TOL)
    assert recall_at_k(scores, labels, 2) == pytest.approx(1.0, abs=TOL)
    assert recall_at_k(scores, labels, 5) == pytest.approx(1.0, abs=TOL)  # k > n items


def test_recall_multiple_relevant_fraction():
    # Rank-order labels = [1, 0, 1, 0]: 2 relevant total.
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 1, 0]
    assert recall_at_k(scores, labels, 1) == pytest.approx(0.5, abs=TOL)  # 1 of 2
    assert recall_at_k(scores, labels, 3) == pytest.approx(1.0, abs=TOL)  # 2 of 2


def test_recall_zero_relevant_is_zero():
    assert recall_at_k([0.9, 0.8], [0, 0], 1) == pytest.approx(0.0, abs=TOL)


def test_recall_all_relevant():
    assert recall_at_k([0.9, 0.8, 0.7], [1, 1, 1], 3) == pytest.approx(1.0, abs=TOL)
    assert recall_at_k([0.9, 0.8, 0.7], [1, 1, 1], 1) == pytest.approx(1.0 / 3.0, abs=TOL)


# --------------------------------------------------------------------------
# Tie-breaking determinism
# --------------------------------------------------------------------------
def test_tie_break_is_stable_input_order():
    # All scores equal -> stable sort keeps input order: labels stay [0, 1].
    # So the relevant item is at rank 2 -> MRR = 1/2, recall@1 = 0.
    scores = [0.5, 0.5]
    labels = [0, 1]
    assert mrr(scores, labels) == pytest.approx(0.5, abs=TOL)
    assert recall_at_k(scores, labels, 1) == pytest.approx(0.0, abs=TOL)
    # Reversing the input (relevant first) puts it at rank 1 deterministically.
    assert mrr([0.5, 0.5], [1, 0]) == pytest.approx(1.0, abs=TOL)


def test_tie_break_repeatable():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 0, 1, 0]
    first = ndcg_at_k(scores, labels, 4)
    for _ in range(5):
        assert ndcg_at_k(scores, labels, 4) == pytest.approx(first, abs=TOL)


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------
def test_empty_query_does_not_crash():
    assert ndcg_at_k([], [], 5) == pytest.approx(0.0, abs=TOL)
    assert mrr([], []) == pytest.approx(0.0, abs=TOL)
    assert recall_at_k([], [], 5) == pytest.approx(0.0, abs=TOL)
    assert dcg_at_k([], [], 5) == pytest.approx(0.0, abs=TOL)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        ndcg_at_k([0.1, 0.2], [1], 2)


def test_single_candidate():
    assert mrr([0.9], [1]) == pytest.approx(1.0, abs=TOL)
    assert ndcg_at_k([0.9], [1], 5) == pytest.approx(1.0, abs=TOL)
    assert recall_at_k([0.9], [1], 5) == pytest.approx(1.0, abs=TOL)


# --------------------------------------------------------------------------
# evaluate_rankings aggregate
# --------------------------------------------------------------------------
def test_evaluate_rankings_keys_and_perfect():
    # Two perfect queries -> every metric 1.0.
    queries = [
        ([0.9, 0.1], [1, 0]),
        ([0.8, 0.7, 0.1], [1, 0, 0]),
    ]
    out = evaluate_rankings(queries)
    assert set(out) == {"ndcg@5", "ndcg@10", "mrr", "recall@1", "recall@5", "recall@10"}
    for value in out.values():
        assert value == pytest.approx(1.0, abs=TOL)


def test_evaluate_rankings_macro_average():
    # Query A perfect (relevant at rank 1), query B relevant at rank 2.
    #   MRR = mean(1.0, 0.5) = 0.75
    #   recall@1 = mean(1.0, 0.0) = 0.5 ; recall@5 = mean(1.0, 1.0) = 1.0
    queries = [
        ([0.9, 0.1], [1, 0]),
        ([0.1, 0.9], [1, 0]),  # relevant item has the lower score -> rank 2
    ]
    out = evaluate_rankings(queries)
    assert out["mrr"] == pytest.approx(0.75, abs=TOL)
    assert out["recall@1"] == pytest.approx(0.5, abs=TOL)
    assert out["recall@5"] == pytest.approx(1.0, abs=TOL)


def test_evaluate_rankings_skips_empty_queries():
    queries = [
        ([], []),  # skipped
        ([0.9, 0.1], [1, 0]),  # perfect
    ]
    out = evaluate_rankings(queries)
    assert out["mrr"] == pytest.approx(1.0, abs=TOL)  # averaged over 1 non-empty query


def test_evaluate_rankings_all_empty_is_zero():
    out = evaluate_rankings([([], []), ([], [])])
    for value in out.values():
        assert value == pytest.approx(0.0, abs=TOL)


# --------------------------------------------------------------------------
# build_retrieval_eval_queries — retrieve-then-rerank candidate assembly
# --------------------------------------------------------------------------
class _FakeRetriever:
    """Returns a fixed candidate list per query id (keyed by the query title).

    Each candidate is a dict shaped like FaissRetriever.retrieve() output:
    ``{"paper_id", "title", "abstract"}``. Retrieval order (list order) is the
    dense-similarity order; the reranker score decides the final ranking.
    """

    def __init__(self, by_query: dict[str, list[dict]]):
        self._by_query = by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[dict]:
        self.calls.append((query, k))
        return list(self._by_query.get(query, []))[:k]


class _FakeReranker:
    """Scores passages by looking each up in a per-query score table.

    Scores are keyed by candidate title so a test can place the gold paper at
    an arbitrary rank by choosing its score. Records the (query, passages) it
    was called with so tests can assert passage construction.
    """

    def __init__(self, score_by_title: dict[str, float]):
        self._score_by_title = score_by_title
        self.seen_passages: list[list[str]] = []
        self.seen_queries: list[str] = []

    def score(self, query: str, passages):
        self.seen_queries.append(query)
        self.seen_passages.append(list(passages))
        # Passage text is "{title} {abstract}"; the candidate title is the first
        # whitespace-delimited token (titles here are single tokens). Exact-match
        # on that token so "neg1" never collides with "neg10". 0.0 if unmapped.
        out = []
        for p in passages:
            head = p.split(" ", 1)[0]
            out.append(self._score_by_title.get(head, 0.0))
        return out


def _candidate(paper_id: str, title: str, abstract: str = "abs") -> dict:
    return {"paper_id": paper_id, "title": title, "abstract": abstract}


def test_build_retrieval_eval_gold_at_rank_one():
    # 50 candidates; gold paper "P0" is retrieved and the reranker gives it the
    # top score -> recall@1/@5/@10 == 1.0, MRR == 1.0.
    cands = [_candidate("P0", "gold")] + [
        _candidate(f"N{i}", f"neg{i}") for i in range(49)
    ]
    retriever = _FakeRetriever({"gold-query": cands})
    # Gold gets the highest score; every negative gets a strictly lower one.
    scores = {"gold": 10.0, **{f"neg{i}": float(-i) for i in range(49)}}
    reranker = _FakeReranker(scores)

    records = [{"id": "P0", "title": "gold-query", "abstract": "whatever"}]
    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)

    assert len(queries) == 1
    q_scores, q_labels = queries[0]
    assert len(q_scores) == 50 and len(q_labels) == 50
    assert sum(q_labels) == 1.0  # exactly one positive
    assert q_labels[0] == 1.0  # gold is candidate 0

    out = evaluate_rankings(queries)
    assert out["recall@1"] == pytest.approx(1.0, abs=TOL)
    assert out["recall@5"] == pytest.approx(1.0, abs=TOL)
    assert out["recall@10"] == pytest.approx(1.0, abs=TOL)
    assert out["mrr"] == pytest.approx(1.0, abs=TOL)


def test_build_retrieval_eval_gold_at_rank_seven():
    # Gold retrieved but reranked to rank 7 (six negatives score higher).
    #   recall@1 = 0, recall@5 = 0, recall@10 = 1, MRR = 1/7.
    cands = [_candidate("P0", "gold")] + [
        _candidate(f"N{i}", f"neg{i}") for i in range(49)
    ]
    retriever = _FakeRetriever({"gold-query": cands})
    # Six negatives beat gold (score 5.0); gold at 4.0; rest below.
    scores = {"gold": 4.0}
    for i in range(6):
        scores[f"neg{i}"] = 5.0 + i  # strictly above gold
    for i in range(6, 49):
        scores[f"neg{i}"] = -1.0
    reranker = _FakeReranker(scores)

    records = [{"id": "P0", "title": "gold-query", "abstract": "x"}]
    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)

    out = evaluate_rankings(queries)
    assert out["recall@1"] == pytest.approx(0.0, abs=TOL)
    assert out["recall@5"] == pytest.approx(0.0, abs=TOL)
    assert out["recall@10"] == pytest.approx(1.0, abs=TOL)
    assert out["mrr"] == pytest.approx(1.0 / 7.0, abs=TOL)


def test_build_retrieval_eval_gold_missing_is_all_zero_and_counted():
    # Gold "P0" is NOT among the 50 retrieved candidates -> all-zero labels.
    # The tuple is still emitted (non-empty ranking) and scored as 0.0.
    cands = [_candidate(f"N{i}", f"neg{i}") for i in range(50)]
    retriever = _FakeRetriever({"gold-query": cands})
    reranker = _FakeReranker({f"neg{i}": float(-i) for i in range(50)})

    records = [{"id": "P0", "title": "gold-query", "abstract": "x"}]
    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)

    assert len(queries) == 1
    q_scores, q_labels = queries[0]
    assert len(q_scores) == 50  # non-empty ranking -> included in macro-average
    assert sum(q_labels) == 0.0  # all-zero labels: pipeline missed the gold

    out = evaluate_rankings(queries)
    assert out["recall@1"] == pytest.approx(0.0, abs=TOL)
    assert out["recall@10"] == pytest.approx(0.0, abs=TOL)
    assert out["mrr"] == pytest.approx(0.0, abs=TOL)


def test_build_retrieval_eval_macro_average_mixed_queries():
    # Query A: gold at rank 1 (MRR 1.0). Query B: gold missing (MRR 0.0).
    #   macro MRR = mean(1.0, 0.0) = 0.5; recall@10 = mean(1.0, 0.0) = 0.5.
    cands_a = [_candidate("A0", "agold")] + [
        _candidate(f"AN{i}", f"aneg{i}") for i in range(49)
    ]
    cands_b = [_candidate(f"BN{i}", f"bneg{i}") for i in range(50)]  # gold B0 absent
    retriever = _FakeRetriever({"a-query": cands_a, "b-query": cands_b})
    scores = {"agold": 10.0}
    scores.update({f"aneg{i}": float(-i) for i in range(49)})
    scores.update({f"bneg{i}": float(-i) for i in range(50)})
    reranker = _FakeReranker(scores)

    records = [
        {"id": "A0", "title": "a-query", "abstract": "x"},
        {"id": "B0", "title": "b-query", "abstract": "y"},
    ]
    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)
    assert len(queries) == 2

    out = evaluate_rankings(queries)
    assert out["mrr"] == pytest.approx(0.5, abs=TOL)
    assert out["recall@1"] == pytest.approx(0.5, abs=TOL)
    assert out["recall@10"] == pytest.approx(0.5, abs=TOL)


def test_build_retrieval_eval_passages_use_candidate_text_not_query():
    # Passage must be built from each candidate's own title/abstract.
    cands = [
        _candidate("P0", "gold", abstract="gold-abs"),
        _candidate("N0", "neg0", abstract="neg-abs"),
    ]
    retriever = _FakeRetriever({"gold-query": cands})
    reranker = _FakeReranker({"gold": 1.0, "neg0": 0.0})
    records = [{"id": "P0", "title": "gold-query", "abstract": "query-abs"}]

    build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)

    passages = reranker.seen_passages[0]
    assert passages == ["gold gold-abs", "neg0 neg-abs"]
    # The query's own title/abstract must NOT leak into the passages.
    assert all("gold-query" not in p and "query-abs" not in p for p in passages)
    assert reranker.seen_queries[0] == "gold-query"  # query text is the title


def test_build_retrieval_eval_normalizes_ids():
    # Retrieved paper_id has surrounding whitespace; record id is clean.
    cands = [_candidate("  P0  ", "gold")]
    retriever = _FakeRetriever({"gold-query": cands})
    reranker = _FakeReranker({"gold": 1.0})
    records = [{"id": "P0", "title": "gold-query", "abstract": "x"}]

    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)
    _scores, labels = queries[0]
    assert labels == [1.0]


def test_build_retrieval_eval_positional_index_id_raises():
    # A record id that is a bare positional index means the real id was lost.
    retriever = _FakeRetriever({"gold-query": [_candidate("P0", "gold")]})
    reranker = _FakeReranker({"gold": 1.0})
    records = [{"id": "7", "title": "gold-query", "abstract": "x"}]

    with pytest.raises(ValueError):
        build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)


def test_build_retrieval_eval_empty_retrieval_yields_empty_ranking():
    # Retriever returns nothing -> empty ranking, dropped by evaluate_rankings.
    retriever = _FakeRetriever({})  # no candidates for any query
    reranker = _FakeReranker({})
    records = [{"id": "P0", "title": "gold-query", "abstract": "x"}]

    queries = build_retrieval_eval_queries(records, retriever, reranker, num_candidates=50)
    assert queries == [([], [])]
    # A single all-empty query -> all metrics 0.0 (n_queries == 0 path).
    out = evaluate_rankings(queries)
    for value in out.values():
        assert value == pytest.approx(0.0, abs=TOL)
