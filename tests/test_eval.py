"""CPU-only tests for ranking metrics.

Hand-computable cases with expected values worked out in the comments, so a
reader can verify the formulas (log2 discount, 1-based ranks) by inspection.
All floats are asserted with a tolerance.
"""

import math

import pytest

from arxivlens.train.eval import (
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
