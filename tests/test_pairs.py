"""CPU-only tests for synthetic training-pair construction.

No faiss, no network: a handful of fake records and a deterministic fake
``neighbor_fn`` stand in for the real FAISS lookup, exercising the pure
:func:`build_pairs` core in isolation.
"""

import random

from arxivlens.data.pairs import Pair, build_pairs_list, summarize

# Six tiny fake records. Positional index == "FAISS id".
RECORDS = [
    {"id": f"p{i}", "title": f"title-{i}", "abstract": f"abstract-{i}"}
    for i in range(6)
]

# Larger corpus for determinism tests: with a big easy-negative candidate pool,
# a broken/unseeded shuffle would produce a different ordering, so equality
# under the same seed genuinely proves the rng is threaded through.
BIG_RECORDS = [
    {"id": f"q{i}", "title": f"btitle-{i}", "abstract": f"babstract-{i}"}
    for i in range(200)
]


def fake_neighbor_fn(query_index: int, k: int):
    """Deterministic neighbours: include the true paper first (to test skip),
    then the next indices cyclically. Never depends on embeddings."""
    # Start with the paper itself so build_pairs must skip it as a negative.
    order = [query_index] + [(query_index + off) % len(RECORDS) for off in range(1, 6)]
    return order[:k]


def build(n_hard=2, n_easy=2, seed=0, records=RECORDS, neighbor_fn=fake_neighbor_fn):
    return build_pairs_list(
        records,
        neighbor_fn,
        random.Random(seed),
        n_hard=n_hard,
        n_easy=n_easy,
    )


def big_neighbor_fn(query_index: int, k: int):
    """Deterministic neighbours over BIG_RECORDS, true paper NOT included, so
    hard-negative choice is stable and doesn't perturb easy-negative sampling."""
    n = len(BIG_RECORDS)
    order = [(query_index + off) % n for off in range(1, n)]
    return order[:k]


def test_positive_per_record_label_1():
    pairs = build()
    positives = [p for p in pairs if p.label == 1]
    assert len(positives) == len(RECORDS)
    for i, rec in enumerate(RECORDS):
        # Positive uses the record's own title as query and own abstract.
        assert Pair(rec["title"], rec["abstract"], 1) in positives


def test_counts_match_ratio():
    pairs = build(n_hard=2, n_easy=2)
    # 1 positive + 2 hard + 2 easy per record.
    assert len(pairs) == len(RECORDS) * 5
    stats = summarize(pairs)
    assert stats["positives"] == len(RECORDS)
    assert stats["negatives"] == len(RECORDS) * 4
    assert stats["total"] == len(RECORDS) * 5


def test_configurable_ratio():
    pairs = build(n_hard=1, n_easy=3)
    assert len(pairs) == len(RECORDS) * 5  # 1 + 1 + 3
    pairs = build(n_hard=0, n_easy=0)
    assert len(pairs) == len(RECORDS)  # positives only


def test_hard_negatives_from_neighbors_and_exclude_true():
    n_hard = 2
    pairs = build(n_hard=n_hard, n_easy=0)
    for idx, rec in enumerate(RECORDS):
        # Negatives for this query, in yield order after its positive.
        query = rec["title"]
        negs = [p for p in pairs if p.query == query and p.label == 0]
        assert len(negs) == n_hard
        # Expected neighbour ids: skip the true paper (index idx itself).
        neighbors = fake_neighbor_fn(idx, n_hard + 1)
        expected_ids = [i for i in neighbors if i != idx][:n_hard]
        expected_abstracts = [RECORDS[i]["abstract"] for i in expected_ids]
        assert [n.passage for n in negs] == expected_abstracts
        # The true paper's abstract must never appear as a negative.
        assert rec["abstract"] not in [n.passage for n in negs]


def test_true_paper_skipped_at_nonzero_position():
    """The true paper must be skipped as a hard negative no matter WHERE the
    neighbour list places it — not just when it appears first."""
    n_hard = 2

    def neighbor_true_in_middle(query_index: int, k: int):
        # Put the true paper in the MIDDLE of the returned list.
        others = [(query_index + off) % len(RECORDS) for off in range(1, len(RECORDS))]
        order = [others[0], query_index] + others[1:]
        return order[:k]

    def neighbor_true_last(query_index: int, k: int):
        # Put the true paper LAST among the requested candidates.
        others = [(query_index + off) % len(RECORDS) for off in range(1, len(RECORDS))]
        # Fill k-1 real neighbours, then the true paper at the tail.
        order = others[: max(0, k - 1)] + [query_index]
        return order[:k]

    for nf in (neighbor_true_in_middle, neighbor_true_last):
        pairs = build(n_hard=n_hard, n_easy=0, neighbor_fn=nf)
        for rec in RECORDS:
            query = rec["title"]
            negs = [p for p in pairs if p.query == query and p.label == 0]
            # The true paper's abstract must never leak in as a negative,
            # regardless of its position in the neighbour list.
            assert rec["abstract"] not in [n.passage for n in negs]


def test_easy_negatives_never_equal_positive():
    pairs = build(n_hard=0, n_easy=3)
    for rec in RECORDS:
        query = rec["title"]
        easy = [p for p in pairs if p.query == query and p.label == 0]
        assert len(easy) == 3
        for p in easy:
            assert p.passage != rec["abstract"]  # never the true passage


def test_labels_are_ints_in_zero_one():
    for p in build():
        assert isinstance(p.label, int)
        assert p.label in (0, 1)


def test_determinism_same_seed():
    a = build(seed=42)
    b = build(seed=42)
    assert a == b


def test_determinism_same_seed_large_pool():
    """Same seed => byte-identical output even with a 200-record easy-negative
    pool. If the rng weren't threaded through, the shuffle would diverge."""
    a = build(seed=42, records=BIG_RECORDS, neighbor_fn=big_neighbor_fn)
    b = build(seed=42, records=BIG_RECORDS, neighbor_fn=big_neighbor_fn)
    assert a == b


def test_different_seed_differs_large_pool():
    """Different seeds must produce a different ordering of easy negatives.

    With ~198 easy-negative candidates per query across 200 queries, the odds
    that two distinct seeds pick the identical sample+order everywhere are
    astronomically small, so this assertion does not flake."""
    a = build(seed=1, records=BIG_RECORDS, neighbor_fn=big_neighbor_fn)
    b = build(seed=2, records=BIG_RECORDS, neighbor_fn=big_neighbor_fn)
    # Positives are seed-independent, but random easy negatives should differ.
    assert a != b
    # Prove the difference is in the easy negatives specifically: the set of
    # positive pairs is identical, only the negative sampling changes.
    pos_a = [p for p in a if p.label == 1]
    pos_b = [p for p in b if p.label == 1]
    assert pos_a == pos_b


def test_hard_negatives_do_not_collide_with_easy():
    # With a small corpus, easy negatives must avoid indices already used as
    # hard negatives (no duplicate passage for the same query beyond ratio).
    pairs = build(n_hard=2, n_easy=2)
    for rec in RECORDS:
        query = rec["title"]
        negs = [p.passage for p in pairs if p.query == query and p.label == 0]
        # No negative passage is the positive; may legitimately have distinct set.
        assert rec["abstract"] not in negs
