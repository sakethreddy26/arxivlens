"""Synthetic (query, passage, label) training pairs for the reranker.

Why this file exists — and its honest limitation
-------------------------------------------------
ArXivLens has **no human relevance judgments**: nobody has labeled which
abstracts are "relevant" to which queries. Training a supervised reranker
therefore requires *synthesizing* labels from the raw corpus, and this module
does exactly that. The scheme is:

- **Positives (label 1):** a paper's TITLE is treated as a query and its own
  ABSTRACT as the relevant passage. Title and abstract describe the same work,
  so they are a genuinely relevant pair — a cheap, high-precision signal.
- **Hard negatives (label 0):** the FAISS index is used to find papers whose
  embeddings are *near* the query but are **not** the true paper. These are
  topically similar yet not the right answer, which is precisely what a
  reranker must learn to push down. Hard negatives train a far sharper decision
  boundary than random ones.
- **Easy negatives (label 0):** a few *random* papers per query. Trivially
  irrelevant, they anchor the low end of the score range and stop the model
  from collapsing onto only the hard cases.

**Limitation (documented in README section 5/section 13):** these labels are a *proxy*, not
ground truth. A title/abstract pair is assumed relevant, and every non-source
paper is assumed irrelevant — but a hard negative could, in reality, be highly
relevant to the title-query yet still be labeled 0. There are no human
judgments to correct this. The approach is a reasonable bootstrap given the
data we have, and evaluation reports numbers honestly against the same
synthetic protocol rather than claiming human-level relevance.

Design
------
The core :func:`build_pairs` is **pure and I/O-free**: it takes already-parsed
records, an *injected* ``neighbor_fn``, and a seeded ``random.Random``. The
FAISS dependency is abstracted entirely behind ``neighbor_fn`` so this module
imports and tests without faiss installed — the real neighbor lookup is wired
up in ``scripts/build_pairs.py`` (lazy faiss import), while tests pass a tiny
deterministic fake. Given the same seed, output is fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Sequence

# A record is any mapping exposing at least "title" and "abstract". Extra keys
# (id, categories, ...) are ignored here.
Record = Mapping[str, object]

# neighbor_fn(query_index, k) -> candidate record indices, nearest first.
# In production this wraps the FAISS index (query the paper's own embedding and
# return the ids of its k nearest neighbours). In tests it is a small fake.
NeighborFn = Callable[[int, int], Sequence[int]]


@dataclass(frozen=True)
class Pair:
    """One training example: a query, a candidate passage, and a 0/1 label.

    ``query_id`` ties together all candidates synthesized from the SAME source
    record (the positive and its hard/easy negatives share one id), so eval can
    group them into a single ranking instead of treating each pair as its own
    1-candidate query.
    """

    query_id: str
    query: str
    passage: str
    label: int

    def as_dict(self) -> dict[str, object]:
        """Serialize to the ``{query_id, query, passage, label}`` JSONL schema."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "passage": self.passage,
            "label": self.label,
        }


def _text(record: Record, key: str) -> str:
    """Fetch a string field from a record, tolerating missing/None values."""
    value = record.get(key)
    return "" if value is None else str(value)


def build_pairs(
    records: Sequence[Record],
    neighbor_fn: NeighborFn,
    rng: random.Random,
    n_hard: int = 2,
    n_easy: int = 2,
    *,
    title_key: str = "title",
    abstract_key: str = "abstract",
    on_empty_neighbors: Callable[[int], None] | None = None,
) -> Iterator[Pair]:
    """Yield synthetic ``Pair`` examples for the whole corpus.

    For each record (indexed by its position in ``records``) this yields, in a
    stable order:

    1. the **positive** ``(title, abstract, 1)``;
    2. up to ``n_hard`` **hard negatives** — the record's nearest FAISS
       neighbours (via ``neighbor_fn``) with the true record skipped, each
       contributing ``(title, neighbour.abstract, 0)``;
    3. up to ``n_easy`` **easy negatives** — random other records, avoiding the
       true record *and* any record already used as a hard negative for this
       query, each ``(title, random.abstract, 0)``.

    The positive:negative ratio is ``1 : (n_hard + n_easy)`` — the section 5 default
    ``n_hard=2, n_easy=2`` gives 1:4. Fewer than the requested negatives may be
    produced if the corpus/neighbour list is too small to supply distinct ones;
    the code never fabricates duplicates to hit the count.

    Determinism: all randomness flows through ``rng``, so the same seeded
    ``random.Random`` yields byte-identical output. ``neighbor_fn`` is expected
    to be deterministic too (FAISS search is).

    Args:
        records: parsed corpus records; positional index == FAISS index.
        neighbor_fn: injected nearest-neighbour lookup (see module docstring).
        rng: a seeded ``random.Random`` driving easy-negative sampling.
        n_hard: hard negatives requested per positive.
        n_easy: easy negatives requested per positive.
        title_key/abstract_key: field names to read as query/passage text.
        on_empty_neighbors: optional callback invoked with the query index
            whenever ``neighbor_fn`` returns *no* usable candidates. Lets the
            caller surface a dead/empty index (which would otherwise silently
            yield all-easy-negative data) instead of hiding it.

    Yields:
        ``Pair`` instances, positive-first per record.
    """
    if n_hard < 0 or n_easy < 0:
        raise ValueError("n_hard and n_easy must be non-negative")

    n_records = len(records)
    for idx, record in enumerate(records):
        query = _text(record, title_key)
        positive_passage = _text(record, abstract_key)
        # All candidates synthesized from this source record share one query_id
        # so eval can group them into a single ranking. Prefer the record's own
        # id; fall back to its positional index as a stable string.
        query_id = _text(record, "id") or str(idx)

        # 1) Positive: this paper's title -> its own abstract.
        yield Pair(query_id=query_id, query=query, passage=positive_passage, label=1)

        used: set[int] = {idx}  # never reuse the source paper as a negative

        # 2) Hard negatives: nearest FAISS neighbours that aren't the source.
        if n_hard > 0:
            # Ask for a couple extra candidates to give some slack for skipping
            # the true paper, a duplicate-content copy of it, or a candidate we
            # can't resolve. This is best-effort, NOT a guarantee: the true
            # paper may appear, several neighbours may be unresolvable, or the
            # index may be tiny — in which case we simply yield fewer than
            # n_hard hard negatives (the loop below handles the shortfall). See
            # on_empty_neighbors for the dead/empty-index guard.
            candidates = neighbor_fn(idx, n_hard + 2)
            if not candidates and on_empty_neighbors is not None:
                on_empty_neighbors(idx)
            taken = 0
            for cand_idx in candidates:
                if taken >= n_hard:
                    break
                if cand_idx in used or not (0 <= cand_idx < n_records):
                    continue
                passage = _text(records[cand_idx], abstract_key)
                # Skip duplicate CONTENT, not just the self row: a corpus can
                # contain the same paper twice (e.g. arXiv v1/v2), and its
                # near-identical abstract would otherwise be labeled 0 in the
                # same query group as the positive.
                if passage == positive_passage:
                    continue
                used.add(cand_idx)
                yield Pair(
                    query_id=query_id,
                    query=query,
                    passage=passage,
                    label=0,
                )
                taken += 1

        # 3) Easy negatives: random other papers not already used.
        if n_easy > 0:
            pool = [i for i in range(n_records) if i not in used]
            rng.shuffle(pool)
            taken_easy = 0
            for cand_idx in pool:
                if taken_easy >= n_easy:
                    break
                passage = _text(records[cand_idx], abstract_key)
                # Same duplicate-content guard as for hard negatives.
                if passage == positive_passage:
                    continue
                used.add(cand_idx)
                yield Pair(
                    query_id=query_id,
                    query=query,
                    passage=passage,
                    label=0,
                )
                taken_easy += 1


def build_pairs_list(
    records: Sequence[Record],
    neighbor_fn: NeighborFn,
    rng: random.Random,
    n_hard: int = 2,
    n_easy: int = 2,
    **kwargs: object,
) -> list[Pair]:
    """Eager convenience wrapper around :func:`build_pairs`."""
    return list(
        build_pairs(
            records,
            neighbor_fn,
            rng,
            n_hard=n_hard,
            n_easy=n_easy,
            **kwargs,  # type: ignore[arg-type]
        )
    )


def summarize(pairs: Iterable[Pair]) -> dict[str, int]:
    """Count positives / negatives / total for the CLI summary."""
    n_pos = 0
    n_neg = 0
    for pair in pairs:
        if pair.label == 1:
            n_pos += 1
        else:
            n_neg += 1
    return {"positives": n_pos, "negatives": n_neg, "total": n_pos + n_neg}
