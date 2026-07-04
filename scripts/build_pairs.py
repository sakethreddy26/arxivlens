"""CLI: turn ``corpus/papers.jsonl`` into synthetic reranker training pairs.

Reads the corpus, wires up a real FAISS-backed ``neighbor_fn`` from the index in
``index/`` (built already by ``embed_corpus.py`` on Sol), and writes a JSONL of
``{query, passage, label}`` for the cross-encoder to train on.

Runtime note: ``faiss`` and the index/corpus live on Sol at
``/scratch/spate472/mlrag/`` — this script is *run there*, not locally. The
``faiss`` import is deliberately **lazy** (inside :func:`build_neighbor_fn`), so
importing this module (and the unit tests for ``pairs.py``) never requires faiss
to be installed. If faiss or the index is missing, the error message points at
Sol.

Example:
    python scripts/build_pairs.py \
        --input corpus/papers.jsonl --index index \
        --output corpus/pairs.jsonl --n-hard 2 --n-easy 2 --seed 0

To also emit a held-out validation split (deterministic, seeded by --seed),
pass --val-output; the last 10%% of the shuffled pairs go there and the rest to
--output:
    python scripts/build_pairs.py \
        --input corpus/papers.jsonl --index index \
        --output corpus/pairs.jsonl --val-output corpus/val_pairs.jsonl --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

# Make ``src/`` importable when run as a plain script (no install required).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from arxivlens.data.pairs import (  # noqa: E402  (after sys.path shim)
    NeighborFn,
    Record,
    build_pairs,
)

_SOL_HINT = (
    "The corpus/index live on Sol at /scratch/spate472/mlrag/ and faiss is "
    "provided by the genai25.09 env — run this there, not locally."
)


def read_records(input_path: Path, limit: int | None = None) -> list[Record]:
    """Load ``papers.jsonl`` into a list of dict records (positional == FAISS id)."""
    if not input_path.exists():
        raise FileNotFoundError(f"corpus not found: {input_path}\n{_SOL_HINT}")
    records: list[Record] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no records read from {input_path}")
    return records


# ASSUMPTION (documented in PROGRESS.md Notes): index/meta.jsonl is one JSON
# object per line, in FAISS-row order (line i == FAISS row i), and each object
# carries the paper id under the key "id". This matches papers.jsonl and the
# {query, paper_id} /explain convention. embed_corpus.py is not
# yet written; if it emits a different key, add it to _META_ID_KEYS below.
_META_ID_KEYS = ("id", "paper_id", "arxiv_id")


def _record_id(obj: Mapping[str, object]) -> object | None:
    """Extract the paper id from a meta/corpus object, trying known key names."""
    for key in _META_ID_KEYS:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def read_meta(index_dir: Path) -> list[object]:
    """Load ``index/meta.jsonl`` -> list of paper ids in FAISS-row order.

    ``meta[i]`` is the paper id sitting at FAISS row ``i``. This is the
    authoritative FAISS-row -> paper-id mapping; we never assume the corpus
    read order matches the FAISS row order.
    """
    meta_path = index_dir / "meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"index meta not found: {meta_path}\n{_SOL_HINT}")
    ids: list[object] = []
    with meta_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = _record_id(obj)
            if pid is None:
                raise SystemExit(
                    f"meta.jsonl line {lineno + 1} has no id under any of "
                    f"{_META_ID_KEYS}: {obj!r}\n{_SOL_HINT}"
                )
            ids.append(pid)
    if not ids:
        raise SystemExit(f"no rows read from {meta_path}\n{_SOL_HINT}")
    return ids


def build_neighbor_fn(
    index_dir: Path, records: Sequence[Record]
) -> NeighborFn:
    """Build a FAISS-backed ``neighbor_fn`` — faiss is imported lazily here.

    Reconstructs each paper's stored embedding from the index and searches for
    its nearest neighbours. Raw FAISS row ids are resolved to record positions
    *explicitly* via ``index/meta.jsonl`` (FAISS-row -> paper id) crossed with a
    paper-id -> record-position map, so a divergence between FAISS order and
    ``papers.jsonl`` order can never silently mislabel a hard negative. Kept out
    of module import scope so ``pairs.py``/tests never depend on faiss.
    """
    try:
        import faiss  # noqa: F401  (lazy: only needed at real run time)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"faiss is not installed; cannot build hard negatives.\n{_SOL_HINT}"
        ) from exc

    faiss_path = index_dir / "index.faiss"
    if not faiss_path.exists():
        raise SystemExit(f"FAISS index not found: {faiss_path}\n{_SOL_HINT}")

    index = faiss.read_index(str(faiss_path))
    meta_ids = read_meta(index_dir)

    # Consistency check: meta rows must line up 1:1 with FAISS vectors.
    if len(meta_ids) != index.ntotal:
        raise SystemExit(
            f"index/meta mismatch: meta.jsonl has {len(meta_ids)} rows but "
            f"index.ntotal={index.ntotal}. The FAISS index and meta are out of "
            f"sync — rebuild the index (embed_corpus.py) at "
            f"/scratch/spate472/mlrag/.\n{_SOL_HINT}"
        )

    # paper id -> record position in the in-memory ``records`` list.
    id_to_record: dict[object, int] = {}
    for pos, rec in enumerate(records):
        pid = _record_id(rec)
        if pid is not None:
            id_to_record[pid] = pos

    # Every meta id should exist in the corpus; a large miss means stale files.
    missing = [pid for pid in meta_ids if pid not in id_to_record]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(meta_ids)} meta ids are absent from the "
            f"corpus (e.g. {missing[:3]}). The index was built from a different "
            f"papers.jsonl — rebuild the index (embed_corpus.py) at "
            f"/scratch/spate472/mlrag/.\n{_SOL_HINT}"
        )

    # FAISS row -> record position, via the paper id (explicit, not positional).
    faiss_row_to_record = [id_to_record[pid] for pid in meta_ids]

    # Record position -> FAISS row, so we can query by a record's own vector.
    record_to_faiss_row = {rec_pos: row for row, rec_pos in enumerate(faiss_row_to_record)}

    def neighbor_fn(query_index: int, k: int) -> Sequence[int]:
        # Reconstruct the query paper's own embedding from its FAISS row.
        faiss_row = record_to_faiss_row.get(int(query_index))
        if faiss_row is None:  # pragma: no cover - guarded by checks above
            return []
        embedding = index.reconstruct(faiss_row).reshape(1, -1)
        _distances, rows = index.search(embedding, k)
        # Resolve each returned FAISS row back to a record position via meta.
        # Unresolvable rows (-1 padding, or ids not in the corpus) are skipped
        # rather than passed through as bogus positional indices.
        out: list[int] = []
        for row in rows[0]:
            row = int(row)
            if row < 0 or row >= len(faiss_row_to_record):
                continue
            out.append(faiss_row_to_record[row])
        return out

    return neighbor_fn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build synthetic (query, passage, label) reranker training "
        "pairs from the ArXiv corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("corpus/papers.jsonl"),
        help="Path to papers.jsonl (default: corpus/papers.jsonl).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("index"),
        help="Directory holding index.faiss (default: index/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus/pairs.jsonl"),
        help="Output JSONL path (default: corpus/pairs.jsonl; gitignored).",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional held-out validation JSONL path. When set, pairs are "
        "shuffled deterministically (by --seed) and the last 10%% are written "
        "here as the val split with the rest going to --output. When omitted, "
        "all pairs go to --output (single-file behaviour).",
    )
    parser.add_argument(
        "--n-hard",
        type=int,
        default=2,
        help="Hard (FAISS-neighbour) negatives per positive (default: 2).",
    )
    parser.add_argument(
        "--n-easy",
        type=int,
        default=2,
        help="Easy (random) negatives per positive (default: 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility (default: 0)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N records (quick subset; default: all).",
    )
    args = parser.parse_args(argv)

    records = read_records(args.input, limit=args.limit)
    neighbor_fn = build_neighbor_fn(args.index, records)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Track queries for which the index returned no usable neighbours — a
    # broken/empty index would otherwise silently produce all-easy-negative
    # data. We collect a few example ids and warn if the fraction is large.
    empty_neighbor_ids: list[int] = []

    def on_empty_neighbors(query_index: int) -> None:
        empty_neighbor_ids.append(query_index)

    VAL_FRACTION = 0.1  # matches configs/reranker.yaml val_fraction

    # Count in-memory as we go — no redundant re-read of the output file.
    n_pos = total = 0

    generated = build_pairs(
        records,
        neighbor_fn,
        rng,
        n_hard=args.n_hard,
        n_easy=args.n_easy,
        on_empty_neighbors=on_empty_neighbors,
    )

    if args.val_output is None:
        # Single-file behaviour (unchanged): stream every pair to --output.
        with args.output.open("w", encoding="utf-8") as fh:
            for pair in generated:
                fh.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")
                total += 1
                if pair.label == 1:
                    n_pos += 1
    else:
        # Materialise all pair dicts, then split GROUP-WISE by query_id:
        # shuffle the unique query ids with a fresh random.Random(seed)
        # (reproducible, independent of however many draws the generation RNG
        # consumed) and hold out the last VAL_FRACTION of QUERIES — whole
        # candidate groups — as validation. Splitting individual pairs would
        # orphan groups across the boundary: a val query whose only positive
        # landed in train scores a forced 0.0, a positive-only singleton a
        # trivial 1.0, so the macro-averaged ranking metrics would converge
        # toward the label ratio regardless of model quality.
        pair_dicts = [pair.as_dict() for pair in generated]

        query_ids = sorted({str(d["query_id"]) for d in pair_dicts})
        random.Random(args.seed).shuffle(query_ids)

        n_val_queries = (
            max(1, int(len(query_ids) * VAL_FRACTION)) if query_ids else 0
        )
        val_query_ids = set(query_ids[len(query_ids) - n_val_queries :])

        train_dicts = [
            d for d in pair_dicts if str(d["query_id"]) not in val_query_ids
        ]
        val_dicts = [d for d in pair_dicts if str(d["query_id"]) in val_query_ids]
        n_train = len(train_dicts)
        n_val = len(val_dicts)

        args.val_output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for d in train_dicts:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        with args.val_output.open("w", encoding="utf-8") as fh:
            for d in val_dicts:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

        total = len(pair_dicts)
        n_pos = sum(1 for d in pair_dicts if d.get("label") == 1)
        print(
            f"[split] {n_train} train -> {args.output} | "
            f"{n_val} val ({n_val_queries} query groups) -> {args.val_output} "
            f"(val_fraction={VAL_FRACTION} of queries, seed={args.seed})"
        )

    n_neg = total - n_pos

    if args.n_hard > 0 and empty_neighbor_ids:
        n_empty = len(empty_neighbor_ids)
        frac = n_empty / len(records)
        msg = (
            f"[warn] {n_empty}/{len(records)} queries ({frac:.0%}) got NO hard "
            f"negatives from FAISS (e.g. records {empty_neighbor_ids[:3]}); "
            f"their negatives are all easy/random."
        )
        if frac >= 0.5:
            msg += (
                " Over half the corpus resolved no neighbours — the index looks "
                "dead or out of sync. Rebuild it (embed_corpus.py) at "
                "/scratch/spate472/mlrag/."
            )
        print(msg, file=sys.stderr)
    # Hard vs easy split is deterministic from the config given full negatives.
    print(
        f"Wrote {total} pairs to {args.output}\n"
        f"  positives : {n_pos}\n"
        f"  negatives : {n_neg} "
        f"(~{args.n_hard} hard + ~{args.n_easy} easy per positive)\n"
        f"  ratio     : 1:{args.n_hard + args.n_easy}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
