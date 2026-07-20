"""Validate listwise train/validation JSONL artifacts without ML dependencies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroupStats:
    count: int = 0
    positives: int = 0


def validate_file(path: Path, candidates_per_query: int) -> set[str]:
    """Validate schema and complete query groups; return the query ids."""
    if not path.is_file():
        raise ValueError(f"file not found: {path}")

    groups: dict[str, GroupStats] = {}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            missing = {"query_id", "query", "passage", "label"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            query_id = str(row["query_id"]).strip()
            if not query_id:
                raise ValueError(f"{path}:{line_number}: empty query_id")
            if not isinstance(row["query"], str) or not isinstance(row["passage"], str):
                raise ValueError(f"{path}:{line_number}: query and passage must be strings")
            if row["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")

            stats = groups.setdefault(query_id, GroupStats())
            stats.count += 1
            stats.positives += int(row["label"])
            rows += 1

    if not groups:
        raise ValueError(f"{path}: no query groups")

    bad_sizes = [qid for qid, stats in groups.items() if stats.count != candidates_per_query]
    if bad_sizes:
        example = bad_sizes[0]
        raise ValueError(
            f"{path}: {len(bad_sizes)} groups do not have {candidates_per_query} "
            f"candidates; {example!r} has {groups[example].count}"
        )
    bad_positives = [qid for qid, stats in groups.items() if stats.positives != 1]
    if bad_positives:
        example = bad_positives[0]
        raise ValueError(
            f"{path}: {len(bad_positives)} groups do not have exactly one "
            f"positive; {example!r} has {groups[example].positives}"
        )

    print(f"[pairs] valid: {path} ({rows} rows, {len(groups)} query groups)")
    return set(groups)


def validate_corpus_coverage(train_ids: set[str], val_ids: set[str], corpus_path: Path) -> None:
    """Require the split to cover every corpus record exactly once."""
    if not corpus_path.is_file():
        raise ValueError(f"corpus file not found: {corpus_path}")

    corpus_ids: list[str] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(line for line in handle if line.strip()):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{corpus_path}: invalid JSON record {index + 1}: {exc}") from exc
            value = record.get("id")
            query_id = "" if value is None else str(value)
            corpus_ids.append(query_id or str(index))

    expected = set(corpus_ids)
    if len(expected) != len(corpus_ids):
        raise ValueError(f"{corpus_path}: duplicate query ids in corpus")
    actual = train_ids | val_ids
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            "pair split does not match corpus: " f"missing={len(missing)}, extra={len(extra)}"
        )

    expected_val = max(1, int(len(expected) * 0.1))
    if len(val_ids) != expected_val:
        raise ValueError(f"validation split has {len(val_ids)} queries; expected {expected_val}")
    print(f"[pairs] corpus coverage valid: all {len(expected)} queries represented")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates-per-query", type=int, default=20)
    args = parser.parse_args()

    if args.candidates_per_query < 2:
        parser.error("--candidates-per-query must be at least 2")

    train_ids = validate_file(args.train, args.candidates_per_query)
    val_ids = validate_file(args.val, args.candidates_per_query)
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"train/val query leakage: {len(overlap)} overlapping ids; "
            f"example={next(iter(overlap))!r}"
        )
    validate_corpus_coverage(train_ids, val_ids, args.corpus)
    print(
        f"[pairs] train/val split valid: {len(train_ids)} train queries, "
        f"{len(val_ids)} validation queries, no overlap"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
