"""Tests for the Sol pair-artifact preflight validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_pair_groups import validate_corpus_coverage, validate_file


def _write_groups(path: Path, query_ids: list[str], size: int = 3) -> None:
    rows = []
    for query_id in query_ids:
        for index in range(size):
            rows.append(
                {
                    "query_id": query_id,
                    "query": f"query {query_id}",
                    "passage": f"passage {query_id} {index}",
                    "label": int(index == 0),
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validate_file_accepts_complete_groups(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_groups(path, ["q0", "q1"])

    assert validate_file(path, candidates_per_query=3) == {"q0", "q1"}


def test_validate_file_rejects_incomplete_group(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_groups(path, ["q0"], size=2)

    with pytest.raises(ValueError, match="do not have 3 candidates"):
        validate_file(path, candidates_per_query=3)


def test_validate_file_rejects_multiple_positives(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"query_id": "q0", "query": "q", "passage": "a", "label": 1},
        {"query_id": "q0", "query": "q", "passage": "b", "label": 1},
        {"query_id": "q0", "query": "q", "passage": "c", "label": 0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one positive"):
        validate_file(path, candidates_per_query=3)


def test_validate_corpus_coverage_rejects_truncated_split(tmp_path: Path) -> None:
    corpus = tmp_path / "papers.jsonl"
    corpus.write_text(
        "".join(json.dumps({"id": f"q{i}"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match corpus"):
        validate_corpus_coverage({"q0", "q1"}, {"q9"}, corpus)


def test_validate_corpus_coverage_accepts_complete_ten_percent_split(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "papers.jsonl"
    corpus.write_text(
        "".join(json.dumps({"id": f"q{i}"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )

    validate_corpus_coverage({f"q{i}" for i in range(9)}, {"q9"}, corpus)
