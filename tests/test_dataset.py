"""CPU-only tests for PairDataset and collate_fn.

All tests use a stub tokenizer — no HuggingFace downloads, no network calls.
The stub satisfies the ``TokenizerLike`` protocol from
:mod:`arxivlens.model.reranker` by implementing ``cls_token_id``,
``sep_token_id``, ``__call__``, and ``convert_ids_to_tokens``.

Test data is written to ``tmp_path`` (pytest fixture) — no fixture files on
disk — so the suite is self-contained and leaves nothing behind.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from arxivlens.data.dataset import PairDataset, collate_fn


# --------------------------------------------------------------------------- #
# Stub tokenizer                                                               #
# --------------------------------------------------------------------------- #

class StubTokenizer:
    """Deterministic offline tokenizer that satisfies the TokenizerLike protocol.

    Tokenizes text by hashing each whitespace-split word modulo ``vocab_size``
    and prepending/appending the special token ids for CLS and SEP.  Always
    returns ``return_tensors="pt"`` style dicts (shape ``(1, seq_len)``) so the
    interface matches HuggingFace's tokenizer output that ``PairDataset``
    expects.

    The vocabulary is tiny (32 tokens) to keep tests fast.  The mapping is
    deterministic md5-based: the same word always maps to the same id, and —
    unlike Python's built-in ``hash`` (salted per process via PYTHONHASHSEED) —
    the ids are stable across processes and runs, which the resume smoke test
    relies on.
    """

    cls_token_id: int = 0
    sep_token_id: int = 1
    _vocab_size: int = 32

    def _word_to_id(self, word: str) -> int:
        """Map a word string to a stable integer id in [3, vocab_size).

        Reserve ids 0, 1, 2 (CLS=0, SEP=1, slot 2 unused) so the ``- 3`` offset
        below guarantees no real word collides with a special id. md5 gives a
        process-stable hash (built-in ``hash`` is salted per process).
        """
        return 3 + (int(hashlib.md5(word.encode()).hexdigest(), 16) % (self._vocab_size - 3))

    def __call__(
        self,
        text_a: str,
        text_b: str | None = None,
        max_length: int = 256,
        truncation: bool = True,
        padding: bool | str = False,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Tokenize ``text_a`` (and optionally ``text_b``) as a pair.

        Output layout: ``[CLS] text_a_tokens [SEP] text_b_tokens [SEP]`` when
        ``text_b`` is provided, or ``[CLS] text_a_tokens [SEP]`` otherwise.
        Truncates to ``max_length`` tokens from the right.
        """
        ids: list[int] = [self.cls_token_id]
        ids += [self._word_to_id(w) for w in text_a.split()]
        ids.append(self.sep_token_id)
        if text_b is not None:
            ids += [self._word_to_id(w) for w in text_b.split()]
            ids.append(self.sep_token_id)

        # Right-truncate to max_length.
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        mask = [1] * len(ids)

        ids_t = torch.tensor([ids], dtype=torch.long)    # (1, seq_len)
        mask_t = torch.tensor([mask], dtype=torch.long)  # (1, seq_len)

        return {"input_ids": ids_t, "attention_mask": mask_t}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Map integer ids back to placeholder token strings."""
        special = {self.cls_token_id: "[CLS]", self.sep_token_id: "[SEP]"}
        return [special.get(i, f"tok_{i}") for i in ids]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tokenizer() -> StubTokenizer:
    """Shared stub tokenizer instance."""
    return StubTokenizer()


@pytest.fixture()
def pairs_file(tmp_path: Path) -> Path:
    """A temporary pairs.jsonl file with 5 examples (3 positive, 2 negative)."""
    records = [
        {"query_id": "q0", "query": "attention is all you need", "passage": "transformer architecture", "label": 1},
        {"query_id": "q0", "query": "bert language model",        "passage": "bidirectional encoder",   "label": 1},
        {"query_id": "q1", "query": "neural network depth",       "passage": "deep learning residuals", "label": 0},
        {"query_id": "q2", "query": "graph neural networks",      "passage": "node embeddings pooling", "label": 0},
        {"query_id": "q0", "query": "contrastive self supervised", "passage": "SimCLR representation",  "label": 1},
    ]
    path = tmp_path / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture()
def dataset(pairs_file: Path, tokenizer: StubTokenizer) -> PairDataset:
    """PairDataset wrapping the 5-example fixture file."""
    return PairDataset(pairs_file, tokenizer, max_length=64)


# --------------------------------------------------------------------------- #
# 1. Length                                                                    #
# --------------------------------------------------------------------------- #

def test_dataset_len(dataset: PairDataset) -> None:
    """Dataset length equals the number of lines in the JSONL file."""
    assert len(dataset) == 5


# --------------------------------------------------------------------------- #
# 2. __getitem__ key set                                                       #
# --------------------------------------------------------------------------- #

def test_getitem_keys(dataset: PairDataset) -> None:
    """__getitem__ returns a dict with the expected keys (incl. query_id)."""
    item = dataset[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "label", "query_id"}


def test_collate_carries_query_ids(dataset: PairDataset) -> None:
    """collate_fn returns a length-B list of str query_ids (not a tensor)."""
    items = [dataset[i] for i in range(3)]
    batch = collate_fn(items)
    assert isinstance(batch["query_ids"], list)
    assert len(batch["query_ids"]) == 3
    assert all(isinstance(q, str) for q in batch["query_ids"])
    # Fixture records 0..2 carry query_ids q0, q0, q1.
    assert batch["query_ids"] == ["q0", "q0", "q1"]


def test_getitem_query_id_defaults_when_absent(
    tmp_path: Path, tokenizer: StubTokenizer
) -> None:
    """A record without a query_id field falls back to str(idx)."""
    path = tmp_path / "no_qid.jsonl"
    path.write_text(
        json.dumps({"query": "a b", "passage": "c d", "label": 1}) + "\n"
        + json.dumps({"query": "e f", "passage": "g h", "label": 0}) + "\n",
        encoding="utf-8",
    )
    ds = PairDataset(path, tokenizer, max_length=64)
    assert ds[0]["query_id"] == "0"
    assert ds[1]["query_id"] == "1"


def test_word_id_is_process_stable() -> None:
    """A known word maps to a hardcoded id — proves the md5 hash is stable.

    Built-in ``hash`` is salted per process (PYTHONHASHSEED), so this assertion
    would flake with the old implementation; md5 makes it reproducible.
    """
    tok = StubTokenizer()
    assert tok._word_to_id("attention") == 30


# --------------------------------------------------------------------------- #
# 3. Tensor shapes                                                             #
# --------------------------------------------------------------------------- #

def test_getitem_shapes(dataset: PairDataset) -> None:
    """input_ids and attention_mask are 1-D; label is a scalar (0-D)."""
    item = dataset[0]
    assert item["input_ids"].dim() == 1, "input_ids must be 1-D"
    assert item["attention_mask"].dim() == 1, "attention_mask must be 1-D"
    assert item["label"].dim() == 0, "label must be a scalar (0-D) tensor"


# --------------------------------------------------------------------------- #
# 4. Label dtype                                                               #
# --------------------------------------------------------------------------- #

def test_label_dtype(dataset: PairDataset) -> None:
    """label must be float32 (required by BCE loss)."""
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["label"].dtype == torch.float32, (
            f"item {i}: expected float32 label, got {item['label'].dtype}"
        )


# --------------------------------------------------------------------------- #
# 5. collate_fn pads to same length                                            #
# --------------------------------------------------------------------------- #

def test_collate_pads_to_same_length(dataset: PairDataset) -> None:
    """All sequences in a collated batch share the same length (max of the batch)."""
    # Take first 3 items, which will have different raw sequence lengths
    # because queries and passages differ.
    items = [dataset[i] for i in range(3)]
    batch = collate_fn(items)

    lens = [item["input_ids"].size(0) for item in items]
    expected_max = max(lens)

    assert batch["input_ids"].size(1) == expected_max
    assert batch["attention_mask"].size(1) == expected_max


# --------------------------------------------------------------------------- #
# 6. attention_mask shape matches input_ids                                    #
# --------------------------------------------------------------------------- #

def test_collate_attention_mask_shape(dataset: PairDataset) -> None:
    """Collated attention_mask and input_ids have exactly the same shape."""
    items = [dataset[i] for i in range(4)]
    batch = collate_fn(items)
    assert batch["attention_mask"].shape == batch["input_ids"].shape


# --------------------------------------------------------------------------- #
# 7. labels shape is (batch_size,)                                             #
# --------------------------------------------------------------------------- #

def test_collate_labels_shape(dataset: PairDataset) -> None:
    """Collated labels tensor has shape (batch_size,)."""
    batch_size = 5
    items = [dataset[i] for i in range(batch_size)]
    batch = collate_fn(items)
    assert batch["labels"].shape == (batch_size,)


# --------------------------------------------------------------------------- #
# 8. DataLoader iteration                                                      #
# --------------------------------------------------------------------------- #

def test_dataloader_iterates(dataset: PairDataset) -> None:
    """A DataLoader wrapping PairDataset completes a full pass without error."""
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
    batches = list(loader)  # consume all; raises on any error

    # With 5 examples and batch_size=2 we expect 3 batches (2, 2, 1).
    assert len(batches) == 3

    for batch in batches:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch


# --------------------------------------------------------------------------- #
# 9. Positive label round-trip                                                 #
# --------------------------------------------------------------------------- #

def test_positive_label(pairs_file: Path, tokenizer: StubTokenizer) -> None:
    """A pair with label=1 in the JSONL file comes back as label tensor == 1.0."""
    # Write a single positive example.
    path = pairs_file.parent / "positive.jsonl"
    path.write_text(
        json.dumps({"query": "deep learning", "passage": "neural networks", "label": 1}) + "\n",
        encoding="utf-8",
    )
    ds = PairDataset(path, tokenizer)
    item = ds[0]
    assert item["label"].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 10. Negative label round-trip                                                #
# --------------------------------------------------------------------------- #

def test_negative_label(pairs_file: Path, tokenizer: StubTokenizer) -> None:
    """A pair with label=0 in the JSONL file comes back as label tensor == 0.0."""
    path = pairs_file.parent / "negative.jsonl"
    path.write_text(
        json.dumps({"query": "random query", "passage": "unrelated passage", "label": 0}) + "\n",
        encoding="utf-8",
    )
    ds = PairDataset(path, tokenizer)
    item = ds[0]
    assert item["label"].item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bonus: input_ids and attention_mask are LongTensors                         #
# --------------------------------------------------------------------------- #

def test_input_ids_dtype(dataset: PairDataset) -> None:
    """input_ids and attention_mask must be long (int64) — required by nn.Embedding."""
    item = dataset[0]
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long


# --------------------------------------------------------------------------- #
# Bonus: padding positions have attention_mask == 0                            #
# --------------------------------------------------------------------------- #

def test_collate_padding_mask_is_zero(dataset: PairDataset) -> None:
    """Positions that are padding (beyond the original sequence) have mask == 0."""
    items = [dataset[i] for i in range(3)]
    raw_lens = [item["input_ids"].size(0) for item in items]
    batch = collate_fn(items)
    max_len = batch["input_ids"].size(1)

    for i, orig_len in enumerate(raw_lens):
        if orig_len < max_len:
            # All positions after orig_len must be masked out.
            pad_mask = batch["attention_mask"][i, orig_len:]
            assert pad_mask.sum().item() == 0, (
                f"Row {i}: expected zeros in mask beyond position {orig_len}"
            )
            # And the padding ids themselves must be 0.
            pad_ids = batch["input_ids"][i, orig_len:]
            assert pad_ids.sum().item() == 0, (
                f"Row {i}: expected zero ids in padding beyond position {orig_len}"
            )
