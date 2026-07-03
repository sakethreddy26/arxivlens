"""PyTorch Dataset and collation utilities for cross-encoder training.

Purpose
-------
The training loop consumes (query, passage, label) triplets from
``pairs.jsonl`` (built by :mod:`arxivlens.data.pairs`, section 5). This module
bridges that file to a :class:`torch.utils.data.DataLoader` by:

1. **Parsing** each JSON line into a Python dict on demand (lazy — not all at
   construction time, so large files don't stall start-up or blow memory).
2. **Tokenizing** the pair as ``[CLS] query [SEP] passage [SEP]`` through an
   *injected* tokenizer, keeping the module network-free: pass a real
   HuggingFace ``AutoTokenizer`` in production, or a tiny offline stub in
   tests.
3. **Collating** a list of variable-length examples into padded batch tensors
   ready for :class:`~arxivlens.model.reranker.CrossEncoderReranker`.

Input contract (pairs.jsonl schema)
------------------------------------
Each line is a JSON object with exactly three fields::

    {"query": "<string>", "passage": "<string>", "label": 0 | 1}

``label`` must be an integer in ``{0, 1}`` — the BCE loss head in the trainer
requires float conversion, which :class:`PairDataset` performs.

Tokenizer protocol
-------------------
The tokenizer is any object satisfying
:class:`~arxivlens.model.reranker.TokenizerLike`::

    cls_token_id: int
    sep_token_id: int
    __call__(*args, **kwargs) -> mapping with "input_ids" / "attention_mask"
    convert_ids_to_tokens(ids) -> list[str]

A HuggingFace ``PreTrainedTokenizerFast`` satisfies this.  The dataset does
NOT import ``transformers`` and does NOT hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """PyTorch Dataset for (query, passage, label) cross-encoder training pairs.

    Reads a JSONL file where each line is ``{"query": str, "passage": str,
    "label": int}``.  Tokenizes each pair as ``[CLS] query [SEP] passage
    [SEP]`` using an injected tokenizer.  The tokenizer is injected (not
    imported internally) so tests can use a stub offline.

    Args:
        path: Path to the pairs.jsonl file produced by
            :mod:`arxivlens.data.pairs`.
        tokenizer: Any object satisfying the ``TokenizerLike`` protocol from
            :mod:`arxivlens.model.reranker` — has ``__call__`` returning a
            dict with ``input_ids`` / ``attention_mask``, and
            ``cls_token_id`` / ``sep_token_id`` attributes.
        max_length: Maximum token sequence length.  Sequences longer than this
            are truncated symmetrically (query first) by the tokenizer.

    ``__getitem__`` returns a dict::

        {
            "input_ids":      LongTensor of shape (seq_len,),
            "attention_mask": LongTensor of shape (seq_len,),
            "label":          FloatTensor scalar (0.0 or 1.0),
            "query_id":       str (groups candidates for eval; falls back to the
                              positional index when the record lacks the key),
        }

    Tokenization is **lazy** — it happens at ``__getitem__`` time, not at
    construction time, so constructing the dataset is O(1) in memory even for
    a multi-million-line file.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self._path = Path(path)
        self._tokenizer = tokenizer
        self._max_length = max_length

        # Read all lines eagerly for random-access indexing.  Each line is kept
        # as a raw string; JSON parsing is deferred to __getitem__ so the
        # constructor stays cheap enough to be called in a DataLoader worker
        # initializer without a noticeable latency spike.
        with self._path.open(encoding="utf-8") as fh:
            self._lines: list[str] = [line for line in fh if line.strip()]

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Number of (query, passage, label) examples in the file."""
        return len(self._lines)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Tokenize the pair at ``idx`` and return tensors.

        Args:
            idx: Integer index in ``[0, len(self))``.

        Returns:
            Dict with keys:

            * ``"input_ids"`` — ``LongTensor`` of shape ``(seq_len,)``.
            * ``"attention_mask"`` — ``LongTensor`` of shape ``(seq_len,)``.
            * ``"label"`` — ``FloatTensor`` scalar (0.0 or 1.0).
            * ``"query_id"`` — ``str`` grouping key (defaults to ``str(idx)``
              when the record has no ``query_id`` field).
        """
        record: dict[str, Any] = json.loads(self._lines[idx])
        query: str = record["query"]
        passage: str = record["passage"]
        label: int = int(record["label"])
        # query_id groups candidates for a single ranking during eval. Older
        # pairs files predate this field, so fall back to the positional index.
        query_id: str = str(record.get("query_id", idx))

        # The tokenizer is called with the query and passage as a pair so it
        # can lay out [CLS] query [SEP] passage [SEP] and handle truncation.
        encoded = self._tokenizer(
            query,
            passage,
            max_length=self._max_length,
            truncation=True,
            padding=False,       # padding is handled batch-wise in collate_fn
            return_tensors="pt", # (1, seq_len) tensors
        )

        # Remove the batch dimension added by return_tensors="pt".
        input_ids: Tensor = encoded["input_ids"].squeeze(0).long()       # (seq_len,)
        attention_mask: Tensor = encoded["attention_mask"].squeeze(0).long()  # (seq_len,)
        label_tensor: Tensor = torch.tensor(float(label), dtype=torch.float32)  # scalar

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label_tensor,
            "query_id": query_id,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of ``PairDataset`` items into padded batch tensors.

    Sequences within a batch are padded on the **right** with zeros to match
    the longest sequence, which is the standard for encoder-only models with
    a padding mask.  Labels are stacked into a 1-D float tensor.

    Args:
        batch: List of dicts, each with ``"input_ids"``, ``"attention_mask"``,
            ``"label"``, and ``"query_id"`` as returned by
            :meth:`PairDataset.__getitem__`.

    Returns:
        Dict with keys:

        * ``"input_ids"``      — ``LongTensor`` of shape ``(B, max_len)``.
        * ``"attention_mask"`` — ``LongTensor`` of shape ``(B, max_len)``.
        * ``"labels"``         — ``FloatTensor`` of shape ``(B,)``.
          Note the plural key name: the trainer reads ``batch["labels"]``
          to match HuggingFace Trainer conventions and avoid shadowing the
          singular ``"label"`` key returned per item.
        * ``"query_ids"``      — ``list[str]`` of length ``B`` (NOT a tensor).
          The eval loop groups scores/labels by this key so all candidates of
          one query form a single ranking.
    """
    max_len: int = max(item["input_ids"].size(0) for item in batch)
    batch_size: int = len(batch)

    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        padded_ids[i, :seq_len] = item["input_ids"]
        padded_mask[i, :seq_len] = item["attention_mask"]

    labels: Tensor = torch.stack([item["label"] for item in batch])  # (B,)
    query_ids: list[str] = [item["query_id"] for item in batch]

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "labels": labels,
        "query_ids": query_ids,
    }
