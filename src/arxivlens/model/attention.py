"""Attention-extraction utilities: turn the encoder's raw attention weights
into a viz-ready form for the ArXivLens "lens".

The transformer encoder already exposes two hooks:
- ``TransformerEncoder.forward(..., return_attention=True)`` returns a list of
  per-layer weights, each shaped ``(batch, n_heads, q_len, k_len)``.
- ``MultiHeadAttention.store_attention`` / ``.last_attention`` cache the last
  forward's weights per module.

This module wraps the first hook into a single clean tensor and adds helpers
that slice the query-token block from the passage-token block given the
position of the separator (``[SEP]``) token. The reranker's
``score_with_attention`` reuses this code rather than re-deriving it, so the
extraction logic lives in exactly one place.

Shape conventions (kept unambiguous on purpose):
- ``extract_attention`` returns the BATCHED stack
  ``(n_layers, batch, n_heads, q_len, k_len)``.
- ``attention_for_pair`` drops the batch axis for a single pair, giving the
  PER-PAIR shape ``(n_layers, n_heads, q_len, k_len)``.
Downstream slicing helpers operate on any tensor whose last two dims are
``(q_len, k_len)``, so they accept either shape.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .transformer import TransformerEncoder


def extract_attention(
    encoder: TransformerEncoder,
    input_ids: Tensor,  # (batch, seq_len)
    attention_mask: Tensor | None = None,  # (batch, seq_len)
) -> tuple[Tensor, Tensor]:
    """Run a forward pass and return (hidden_states, attentions).

    hidden_states: (batch, seq_len, d_model)
    attentions: (n_layers, batch, n_heads, q_len, k_len)

    The per-layer list returned by the encoder is stacked into a single tensor
    along a new leading layer axis so downstream code can index
    ``attentions[layer]`` uniformly. Weights are detached from the graph — they
    are for inspection, not for gradient flow.
    """
    hidden_states, attentions = encoder(
        input_ids,
        attention_mask=attention_mask,
        return_attention=True,
    )
    if not attentions:
        raise RuntimeError("encoder returned no attention weights")
    # list[n_layers] of (batch, n_heads, q_len, k_len) -> stacked tensor.
    stacked = torch.stack([a.detach() for a in attentions], dim=0)
    return hidden_states, stacked  # (n_layers, batch, n_heads, q_len, k_len)


def attention_for_pair(
    encoder: TransformerEncoder,
    input_ids: Tensor,  # (1, seq_len) or (seq_len,)
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Extract per-layer attention for a SINGLE (query, passage) pair.

    Returns a tensor shaped ``(n_layers, n_heads, seq_len, seq_len)`` — the
    batch axis is dropped because a single pair is expected. This is the exact
    structure the reranker hands to the notebook for heatmapping.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, seq_len)
    if input_ids.size(0) != 1:
        raise ValueError(
            f"attention_for_pair expects a single pair (batch=1), got batch={input_ids.size(0)}"
        )
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    _, attentions = extract_attention(encoder, input_ids, attention_mask)
    return attentions[:, 0]  # (n_layers, n_heads, seq_len, seq_len)


def find_sep_positions(input_ids: Tensor, sep_token_id: int) -> list[int]:
    """Return the indices of ``sep_token_id`` in a 1-D id sequence.

    For a ``[CLS] query [SEP] passage [SEP]`` layout there are two separators;
    the first marks the query/passage boundary and the second the end. Used to
    slice the query block from the passage block for the heatmap.
    """
    if input_ids.dim() != 1:
        raise ValueError(f"expected a 1-D id sequence, got shape {tuple(input_ids.shape)}")
    return (input_ids == sep_token_id).nonzero(as_tuple=True)[0].tolist()


def query_passage_block(
    attention: Tensor,  # (..., seq_len, seq_len)
    sep_index: int,
    passage_end: int | None = None,
) -> Tensor:
    """Slice the query-token (rows) x passage-token (cols) sub-block.

    Layout assumed: ``[CLS] query [SEP] passage [SEP] (pad...)``.

    - Query ROWS are positions ``1 .. sep_index-1`` — the real query tokens,
      excluding ``[CLS]`` (row 0) and the first ``[SEP]`` (row ``sep_index``).
    - Passage COLUMNS are positions ``sep_index+1 .. passage_end-1`` — the real
      passage tokens strictly between the two ``[SEP]`` tokens, excluding BOTH
      the trailing ``[SEP]`` and any right-padding.

    ``passage_end`` is the index of the SECOND ``[SEP]`` (i.e. the first column
    past the passage). When ``None`` — no second ``[SEP]`` located — it defaults
    to ``attention.size(-1)``, running to the end of the sequence.

    Returns the block of attention FROM query tokens TO passage tokens — the
    "which passage words does the query look at" view for the lens. The leading
    dims (layers, heads, ...) are preserved untouched.
    """
    if passage_end is None:
        passage_end = attention.size(-1)
    q_slice = slice(1, sep_index)  # query tokens, excluding [CLS] and first [SEP]
    k_slice = slice(sep_index + 1, passage_end)  # passage tokens, excluding trailing [SEP]/pad
    return attention[..., q_slice, k_slice]
