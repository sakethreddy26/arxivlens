"""A transformer encoder implemented from first principles in pure PyTorch.

This module is the centerpiece of ArXivLens: it is the body of the
cross-encoder reranker, written without `transformers` model classes so that
every design decision (attention scaling, pre-norm residuals, sinusoidal
positions) is explicit and inspectable.

Conventions used throughout:
- Shape comments annotate every tensor operation, e.g. ``(batch, seq_len, d_model)``.
- Attention masks are boolean and broadcastable to
  ``(batch, n_heads, q_len, k_len)``, with ``True`` = "may attend" and
  ``False`` = "masked out". `TransformerEncoder` accepts the more common
  ``(batch, seq_len)`` 0/1 padding mask and converts it internally.
- Every attention module returns its attention weights and can optionally
  cache them (``store_attention=True`` -> ``.last_attention``) so the
  visualization layer can extract them after a forward pass.

The tokenizer is deliberately NOT implemented here: we borrow a pretrained
WordPiece tokenizer (bert-base-uncased) elsewhere in the project. The model
is from scratch; tokenization is commodity plumbing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    """Hyperparameters for `TransformerEncoder`.

    Attributes:
        vocab_size: number of token ids the embedding table covers.
        d_model: width of the residual stream (embedding / hidden size).
        n_heads: number of attention heads; must divide d_model.
        n_layers: number of stacked encoder layers.
        d_ff: hidden width of the position-wise feed-forward network
            (conventionally ~4x d_model).
        max_len: longest sequence the positional encoding table supports.
        dropout: dropout probability applied to attention weights,
            feed-forward activations, and residual branches.
    """

    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_len: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V.

    Why scale by √d_k: each entry of Q Kᵀ is a sum of d_k products. For
    unit-variance inputs its variance grows linearly with d_k, so for large
    d_k the logits drift into the saturated tails of the softmax where
    gradients vanish. Dividing by √d_k keeps the logit variance ~1
    regardless of head width (Vaswani et al., 2017, section 3.2.1).

    Returns both the attended values AND the attention weights — the weights
    feed the attention-visualization "lens".
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,  # (batch, n_heads, q_len, d_head)
        key: Tensor,  # (batch, n_heads, k_len, d_head)
        value: Tensor,  # (batch, n_heads, k_len, d_head)
        mask: Tensor | None = None,  # bool, broadcastable to (batch, n_heads, q_len, k_len)
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, attention_weights).

        output: (batch, n_heads, q_len, d_head)
        attention_weights: (batch, n_heads, q_len, k_len), rows sum to 1.
        """
        d_head = query.size(-1)

        # Similarity of every query position to every key position.
        scores = query @ key.transpose(-2, -1)  # (batch, n_heads, q_len, k_len)
        scores = scores / math.sqrt(d_head)  # keep logit variance ~1 (see docstring)

        if mask is not None:
            # Use the dtype's most negative finite value rather than -inf:
            # softmax maps it to ~0 without producing NaNs if a row is
            # ever fully masked.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, q_len, k_len)

        # Dropout regularizes which positions each head relies on. It is
        # applied only to the copy used to compute the output; the returned
        # `attention_weights` are pre-dropout and always sum to 1 along dim=-1.
        output = self.dropout(attention_weights) @ value  # (batch, n_heads, q_len, d_head)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """h parallel attention heads over learned Q/K/V projections.

    A single attention head can only mix information through one similarity
    pattern per position. Splitting d_model into h subspaces of width
    d_head = d_model / h lets each head learn a different relation (e.g.
    syntactic vs. topical matching) at the same total cost as one wide head.
    Head outputs are concatenated and mixed by a final output projection.

    Set ``store_attention=True`` (or pass it per-call) to cache the per-head
    weights of the LAST STORED forward pass in ``.last_attention`` with shape
    (batch, n_heads, q_len, k_len) for later extraction/visualization. A
    non-storing pass clears the cache back to ``None`` so stale weights from a
    previous input are never left behind.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Re-validated here (not only in TransformerConfig) because MHA is a
        # standalone module that can be constructed without a TransformerConfig.
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One fused linear per role; heads are split by reshaping afterwards.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.store_attention = False
        self.last_attention: Tensor | None = None  # (batch, n_heads, q_len, k_len)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_head)."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_head)  # (batch, seq_len, n_heads, d_head)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_head)

    def forward(
        self,
        query: Tensor,  # (batch, q_len, d_model)
        key: Tensor,  # (batch, k_len, d_model)
        value: Tensor,  # (batch, k_len, d_model)
        mask: Tensor | None = None,  # bool, broadcastable to (batch, n_heads, q_len, k_len)
        store_attention: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, attention_weights).

        output: (batch, q_len, d_model)
        attention_weights: (batch, n_heads, q_len, k_len)
        """
        batch, q_len, _ = query.shape

        q = self._split_heads(self.w_q(query))  # (batch, n_heads, q_len, d_head)
        k = self._split_heads(self.w_k(key))  # (batch, n_heads, k_len, d_head)
        v = self._split_heads(self.w_v(value))  # (batch, n_heads, k_len, d_head)

        # attended: (batch, n_heads, q_len, d_head)
        # attention_weights: (batch, n_heads, q_len, k_len)
        attended, attention_weights = self.attention(q, k, v, mask=mask)

        # Per-call flag overrides the module-level default when provided.
        should_store = self.store_attention if store_attention is None else store_attention
        if should_store:
            self.last_attention = attention_weights.detach()
        else:
            self.last_attention = None  # clear stale weights from a prior input

        # Concatenate heads back into the model dimension.
        attended = attended.transpose(1, 2)  # (batch, q_len, n_heads, d_head)
        attended = attended.reshape(batch, q_len, self.d_model)  # (batch, q_len, d_model)

        output = self.w_o(attended)  # (batch, q_len, d_model)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal position embeddings added to the token embeddings.

    Self-attention is permutation-invariant: without positional information
    "cats chase dogs" and "dogs chase cats" would be indistinguishable. The
    sinusoidal formulation encodes position pos in dimension pair (2i, 2i+1) as

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    i.e. each dimension pair rotates at a different geometric frequency.
    Because sin/cos of a shifted position is a fixed linear function of the
    unshifted pair, relative offsets are easy for attention to represent —
    and there are no parameters to learn, so it generalizes to any position
    up to max_len.

    The table is a non-persistent registered buffer: it moves across devices
    with the module but is excluded from ``state_dict`` (``persistent=False``),
    since it is fully recomputable from d_model and max_len and need not be
    serialized. It is not a trainable parameter.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        # 1 / 10000^(2i/d_model) for each even dimension index 2i, computed
        # in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — broadcasts over batch

        self.register_buffer("pe", pe, persistent=False)
        self.pe: Tensor  # for type checkers; set by register_buffer

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.size(1)}"
            )
        x = x + self.pe[:, :seq_len]  # (batch, seq_len, d_model)
        return self.dropout(x)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear.

    Applied independently at every position, this is where the model does
    per-token computation on the information that attention gathered. The
    expansion to d_ff (conventionally 4x d_model) gives capacity; GELU is the
    smooth ReLU variant used by BERT/GPT.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        x = self.linear1(x)  # (batch, seq_len, d_ff)
        x = torch.nn.functional.gelu(x)  # (batch, seq_len, d_ff)
        x = self.dropout(x)
        x = self.linear2(x)  # (batch, seq_len, d_model)
        return x


class EncoderLayer(nn.Module):
    """One transformer encoder block with PRE-norm residual connections.

        x = x + Dropout(SelfAttention(LayerNorm(x)))
        x = x + Dropout(FeedForward(LayerNorm(x)))

    Why pre-norm (normalize the branch input) instead of the original
    post-norm (normalize after the residual add): with pre-norm the residual
    stream is an unbroken identity path from embeddings to output, so
    gradients reach early layers unattenuated. Post-norm places a LayerNorm
    on the main path, which makes deep stacks unstable without carefully
    tuned learning-rate warmup (Xiong et al., 2020).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,  # (batch, seq_len, d_model)
        mask: Tensor | None = None,  # bool, broadcastable to (batch, n_heads, seq_len, seq_len)
        store_attention: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, attention_weights).

        output: (batch, seq_len, d_model)
        attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        # --- Self-attention sub-layer (pre-norm residual) ---
        normed = self.attention_norm(x)  # (batch, seq_len, d_model)
        attended, attention_weights = self.attention(
            normed, normed, normed, mask=mask, store_attention=store_attention
        )
        x = x + self.dropout(attended)  # (batch, seq_len, d_model)

        # --- Feed-forward sub-layer (pre-norm residual) ---
        normed = self.ffn_norm(x)  # (batch, seq_len, d_model)
        x = x + self.dropout(self.ffn(normed))  # (batch, seq_len, d_model)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """Full encoder: embeddings + positions -> N encoder layers -> LayerNorm.

    The final LayerNorm is required with pre-norm blocks: because each block
    only normalizes its branch *inputs*, the residual stream itself is never
    normalized, so its magnitude grows with depth; one last LayerNorm gives
    downstream heads (e.g. the reranker's scoring head) a well-scaled
    representation.

    Configured via `TransformerConfig`. The forward pass accepts a standard
    HuggingFace-style 0/1 padding mask and can return every layer's per-head
    attention weights for the visualization lens.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_len, dropout=config.dropout
        )
        self.layers = nn.ModuleList(
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, dropout=config.dropout)
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _expand_padding_mask(attention_mask: Tensor) -> Tensor:
        """(batch, seq_len) 0/1 padding mask -> (batch, 1, 1, seq_len) bool.

        The singleton dims broadcast over heads and query positions: every
        query is forbidden from attending to padded KEY positions. (Padded
        queries still produce rows, but nothing downstream reads them once
        pooling respects the same mask.)
        """
        return attention_mask.bool()[:, None, None, :]  # (batch, 1, 1, seq_len)

    def forward(
        self,
        input_ids: Tensor,  # (batch, seq_len) int64 token ids
        attention_mask: Tensor | None = None,  # (batch, seq_len), 1 = real token, 0 = padding
        return_attention: bool = False,
        store_attention: bool | None = None,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Returns (hidden_states, attentions).

        hidden_states: (batch, seq_len, d_model)
        attentions: list of n_layers tensors, each
            (batch, n_heads, seq_len, seq_len), or None unless
            return_attention=True.
        """
        mask = None
        if attention_mask is not None:
            mask = self._expand_padding_mask(attention_mask)  # (batch, 1, 1, seq_len)

        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        # Scale embeddings by √d_model (Vaswani et al., 2017, section 3.4) so they
        # are not drowned out by the unit-amplitude positional encodings.
        x = x * math.sqrt(self.config.d_model)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)

        attentions: list[Tensor] | None = [] if return_attention else None
        for layer in self.layers:
            # x: (batch, seq_len, d_model)
            # attention_weights: (batch, n_heads, seq_len, seq_len)
            x, attention_weights = layer(x, mask=mask, store_attention=store_attention)
            if attentions is not None:
                attentions.append(attention_weights)

        x = self.final_norm(x)  # (batch, seq_len, d_model)
        return x, attentions
