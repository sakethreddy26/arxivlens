"""Cross-encoder reranker built on the from-scratch transformer.

Bi-encoder vs. cross-encoder — why this file exists
---------------------------------------------------
Retrieval (the first stage of ArXivLens) uses a **bi-encoder**: the query and
every passage are embedded *independently* into vectors, and relevance is a
cheap dot product. That independence is what makes it scale — passage vectors
are precomputed once into a FAISS index and a query only has to be embedded and
compared against them. The cost is accuracy: the query never actually "sees"
the passage, so fine-grained interactions (a query term matching a specific
phrase in the abstract) are lost.

A **cross-encoder** — this module — does the opposite. It concatenates the
query and passage into a *single* sequence

    [CLS] query tokens [SEP] passage tokens [SEP]

and runs them jointly through the transformer, so every query token can attend
to every passage token and vice versa. That joint attention is exactly what
makes cross-encoders rank better. The price is that relevance is no longer
factorizable: you cannot precompute passage vectors, so scoring N candidates
costs N full forward passes. That is O(candidates) and cannot scale to the
whole corpus — hence the classic **retrieve-then-rerank** pipeline: the
bi-encoder cheaply narrows millions of papers to a top-k, and this
cross-encoder expensively reranks just those k.

The transformer body is from scratch (see ``transformer.py``); only the
WordPiece tokenizer is borrowed (bert-base-uncased via ``AutoTokenizer``),
because building a tokenizer is neither the point nor in scope. To keep the
model unit-testable fully offline, the tokenizer is injected via the
constructor: pass a real ``AutoTokenizer`` in production, or a tiny stub in
tests, and the reranker never touches the network itself.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn

from .attention import extract_attention, find_sep_positions, query_passage_block
from .transformer import TransformerConfig, TransformerEncoder


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal tokenizer interface the reranker depends on.

    A HuggingFace ``PreTrainedTokenizerFast`` satisfies this, and so does a
    small stub in tests. Calling it on a batch of (query, passage) text pairs
    must return a mapping with ``input_ids`` and ``attention_mask`` tensors.
    ``cls_token_id`` / ``sep_token_id`` are used to locate the special tokens
    for attention slicing and pooling.
    """

    cls_token_id: int
    sep_token_id: int

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def convert_ids_to_tokens(self, ids: Any) -> list[str]: ...


class CrossEncoderReranker(nn.Module):
    """Score (query, passage) relevance with a from-scratch cross-encoder.

    Architecture: ``TransformerEncoder`` -> take the ``[CLS]`` hidden state
    (position 0, which attended over the whole joint sequence) -> ``Linear(
    d_model, 1)`` -> a single relevance logit. Higher logit = more relevant.
    Apply a sigmoid for a probability; the raw logit is what training's BCE
    loss consumes and what ranking sorts on.

    The tokenizer is optional: ``forward`` and the low-level path work on token
    id tensors alone (handy for training and tests), while ``score`` /
    ``score_with_attention`` accept raw text and require a tokenizer to be
    present.
    """

    def __init__(
        self,
        config: TransformerConfig,
        tokenizer: TokenizerLike | None = None,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encoder = TransformerEncoder(config)
        # Pool [CLS] -> scalar relevance logit.
        self.scorer = nn.Linear(config.d_model, 1)

    # ------------------------------------------------------------------ #
    # Core forward (token-id level) — used by training and by score().    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Tensor,  # (batch, seq_len)
        attention_mask: Tensor | None = None,  # (batch, seq_len)
    ) -> Tensor:
        """Return relevance logits of shape ``(batch,)``.

        Pools the ``[CLS]`` token (index 0) of each sequence, which — because
        cross-attention let it see the entire query+passage — summarizes the
        pair, then maps it to one scalar per example.
        """
        hidden_states, _ = self.encoder(
            input_ids, attention_mask=attention_mask, return_attention=False
        )
        cls_hidden = hidden_states[:, 0]  # (batch, d_model) — the [CLS] slot
        logits = self.scorer(cls_hidden).squeeze(-1)  # (batch,)
        return logits

    # ------------------------------------------------------------------ #
    # Tokenization helpers.                                               #
    # ------------------------------------------------------------------ #
    def _require_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise RuntimeError(
                "CrossEncoderReranker was constructed without a tokenizer; "
                "pass one to use text-level score()/score_with_attention(), "
                "or call forward() with token-id tensors directly."
            )
        return self.tokenizer

    def _encode_pairs(
        self, queries: Sequence[str], passages: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        """Tokenize (query, passage) text pairs -> (input_ids, attention_mask).

        Relies on the tokenizer to lay out ``[CLS] query [SEP] passage [SEP]``
        and to pad the batch. Returns tensors on the model's device.
        """
        tokenizer = self._require_tokenizer()
        encoded = tokenizer(
            list(queries),
            list(passages),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self.scorer.weight.device
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    # ------------------------------------------------------------------ #
    # Public ranking API.                                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, query: str, passages: Sequence[str]) -> Tensor:
        """Score a list of passages against one query for ranking.

        Returns a 1-D tensor of ``len(passages)`` relevance logits; higher =
        more relevant. Sort passages by this to produce the reranking. Runs in
        eval mode with gradients disabled.
        """
        if isinstance(passages, str):
            raise TypeError("passages must be a sequence of strings, not a single string")
        was_training = self.training
        self.eval()
        try:
            input_ids, attention_mask = self._encode_pairs(
                [query] * len(passages), list(passages)
            )
            logits = self.forward(input_ids, attention_mask=attention_mask)  # (n_passages,)
        finally:
            if was_training:
                self.train()
        return logits

    @torch.no_grad()
    def score_with_attention(
        self, query: str, passage: str
    ) -> tuple[Tensor, dict[str, Any]]:
        """Score one (query, passage) pair AND return attention for the lens.

        Returns ``(score, attention_info)`` where ``score`` is a scalar tensor
        and ``attention_info`` is a dict:
            - ``weights``: ``(n_layers, n_heads, seq_len, seq_len)`` per-head
              attention over the full joint sequence.
            - ``tokens``: the token strings (for axis labels), if the tokenizer
              can produce them, else ``None``.
            - ``sep_index``: index of the first ``[SEP]`` (query/passage
              boundary), or ``None`` if not locatable.
            - ``query_passage_attention``: the query-rows x passage-cols
              sub-block (what the heatmap plots), bounded by the first ``[SEP]``
              (query/passage boundary) and the second ``[SEP]`` so the trailing
              ``[SEP]`` and padding are excluded, or ``None``.

        Extraction is delegated to ``attention.py`` so the logic is not
        duplicated between here and the notebook.
        """
        tokenizer = self._require_tokenizer()
        was_training = self.training
        self.eval()
        try:
            input_ids, attention_mask = self._encode_pairs([query], [passage])
            # Single forward pass yields BOTH the hidden states (for the [CLS]
            # logit) and the per-layer attention (for the lens) — no second
            # forward through the encoder.
            hidden_states, attentions = extract_attention(
                self.encoder, input_ids, attention_mask
            )
            cls_hidden = hidden_states[:, 0]  # (1, d_model) — the [CLS] slot
            logits = self.scorer(cls_hidden).squeeze(-1)  # (1,)
            score = logits[0]

            # Drop the batch axis: (n_layers, n_heads, seq_len, seq_len).
            weights = attentions[:, 0]
        finally:
            if was_training:
                self.train()

        ids_1d = input_ids[0]

        tokens: list[str] | None = None
        try:
            tokens = tokenizer.convert_ids_to_tokens(ids_1d.tolist())
        except Exception:  # pragma: no cover - tokenizer without this method
            tokens = None

        sep_index: int | None = None
        query_passage_attention: Tensor | None = None
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if sep_id is not None:
            sep_positions = find_sep_positions(ids_1d, sep_id)
            if sep_positions:
                sep_index = int(sep_positions[0])
                # Second [SEP] bounds the passage columns, excluding the
                # trailing [SEP] and any right-padding.
                passage_end = int(sep_positions[1]) if len(sep_positions) > 1 else None
                query_passage_attention = query_passage_block(
                    weights, sep_index, passage_end=passage_end
                )

        attention_info: dict[str, Any] = {
            "weights": weights,
            "tokens": tokens,
            "sep_index": sep_index,
            "query_passage_attention": query_passage_attention,
        }
        return score, attention_info
