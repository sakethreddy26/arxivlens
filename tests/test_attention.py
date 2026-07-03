"""CPU-friendly tests for attention extraction and the cross-encoder reranker.

Everything runs offline: a tiny stub tokenizer stands in for the borrowed
bert-base-uncased WordPiece tokenizer, so no network access or model download
is needed. Configs are tiny (d_model=32, n_heads=4, n_layers=2, small vocab)
so the file runs in seconds.
"""

import torch

from arxivlens.model.attention import (
    attention_for_pair,
    extract_attention,
    find_sep_positions,
    query_passage_block,
)
from arxivlens.model.reranker import CrossEncoderReranker
from arxivlens.model.transformer import TransformerConfig, TransformerEncoder

# --- tiny config, per section 11 --------------------------------------------------
VOCAB_SIZE = 20
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
MAX_LEN = 32

# Reserved special-token ids for the stub tokenizer.
CLS_ID = 1
SEP_ID = 2
PAD_ID = 0


def make_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=0.0,
    )


class StubTokenizer:
    """A minimal offline tokenizer that lays out [CLS] q [SEP] p [SEP].

    "Tokenization" is just hashing each whitespace word into the content-id
    range ``[3, VOCAB_SIZE)``. It supports the exact call signature the
    reranker uses (batched text pairs -> padded tensors) plus
    ``convert_ids_to_tokens`` for axis labels — no network, no vocab files.
    """

    cls_token_id = CLS_ID
    sep_token_id = SEP_ID
    pad_token_id = PAD_ID

    def _word_id(self, word: str) -> int:
        return 3 + (hash(word) % (VOCAB_SIZE - 3))

    def _encode_one(self, query: str, passage: str) -> list[int]:
        q = [self._word_id(w) for w in query.split()]
        p = [self._word_id(w) for w in passage.split()]
        return [CLS_ID, *q, SEP_ID, *p, SEP_ID]

    def __call__(
        self,
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ):
        rows = [self._encode_one(q, p) for q, p in zip(queries, passages)]
        if truncation:
            rows = [r[:max_length] for r in rows]
        width = max(len(r) for r in rows)
        input_ids, attention_mask = [], []
        for r in rows:
            pad = width - len(r)
            input_ids.append(r + [PAD_ID] * pad)
            attention_mask.append([1] * len(r) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        names = {CLS_ID: "[CLS]", SEP_ID: "[SEP]", PAD_ID: "[PAD]"}
        return [names.get(i, f"tok{i}") for i in ids]


# --- attention extraction --------------------------------------------------
def test_extract_attention_shapes():
    encoder = TransformerEncoder(make_config())
    batch, seq_len = 2, 7
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))

    _, attentions = extract_attention(encoder, input_ids)
    # (n_layers, batch, n_heads, q_len, k_len)
    assert attentions.shape == (N_LAYERS, batch, N_HEADS, seq_len, seq_len)


def test_attention_for_pair_per_layer_shape():
    encoder = TransformerEncoder(make_config())
    seq_len = 6
    input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))

    weights = attention_for_pair(encoder, input_ids)
    # [n_layers] of (n_heads, q_len, k_len)
    assert weights.shape == (N_LAYERS, N_HEADS, seq_len, seq_len)
    for layer in weights:
        assert layer.shape == (N_HEADS, seq_len, seq_len)


def test_attention_weights_are_distributions():
    encoder = TransformerEncoder(make_config())
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))

    weights = attention_for_pair(encoder, input_ids)
    assert torch.all(weights >= 0)  # non-negative
    key_sums = weights.sum(dim=-1)  # sum over key axis
    assert torch.allclose(key_sums, torch.ones_like(key_sums), atol=1e-5)


def test_sep_slicing_helpers():
    ids = torch.tensor([CLS_ID, 5, 6, SEP_ID, 7, 8, 9, SEP_ID])
    seps = find_sep_positions(ids, SEP_ID)
    assert seps == [3, 7]

    # attention shaped (n_layers, n_heads, seq, seq); slice query x passage.
    attn = torch.rand(N_LAYERS, N_HEADS, ids.numel(), ids.numel())
    block = query_passage_block(attn, sep_index=3)
    # query rows: positions 1,2 (2 rows); passage cols: positions 4..7 (4 cols)
    assert block.shape == (N_LAYERS, N_HEADS, 2, 4)


# --- reranker: scoring -----------------------------------------------------
def test_score_returns_one_scalar_per_passage():
    model = CrossEncoderReranker(make_config(), tokenizer=StubTokenizer())
    passages = ["deep learning for vision", "graph neural networks", "random text here"]

    scores = model.score("attention transformers", passages)
    assert scores.shape == (len(passages),)
    assert torch.all(torch.isfinite(scores))


def test_forward_logits_shape_and_finite():
    model = CrossEncoderReranker(make_config())
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 9))
    attention_mask = torch.ones(4, 9, dtype=torch.long)

    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4,)
    assert torch.all(torch.isfinite(logits))


# --- reranker: score_with_attention ---------------------------------------
def test_score_with_attention_structure():
    model = CrossEncoderReranker(make_config(), tokenizer=StubTokenizer())

    score, info = model.score_with_attention(
        "attention transformers", "deep learning for vision"
    )
    assert score.shape == ()  # scalar
    assert torch.isfinite(score)

    weights = info["weights"]
    seq_len = weights.size(-1)
    assert weights.shape == (N_LAYERS, N_HEADS, seq_len, seq_len)

    # consistent with layers/heads and still a valid distribution
    key_sums = weights.sum(dim=-1)
    assert torch.allclose(key_sums, torch.ones_like(key_sums), atol=1e-5)

    assert info["tokens"] is not None
    assert len(info["tokens"]) == seq_len
    assert info["sep_index"] is not None
    # query x passage block has correct leading dims
    qp = info["query_passage_attention"]
    assert qp.shape[:2] == (N_LAYERS, N_HEADS)


# --- reranker: masking + [CLS] pooling ------------------------------------
def test_masked_positions_do_not_affect_logit_but_real_tokens_do():
    """Pins two invariants at once:

    - [CLS] pooling at index 0 + padded-key masking: changing a NON-[CLS],
      non-padded token must move the logit (real tokens are attended to).
    - Padded positions are masked out (mask=0), so overwriting ONLY padded
      slots must leave the logit unchanged.
    """
    torch.manual_seed(0)
    model = CrossEncoderReranker(make_config())
    model.eval()

    # Explicit layout: [CLS] q q [SEP] p p [SEP] (pad) (pad)
    real = [CLS_ID, 5, 6, SEP_ID, 7, 8, SEP_ID]
    n_pad = 2
    input_ids = torch.tensor([real + [PAD_ID] * n_pad], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1] * len(real) + [0] * n_pad], dtype=torch.long
    )

    with torch.no_grad():
        base = model(input_ids, attention_mask=attention_mask)

    # (a) Alter ONLY padded positions -> logit must NOT change.
    padded_only = input_ids.clone()
    padded_only[0, len(real):] = 9  # non-pad ids sitting where mask=0
    with torch.no_grad():
        masked_change = model(padded_only, attention_mask=attention_mask)
    assert torch.allclose(base, masked_change, atol=1e-6)

    # (b) Alter a real, non-[CLS] token (index 2) -> logit MUST change.
    real_change_ids = input_ids.clone()
    real_change_ids[0, 2] = 10
    with torch.no_grad():
        real_change = model(real_change_ids, attention_mask=attention_mask)
    assert not torch.allclose(base, real_change, atol=1e-6)


def test_sep_index_and_query_passage_shape_exclude_trailing_sep():
    """sep_index is the first [SEP]; the query x passage block excludes the
    trailing [SEP] and padding, so its last two dims are exactly
    (num query tokens, num passage tokens)."""
    model = CrossEncoderReranker(make_config(), tokenizer=StubTokenizer())

    # Two query words, three passage words -> known layout:
    # [CLS] q q [SEP] p p p [SEP]  (first [SEP] at index 3, second at 7)
    _, info = model.score_with_attention("alpha beta", "gamma delta epsilon")

    assert info["sep_index"] == 3
    qp = info["query_passage_attention"]
    # 2 query tokens (rows 1,2), 3 passage tokens (cols 4,5,6) — NOT the SEP at 7.
    assert qp.shape[-2:] == (2, 3)


# --- reranker: gradients flow ---------------------------------------------
def test_gradients_flow_through_reranker():
    model = CrossEncoderReranker(make_config())
    input_ids = torch.randint(0, VOCAB_SIZE, (3, 8))
    attention_mask = torch.ones(3, 8, dtype=torch.long)
    labels = torch.tensor([1.0, 0.0, 1.0])

    logits = model(input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    # BCE loss on the logit must populate grads on reranker params.
    assert model.scorer.weight.grad is not None
    assert torch.any(model.scorer.weight.grad != 0)
    embed_grad = model.encoder.token_embedding.weight.grad
    assert embed_grad is not None
    assert torch.any(embed_grad != 0)
