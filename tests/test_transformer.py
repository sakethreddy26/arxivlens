"""CPU-friendly shape / correctness / gradient tests for the from-scratch
transformer. Configs are tiny so the whole file runs in
seconds: d_model=32, n_heads=4, n_layers=2, vocab_size=8.
"""

import math

import pytest
import torch

from arxivlens.model.transformer import (
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    TransformerConfig,
    TransformerEncoder,
)

VOCAB_SIZE = 8
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
MAX_LEN = 16
D_HEAD = D_MODEL // N_HEADS

BATCH = 2
SEQ_LEN = 10


@pytest.fixture()
def config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=0.0,  # deterministic forward passes for exact assertions
    )


@pytest.fixture()
def encoder(config: TransformerConfig) -> TransformerEncoder:
    torch.manual_seed(0)
    model = TransformerEncoder(config)
    model.eval()
    return model


@pytest.fixture()
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture()
def padding_mask() -> torch.Tensor:
    """Last 3 positions of every sequence are padding."""
    mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    mask[:, -3:] = 0
    return mask


# ---------------------------------------------------------------------------
# ScaledDotProductAttention
# ---------------------------------------------------------------------------


class TestScaledDotProductAttention:
    def test_output_shapes(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)

        output, weights = attn(q, k, v)

        assert output.shape == (BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_weights_are_valid_distribution(self):
        attn = ScaledDotProductAttention()
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)

        _, weights = attn(q, k, v)

        assert (weights >= 0).all()
        sums = weights.sum(dim=-1)  # (BATCH, N_HEADS, SEQ_LEN)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_returned_weights_are_pre_dropout_in_train_mode(self):
        """Returned attention weights ignore dropout: even with dropout=0.5 in
        .train() mode they are the pre-dropout softmax and sum to 1 (only the
        internal copy used for the output has positions zeroed)."""
        torch.manual_seed(0)
        attn = ScaledDotProductAttention(dropout=0.5)
        attn.train()
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)

        _, weights = attn(q, k, v)

        assert (weights >= 0).all()
        sums = weights.sum(dim=-1)  # (BATCH, N_HEADS, SEQ_LEN)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_matches_torch_reference(self):
        """Cross-check against torch.nn.functional.scaled_dot_product_attention
        to pin the √d_head scaling and bool-mask semantics."""
        torch.manual_seed(0)
        attn = ScaledDotProductAttention()  # no dropout
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD)

        # Unmasked.
        output, _ = attn(q, k, v)
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(output, reference, atol=1e-5)

        # With a bool attn_mask: True = "may attend". F.sdpa uses the same
        # convention for a bool mask, so both must agree.
        mask = torch.rand(BATCH, N_HEADS, SEQ_LEN, SEQ_LEN) > 0.3
        mask[..., 0] = True  # guarantee every row has at least one live key
        output_m, _ = attn(q, k, v, mask=mask)
        reference_m = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        assert torch.allclose(output_m, reference_m, atol=1e-5)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def test_output_shapes(self):
        mha = MultiHeadAttention(D_MODEL, N_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        output, weights = mha(x, x, x)

        assert output.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_per_head_weights_sum_to_one(self):
        mha = MultiHeadAttention(D_MODEL, N_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        _, weights = mha(x, x, x)

        sums = weights.sum(dim=-1)  # (BATCH, N_HEADS, SEQ_LEN)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_store_attention_caches_weights(self):
        mha = MultiHeadAttention(D_MODEL, N_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        assert mha.last_attention is None
        _, weights = mha(x, x, x, store_attention=True)

        assert mha.last_attention is not None
        assert mha.last_attention.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)
        assert torch.equal(mha.last_attention, weights)
        assert not mha.last_attention.requires_grad  # detached cache

    def test_rejects_indivisible_heads(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=30, n_heads=4)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(D_MODEL, MAX_LEN)
        x = torch.zeros(BATCH, SEQ_LEN, D_MODEL)

        assert pe(x).shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_position_zero_values(self):
        """At pos 0: sin(0) = 0 on even dims, cos(0) = 1 on odd dims."""
        pe = PositionalEncoding(D_MODEL, MAX_LEN)
        table = pe.pe  # (1, MAX_LEN, D_MODEL)

        assert torch.allclose(table[0, 0, 0::2], torch.zeros(D_MODEL // 2))
        assert torch.allclose(table[0, 0, 1::2], torch.ones(D_MODEL // 2))

    def test_position_one_matches_formula(self):
        """Spot-check PE(1, 2i) = sin(1 / 10000^(2i/d_model))."""
        pe = PositionalEncoding(D_MODEL, MAX_LEN)
        table = pe.pe

        for i in (0, 1, D_MODEL // 2 - 1):
            expected = math.sin(1.0 / (10000.0 ** (2 * i / D_MODEL)))
            assert table[0, 1, 2 * i].item() == pytest.approx(expected, abs=1e-5)

    def test_is_buffer_not_parameter(self):
        pe = PositionalEncoding(D_MODEL, MAX_LEN)

        assert "pe" in dict(pe.named_buffers())
        assert len(list(pe.parameters())) == 0

    def test_rejects_too_long_sequence(self):
        pe = PositionalEncoding(D_MODEL, MAX_LEN)
        x = torch.zeros(1, MAX_LEN + 1, D_MODEL)

        with pytest.raises(ValueError):
            pe(x)


# ---------------------------------------------------------------------------
# FeedForward / EncoderLayer
# ---------------------------------------------------------------------------


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(D_MODEL, D_FF)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        assert ffn(x).shape == (BATCH, SEQ_LEN, D_MODEL)


class TestEncoderLayer:
    def test_output_shapes(self):
        layer = EncoderLayer(D_MODEL, N_HEADS, D_FF)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        output, weights = layer(x)

        assert output.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)


# ---------------------------------------------------------------------------
# TransformerEncoder (full stack)
# ---------------------------------------------------------------------------


class TestTransformerEncoder:
    def test_forward_shapes(self, encoder, input_ids):
        hidden, attentions = encoder(input_ids)

        assert hidden.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert attentions is None  # not requested

    def test_return_attention_all_layers(self, encoder, input_ids):
        _, attentions = encoder(input_ids, return_attention=True)

        assert attentions is not None
        assert len(attentions) == N_LAYERS
        for layer_weights in attentions:
            assert layer_weights.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)
            sums = layer_weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_padding_mask_zeroes_attention_to_padded_keys(
        self, encoder, input_ids, padding_mask
    ):
        _, attentions = encoder(
            input_ids, attention_mask=padding_mask, return_attention=True
        )

        for layer_weights in attentions:
            # (BATCH, N_HEADS, SEQ_LEN, 3): weight put on the padded keys.
            weight_on_padding = layer_weights[..., -3:]
            assert torch.allclose(
                weight_on_padding, torch.zeros_like(weight_on_padding), atol=1e-6
            )
            # Rows still sum to 1 over the surviving (real) key positions.
            sums = layer_weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_padding_content_does_not_change_real_outputs(self, encoder, padding_mask):
        """Changing token ids in PADDED slots must not affect real positions."""
        torch.manual_seed(2)
        ids_a = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        ids_b = ids_a.clone()
        ids_b[:, -3:] = (ids_b[:, -3:] + 1) % VOCAB_SIZE  # perturb padding only

        hidden_a, _ = encoder(ids_a, attention_mask=padding_mask)
        hidden_b, _ = encoder(ids_b, attention_mask=padding_mask)

        assert torch.allclose(hidden_a[:, :-3], hidden_b[:, :-3], atol=1e-6)

    def test_gradients_flow_to_all_parameters(self, config, input_ids):
        torch.manual_seed(3)
        model = TransformerEncoder(config)
        model.train()

        hidden, _ = model(input_ids)
        loss = hidden.pow(2).mean()  # scalar
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"no gradient reached {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"

    def test_store_attention_flag_reaches_every_layer(self, encoder, input_ids):
        encoder(input_ids, store_attention=True)

        for layer in encoder.layers:
            cached = layer.attention.last_attention
            assert cached is not None
            assert cached.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_rejects_bad_config(self):
        with pytest.raises(ValueError):
            TransformerConfig(vocab_size=VOCAB_SIZE, d_model=30, n_heads=4)
