"""Regression test for the P0 serving-config fix in ``arxivlens.serve.api``.

The bug: ``get_pipeline()`` used to build the reranker with a HARDCODED
architecture (d_model=256 in an earlier version, then 512 to match
configs/reranker.yaml). Loading a checkpoint whose stored architecture differed
from that hardcode raised a state_dict shape mismatch (a 512/6 checkpoint could
not load into a 256-wide model, and any future retrain that changes the arch
would silently break serving again).

The fix: when ``CHECKPOINT`` is set, ``get_pipeline()`` reads the architecture
from ``state["config"]["model"]`` inside the checkpoint and builds
``TransformerConfig`` from THAT, so the reranker's shape always matches the
saved weights by construction.

This test locks that in. It builds a real tiny checkpoint on disk with a
DISTINCTIVE architecture (d_model=32) that differs from BOTH the old hardcoded
256 AND the current yaml 512, then drives the real ``get_pipeline()`` code path.
``load_state_dict`` only succeeds if the config came from the checkpoint — a
hardcoded config would raise a shape mismatch. We then assert the resulting
reranker's encoder config reflects the checkpoint's tiny arch.

The retriever/pipeline collaborators and the tokenizer are monkeypatched so the
config-from-checkpoint logic is exercised WITHOUT any real FAISS files or network
access. The whole module skips cleanly where torch/transformers are absent (e.g.
this dev box); it runs for real in CI/Sol where those packages exist.
"""
from __future__ import annotations

import pytest

# Skip the entire module cleanly where the heavy deps are missing. These imports
# gate collection so the file has no import/syntax errors either way.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


# Distinctive tiny architecture stored inside the checkpoint. Crucially:
#   - d_model=32 differs from the OLD hardcoded 256 and the yaml 512, so
#     load_state_dict only succeeds if the config is read from the checkpoint.
#   - n_heads=4 divides d_model=32; d_model is even (PositionalEncoding needs it).
TINY_MODEL_CFG = {
    "vocab_size": 30522,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 64,
    "max_len": 64,
    "dropout": 0.1,
}


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer.

    The reranker only stores this on construction (it is not invoked here since
    we never call score()), so structural attributes are enough.
    """

    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, *args, **kwargs):  # pragma: no cover - not exercised here
        raise AssertionError("tokenizer should not be called during get_pipeline()")

    def convert_ids_to_tokens(self, ids):  # pragma: no cover - not exercised here
        return []


class _FakeRetriever:
    """Accepts (index_path, meta_path) and does nothing — no real FAISS files."""

    def __init__(self, index_path, meta_path):
        self.index_path = index_path
        self.meta_path = meta_path


class _FakeRetrieveReranker:
    """Stores the collaborators it is handed so the test can inspect them."""

    def __init__(self, retriever, reranker, retrieve_k):
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k


def _write_tiny_checkpoint(path) -> None:
    """Write a real .pt checkpoint whose weights MATCH TINY_MODEL_CFG.

    The state_dict is produced by constructing a CrossEncoderReranker with the
    same tiny config, so ``load_state_dict`` must succeed iff get_pipeline()
    rebuilds that same architecture from the checkpoint's stored config.
    """
    from arxivlens.model.reranker import CrossEncoderReranker
    from arxivlens.model.transformer import TransformerConfig

    m = TINY_MODEL_CFG
    config = TransformerConfig(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        d_ff=m["d_ff"],
        max_len=m["max_len"],
        dropout=m["dropout"],
    )
    reranker = CrossEncoderReranker(config)
    state = {
        "config": {"model": dict(TINY_MODEL_CFG)},
        "model_state_dict": reranker.state_dict(),
    }
    torch.save(state, str(path))


@pytest.fixture
def wired_env(tmp_path, monkeypatch):
    """Monkeypatch collaborators + env and yield a fresh, cache-cleared pipeline.

    Patches the source-module attributes that get_pipeline() imports lazily
    (``from arxivlens.retrieve.index import FaissRetriever`` etc.), so the real
    API code path runs but touches no FAISS files, network, or GPU.
    """
    from arxivlens.serve import api

    # Clear the lru_cache so a prior successful call can't leak a cached pipeline.
    api.get_pipeline.cache_clear()

    # Real checkpoint on disk with the distinctive tiny architecture.
    ckpt = tmp_path / "checkpoint_epoch0004.pt"
    _write_tiny_checkpoint(ckpt)

    # Swap heavy collaborators for fakes. get_pipeline() imports these lazily by
    # name from their source modules, so patching the module attribute suffices.
    monkeypatch.setattr("arxivlens.retrieve.index.FaissRetriever", _FakeRetriever)
    monkeypatch.setattr(
        "arxivlens.retrieve.pipeline.RetrieveReranker", _FakeRetrieveReranker
    )

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeTokenizer()),
    )

    # Env: real FAISS paths are only handed to the fake retriever, never opened.
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "index.faiss"))
    monkeypatch.setenv("META_PATH", str(tmp_path / "meta.jsonl"))
    monkeypatch.setenv("CHECKPOINT", str(ckpt))
    monkeypatch.setenv("TOKENIZER", "bert-base-uncased")
    monkeypatch.setenv("RETRIEVE_K", "7")

    yield api

    # Don't leak the cached pipeline (built with fakes/tmp paths) into other tests.
    api.get_pipeline.cache_clear()


def test_get_pipeline_loads_checkpoint_config(wired_env):
    """get_pipeline() must succeed and build the reranker from the checkpoint arch.

    If get_pipeline() still hardcoded d_model=512 (or the old 256), load_state_dict
    would raise a shape mismatch against the tiny (d_model=32) weights. That it
    succeeds proves the architecture was read from state["config"]["model"].
    """
    api = wired_env

    pipeline = api.get_pipeline()  # must not raise (load_state_dict succeeds)

    reranker = pipeline.reranker
    # The reranker's architecture must reflect the checkpoint, not a hardcode.
    assert reranker.config.d_model == TINY_MODEL_CFG["d_model"] == 32
    assert reranker.config.n_layers == TINY_MODEL_CFG["n_layers"]
    assert reranker.config.n_heads == TINY_MODEL_CFG["n_heads"]
    assert reranker.config.d_ff == TINY_MODEL_CFG["d_ff"]
    assert reranker.config.max_len == TINY_MODEL_CFG["max_len"]
    # And it must NOT be either hardcoded default that used to be baked in.
    assert reranker.config.d_model not in (256, 512)

    # The encoder itself was built at the checkpoint's width (proves the weights
    # actually loaded into a matching module, not just that config was stored).
    assert reranker.encoder.config.d_model == 32
    assert reranker.scorer.weight.shape == (1, 32)

    # Retriever got the env FAISS paths; retrieve_k came from RETRIEVE_K.
    assert isinstance(reranker, __import__(
        "arxivlens.model.reranker", fromlist=["CrossEncoderReranker"]
    ).CrossEncoderReranker)
    assert pipeline.retrieve_k == 7


def test_get_pipeline_reranker_is_eval_mode(wired_env):
    """A checkpoint-loaded reranker must be put in eval() mode for serving."""
    api = wired_env
    pipeline = api.get_pipeline()
    assert pipeline.reranker.training is False


def test_checkpoint_arch_differs_from_serving_default(wired_env):
    """Guard the guard: the tiny checkpoint arch must differ from the yaml default.

    If the checkpoint happened to use d_model=512 (the no-CHECKPOINT default),
    this regression test would pass even with the bug present. Assert the fixture
    keeps them distinct so the test remains meaningful.
    """
    assert TINY_MODEL_CFG["d_model"] not in (256, 512)
    assert TINY_MODEL_CFG["n_layers"] != 6  # yaml default n_layers
