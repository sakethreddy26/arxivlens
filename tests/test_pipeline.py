"""Tests for RetrieveReranker pipeline and FastAPI endpoints.

Runs on CPU only — no FAISS, no sentence-transformers, no HuggingFace downloads.
Stubs satisfy the RetrieverLike / RerankerLike structural protocols defined in
pipeline.py so the full pipeline code paths execute under controlled data.
"""
from __future__ import annotations

import torch
import pytest

from arxivlens.retrieve.pipeline import ExplainInfo, RetrieveReranker
from arxivlens.serve.api import app, get_pipeline
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Stub data                                                                    #
# --------------------------------------------------------------------------- #

STUB_CANDIDATES = [
    {"paper_id": "p1", "title": "Attention Is All You Need", "abstract": "We propose transformers."},
    {"paper_id": "p2", "title": "BERT Pre-training", "abstract": "Bidirectional encoder."},
    {"paper_id": "p3", "title": "GPT Language Models", "abstract": "Unsupervised multitask."},
    {"paper_id": "p4", "title": "Word2Vec", "abstract": "Efficient word embeddings."},
    {"paper_id": "p5", "title": "ResNet", "abstract": "Deep residual learning."},
]
STUB_SCORES = [0.9, 0.7, 0.5, 0.3, 0.1]  # aligned to STUB_CANDIDATES order


class StubRetriever:
    def retrieve(self, query: str, k: int = 50):
        return STUB_CANDIDATES[:k]


class StubReranker:
    """Returns STUB_SCORES in order for any input."""

    def score(self, query, passages):
        n = len(passages)
        assert n <= len(STUB_SCORES), (
            f"StubReranker only has {len(STUB_SCORES)} scores, got {n} passages"
        )
        return torch.tensor(STUB_SCORES[:n], dtype=torch.float32)

    def score_with_attention(self, query, passage):
        score = torch.tensor(0.9)
        attn_info = {
            "tokens": ["[CLS]", "test", "[SEP]", "paper", "[SEP]"],
            "sep_index": 2,
            "query_passage_attention": torch.ones(2, 4, 2, 3),  # (layers, heads, q_toks, p_toks)
            "weights": torch.ones(2, 4, 5, 5),
        }
        return score, attn_info


class RecordingReranker(StubReranker):
    def __init__(self):
        self.passages = []

    def score(self, query, passages):
        self.passages.append(list(passages))
        return super().score(query, passages)

    def score_with_attention(self, query, passage):
        self.passages.append([passage])
        return super().score_with_attention(query, passage)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_pipeline():
    return RetrieveReranker(retriever=StubRetriever(), reranker=StubReranker(), retrieve_k=5)


@pytest.fixture
def client(stub_pipeline):
    app.dependency_overrides[get_pipeline] = lambda: stub_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Pipeline unit tests                                                          #
# --------------------------------------------------------------------------- #


def test_search_returns_list(stub_pipeline):
    """search() must return a plain list."""
    result = stub_pipeline.search("transformers", top_n=3)
    assert isinstance(result, list)


def test_search_result_count(stub_pipeline):
    """top_n=3 with 5 candidates available must yield exactly 3 results."""
    result = stub_pipeline.search("transformers", top_n=3)
    assert len(result) == 3


def test_search_ordered_by_score(stub_pipeline):
    """Results must be monotonically non-increasing in score."""
    results = stub_pipeline.search("transformers", top_n=5)
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
        f"Scores not sorted descending: {scores}"
    )


def test_search_rank_field(stub_pipeline):
    """results[i].rank must equal i + 1 (1-based) for every position."""
    results = stub_pipeline.search("transformers", top_n=5)
    for i, r in enumerate(results):
        assert r.rank == i + 1, f"Expected rank {i + 1}, got {r.rank}"


def test_search_result_has_all_fields(stub_pipeline):
    """Every result must carry paper_id, title, abstract, score, and rank."""
    results = stub_pipeline.search("transformers", top_n=3)
    for r in results:
        assert hasattr(r, "paper_id") and r.paper_id
        assert hasattr(r, "title") and r.title
        assert hasattr(r, "abstract") and r.abstract
        assert hasattr(r, "score")
        assert hasattr(r, "rank")


def test_search_top1_is_highest_scored(stub_pipeline):
    """The first result must be the paper with score 0.9, which is p1."""
    results = stub_pipeline.search("transformers", top_n=5)
    assert results[0].paper_id == "p1", (
        f"Expected p1 at rank 1, got {results[0].paper_id} (score={results[0].score})"
    )


def test_search_fewer_than_top_n(stub_pipeline):
    """When top_n exceeds available candidates, all candidates are returned."""
    # retrieve_k=5, top_n=10: only 5 candidates exist, so 5 must be returned.
    results = stub_pipeline.search("transformers", top_n=10)
    assert len(results) == 5


def test_abstract_passage_format_is_used_for_search_and_explain():
    reranker = RecordingReranker()
    pipeline = RetrieveReranker(
        retriever=StubRetriever(),
        reranker=reranker,
        retrieve_k=5,
        passage_format="abstract",
    )

    pipeline.search("transformers", top_n=1)
    pipeline.explain("transformers", "p1")

    assert reranker.passages[0] == [c["abstract"] for c in STUB_CANDIDATES]
    assert reranker.passages[1] == [STUB_CANDIDATES[0]["abstract"]]


def test_unknown_passage_format_is_rejected():
    with pytest.raises(ValueError, match="passage_format"):
        RetrieveReranker(
            retriever=StubRetriever(),
            reranker=StubReranker(),
            passage_format="unknown",
        )


def test_explain_returns_explain_info(stub_pipeline):
    """explain() must return an ExplainInfo instance."""
    info = stub_pipeline.explain("transformers", "p1")
    assert isinstance(info, ExplainInfo)


def test_explain_attention_is_2d_list(stub_pipeline):
    """attention_weights must be a non-empty list of lists (2-D matrix)."""
    info = stub_pipeline.explain("transformers", "p1")
    weights = info.attention_weights
    assert isinstance(weights, list) and len(weights) > 0, "attention_weights is empty"
    assert all(isinstance(row, list) for row in weights), (
        "attention_weights rows must themselves be lists"
    )
    assert all(len(row) > 0 for row in weights), "inner lists must be non-empty"


def test_explain_invalid_paper_id_raises(stub_pipeline):
    """explain() must raise ValueError when paper_id is not in candidates."""
    with pytest.raises(ValueError, match="nonexistent"):
        stub_pipeline.explain("transformers", "nonexistent")


def test_explain_score_is_float(stub_pipeline):
    """explain() score field must be a Python float."""
    info = stub_pipeline.explain("transformers", "p1")
    assert isinstance(info.score, float), f"Expected float, got {type(info.score)}"


# --------------------------------------------------------------------------- #
# FastAPI integration tests                                                    #
# --------------------------------------------------------------------------- #


def test_health_200(client):
    """GET /health must return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(client):
    """GET /health response body must contain status == 'ok'."""
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_search_200(client):
    """POST /search with a valid query must return HTTP 200."""
    response = client.post("/search", json={"query": "transformers"})
    assert response.status_code == 200


def test_search_results_nonempty(client):
    """POST /search must return a non-empty results list."""
    response = client.post("/search", json={"query": "transformers"})
    assert len(response.json()["results"]) > 0


def test_search_result_fields(client):
    """First result from POST /search must have all expected fields."""
    response = client.post("/search", json={"query": "transformers"})
    first = response.json()["results"][0]
    for field in ("paper_id", "title", "abstract", "score", "rank"):
        assert field in first, f"Missing field: {field}"


def test_search_results_ordered(client):
    response = client.post("/search", json={"query": "transformers"})
    assert response.status_code == 200
    results = response.json()["results"]
    scores = [r["score"] for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
        f"Results not sorted by score descending: {scores}"
    )


def test_search_top_n_param(client):
    """POST /search with top_n=2 must return exactly 2 results."""
    response = client.post("/search", json={"query": "transformers", "top_n": 2})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_explain_200(client):
    """POST /explain with a known paper_id must return HTTP 200."""
    response = client.post("/explain", json={"query": "transformers", "paper_id": "p1"})
    assert response.status_code == 200


def test_explain_has_attention_weights(client):
    """POST /explain response must include the attention_weights key."""
    response = client.post("/explain", json={"query": "transformers", "paper_id": "p1"})
    assert "attention_weights" in response.json()


def test_explain_404_on_unknown_paper(client):
    """POST /explain with an unknown paper_id must return HTTP 404."""
    response = client.post("/explain", json={"query": "transformers", "paper_id": "zzz"})
    assert response.status_code == 404


def test_search_empty_query_422(client):
    """POST /search with an empty query string must return HTTP 422 (validation error)."""
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422
