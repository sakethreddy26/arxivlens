"""FastAPI service for ArXivLens.

Endpoints:
  GET  /health   — liveness check
  POST /search   — retrieve and rerank papers for a query
  POST /explain  — return cross-encoder attention weights for a (query, paper) pair

The pipeline (retriever + reranker) is provided via FastAPI dependency injection
(``Depends(get_pipeline)``) so integration tests can override it without
monkey-patching. In production, ``get_pipeline()`` builds the singleton from
environment variables; in tests, ``app.dependency_overrides[get_pipeline]`` injects
a stub pipeline.

Usage (production):
    INDEX_PATH=/scratch/spate472/mlrag/index/index.faiss \\
    META_PATH=/scratch/spate472/mlrag/index/meta.jsonl \\
    CHECKPOINT=/scratch/spate472/mlrag/checkpoints/checkpoint_epoch0004_step*.pt \\
    uvicorn arxivlens.serve.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, HTTPException

from arxivlens.serve.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PaperResult,
    SearchRequest,
    SearchResponse,
)

# Pipeline types are imported only for static analysis; at runtime the module
# ``arxivlens.retrieve.pipeline`` is loaded lazily inside get_pipeline() and
# the endpoint functions.  This lets ``from arxivlens.serve.api import app``
# succeed in tests (and in CI) before pipeline.py exists or before its heavy
# runtime dependencies (faiss, torch, transformers) are installed.
if TYPE_CHECKING:
    from arxivlens.retrieve.pipeline import ExplainInfo, RankedResult, RetrieveReranker


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    """Build and cache the singleton pipeline from environment variables.

    Environment variables:
        INDEX_PATH   — path to index.faiss (required in production)
        META_PATH    — path to meta.jsonl  (required in production)
        CHECKPOINT   — path to a .pt checkpoint file (optional; loads trained weights)
        TOKENIZER    — HuggingFace tokenizer name (default: bert-base-uncased)
        RETRIEVE_K   — number of FAISS candidates to retrieve (default: 50)

    If INDEX_PATH / META_PATH are not set, raises RuntimeError with a clear
    message pointing to /scratch/spate472/mlrag/ on Sol.

    All heavy imports (faiss, torch, transformers) are inside this function so
    that ``from arxivlens.serve.api import app`` succeeds in tests without those
    packages being present.

    Returns:
        A ``RetrieveReranker`` instance (typed as ``Any`` to avoid a top-level
        import of the not-yet-existing pipeline module).

    The result is cached for the lifetime of the process (``@lru_cache``);
    changing environment variables after the first successful call has no effect.
    """
    from arxivlens.model.reranker import CrossEncoderReranker
    from arxivlens.model.transformer import TransformerConfig
    from arxivlens.retrieve.index import FaissRetriever
    from arxivlens.retrieve.pipeline import RetrieveReranker  # noqa: PLC0415

    import torch
    from transformers import AutoTokenizer

    index_path = os.environ.get("INDEX_PATH")
    meta_path = os.environ.get("META_PATH")
    if not index_path or not meta_path:
        raise RuntimeError(
            "INDEX_PATH and META_PATH environment variables must be set. "
            "On Sol: export INDEX_PATH=/scratch/spate472/mlrag/index/index.faiss "
            "META_PATH=/scratch/spate472/mlrag/index/meta.jsonl"
        )

    retriever = FaissRetriever(index_path, meta_path)

    tokenizer_name = os.environ.get("TOKENIZER", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Build the reranker architecture. When a checkpoint is supplied, the config
    # stored *inside* the checkpoint is authoritative (it is the arch the weights
    # were trained with) — mirrors slurm/eval_reranker.sh so shapes always match.
    checkpoint = os.environ.get("CHECKPOINT")
    passage_format = "title_abstract"
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        cfg = state["config"]
        passage_format = str(
            cfg.get("training", {}).get("eval_passage_format", passage_format)
        )
        m = cfg["model"]
        config = TransformerConfig(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            d_ff=m["d_ff"],
            max_len=m["max_len"],
            dropout=m.get("dropout", 0.1),
        )
        reranker = CrossEncoderReranker(config, tokenizer=tokenizer)
        reranker.load_state_dict(state["model_state_dict"])
        reranker.eval()
    else:
        # Dev / random-weights mode (no CHECKPOINT). These values mirror the
        # current configs/reranker.yaml; a real CHECKPOINT overrides them with
        # the arch stored in the checkpoint above.
        config = TransformerConfig(
            vocab_size=30522,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_len=512,
        )
        reranker = CrossEncoderReranker(config, tokenizer=tokenizer)

    retrieve_k = int(os.environ.get("RETRIEVE_K", "50"))
    passage_format = os.environ.get("PASSAGE_FORMAT", passage_format)
    return RetrieveReranker(
        retriever=retriever,
        reranker=reranker,
        retrieve_k=retrieve_k,
        passage_format=passage_format,
    )


app = FastAPI(
    title="ArXivLens",
    description=(
        "Semantic search over ArXiv ML papers with a from-scratch cross-encoder reranker. "
        "Architecture: FAISS dense retrieval → cross-encoder reranking → optional LLM generation."
    ),
    version="0.1.0",
)

# Annotated alias so endpoint signatures stay concise and FastAPI still wires DI.
PipelineDep = Annotated[Any, Depends(get_pipeline)]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(pipeline: PipelineDep) -> HealthResponse:
    """Liveness check. Returns 200 when the pipeline is loaded and ready."""
    return HealthResponse(
        status="ok",
        retriever=type(pipeline.retriever).__name__,
        reranker=type(pipeline.reranker).__name__,
    )


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest, pipeline: PipelineDep) -> SearchResponse:
    """Retrieve and rerank arXiv papers for a free-text query."""
    results: list[Any] = pipeline.search(req.query, top_n=req.top_n)
    paper_results = [
        PaperResult(
            paper_id=r.paper_id,
            title=r.title,
            abstract=r.abstract,
            score=r.score,
            rank=r.rank,
        )
        for r in results
    ]
    return SearchResponse(
        query=req.query,
        results=paper_results,
        retrieval_count=pipeline.retrieve_k,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["search"])
def explain(req: ExplainRequest, pipeline: PipelineDep) -> ExplainResponse:
    """Return cross-encoder attention weights for a specific (query, paper) pair.

    Raises 404 if ``paper_id`` is not found in the index.
    """
    try:
        info: Any = pipeline.explain(req.query, req.paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExplainResponse(
        query=req.query,
        paper_id=info.paper_id,
        title=info.title,
        tokens=info.tokens,
        attention_weights=info.attention_weights,
        score=info.score,
    )
