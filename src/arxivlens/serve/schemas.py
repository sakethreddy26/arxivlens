"""Pydantic request/response schemas for the ArXivLens FastAPI service.

All models use strict types with Field descriptions — this doubles as API
documentation (FastAPI renders these in the /docs OpenAPI UI).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., description="Free-text search query", min_length=1, max_length=512)
    top_n: int = Field(10, description="Number of reranked results to return", ge=1, le=100)


class PaperResult(BaseModel):
    """A single reranked paper returned inside SearchResponse."""

    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    score: float = Field(
        ..., description="Cross-encoder relevance logit (higher = more relevant)"
    )
    rank: int = Field(..., description="1-based rank after reranking (1 = most relevant)")


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str = Field(..., description="The original query string")
    results: list[PaperResult] = Field(..., description="Reranked papers, rank 1 first")
    retrieval_count: int = Field(
        ...,
        description="Configured retrieval cap (maximum candidates fetched from FAISS before reranking). Actual candidate count may be lower if the index is smaller than this cap.",
    )


class ExplainRequest(BaseModel):
    """Request body for POST /explain."""

    query: str = Field(..., description="The query to explain", min_length=1, max_length=512)
    paper_id: str = Field(..., description="ID of the paper to explain relevance for")


class ExplainResponse(BaseModel):
    """Response body for POST /explain.

    The ``attention_weights`` field is a 2-D matrix of shape
    ``query_tokens × passage_tokens``, averaged over all transformer layers and
    attention heads.  It is intended as an interpretability aid, not a
    definitive explanation (see 'Attention is not Explanation',
    Jain & Wallace 2019).
    """

    query: str
    paper_id: str
    title: str
    tokens: list[str] | None = Field(
        None, description="Token strings for heatmap axis labels"
    )
    attention_weights: list[list[float]] = Field(
        ...,
        description=(
            "2-D attention matrix: query_tokens × passage_tokens. "
            "Averaged over all transformer layers and attention heads. "
            "NOTE: attention weights are an interpretability aid, not a definitive "
            "explanation (see 'Attention is not Explanation', Jain & Wallace 2019)."
        ),
    )
    score: float = Field(
        ..., description="Cross-encoder relevance logit for this pair"
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="'ok' when the service is healthy")
    retriever: str = Field(..., description="Retriever model name or 'stub'")
    reranker: str = Field(..., description="Reranker description")
