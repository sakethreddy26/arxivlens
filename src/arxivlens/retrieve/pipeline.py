"""Retrieve-then-rerank pipeline and result types.

Architecture
------------
Stage 1 — Dense retrieval: FaissRetriever embeds the query with MiniLM and
returns top-k candidate papers from the FAISS index (~50 ms for k=50).

Stage 2 — Cross-encoder reranking: CrossEncoderReranker scores each candidate
as a (query, passage) pair through the from-scratch transformer. Much slower
per candidate (full forward pass) but only runs over the small top-k, not the
whole corpus.

Stage 3 (future seam) — Generation: a clean hook is left for a vLLM-based LLM
to generate a grounded answer over the reranked passages. This module does NOT
implement generation; it returns structured results that a generator can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class RankedResult:
    """One reranked paper returned by the pipeline."""

    paper_id: str
    title: str
    abstract: str
    score: float  # cross-encoder relevance logit (higher = more relevant)
    rank: int  # 1-based rank after reranking


@dataclass
class ExplainInfo:
    """Attention-based explanation for a (query, paper) pair."""

    paper_id: str
    title: str
    tokens: list[str] | None  # token strings for heatmap axis labels
    attention_weights: list[list[float]]  # query_tokens × passage_tokens, avg over layers+heads
    score: float


# --------------------------------------------------------------------------- #
# Structural protocols for dependency injection                                #
# --------------------------------------------------------------------------- #


@runtime_checkable
class RetrieverLike(Protocol):
    """Minimal interface expected from a retriever.

    Production implementation: FaissRetriever.
    Tests can pass any object with a matching .retrieve() signature.
    """

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class RerankerLike(Protocol):
    """Minimal interface expected from a reranker.

    Production implementation: CrossEncoderReranker.
    Tests can pass any object with matching .score() / .score_with_attention().
    """

    def score(self, query: str, passages: Sequence[str]) -> Any: ...

    def score_with_attention(self, query: str, passage: str) -> tuple[Any, dict[str, Any]]: ...


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #


class RetrieveReranker:
    """Two-stage retrieve-then-rerank pipeline.

    Args:
        retriever:   any object with .retrieve(query: str, k: int) -> list[dict]
                     (production: FaissRetriever; tests: stub)
        reranker:    CrossEncoderReranker instance (or any object with
                     .score(query, passages) -> Tensor and
                     .score_with_attention(query, passage) -> (score, attention_info))
        retrieve_k:  number of candidates to pull from the retriever before reranking
    """

    def __init__(
        self,
        retriever: RetrieverLike,
        reranker: RerankerLike,
        retrieve_k: int = 50,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k

    def search(self, query: str, top_n: int = 10) -> list[RankedResult]:
        """Full pipeline: retrieve then rerank, return top_n results.

        Retrieves retrieve_k candidates from the dense index, re-scores all of
        them with the cross-encoder, and returns the top_n sorted by score
        descending (rank 1 = most relevant). If fewer than top_n candidates are
        retrieved, all of them are returned.

        Args:
            query: natural-language query string.
            top_n: maximum number of results to return (default 10).

        Returns:
            List of RankedResult, sorted by score descending, rank 1 first.
        """
        candidates = self.retriever.retrieve(query, k=self.retrieve_k)
        if not candidates:
            return []

        passages = [f"{c['title']} {c['abstract']}".strip() for c in candidates]
        scores = self.reranker.score(query, passages)  # Tensor (n_candidates,)
        scores_list = scores.cpu().tolist()

        ranked = sorted(
            zip(candidates, scores_list),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            RankedResult(
                paper_id=c["paper_id"],
                title=c["title"],
                abstract=c["abstract"],
                score=float(s),
                rank=i + 1,
            )
            for i, (c, s) in enumerate(ranked[:top_n])
        ]

    def explain(self, query: str, paper_id: str) -> ExplainInfo:
        """Score a specific paper against the query and extract the attention heatmap.

        Retrieves retrieve_k candidates from the dense index, locates the paper
        by paper_id, then runs a single cross-encoder forward pass with attention
        extraction. The query_passage_attention sub-block is averaged over all
        layers and heads to produce a (q_tokens × p_tokens) heatmap matrix.

        Used by POST /explain in the API.

        Args:
            query:    natural-language query string.
            paper_id: the paper to explain — must appear in the top-retrieve_k
                      candidates for this query.

        Returns:
            ExplainInfo with attention_weights averaged over layers and heads.

        Raises:
            ValueError: if paper_id is not found in the retrieved candidates.
        """
        candidates = self.retriever.retrieve(query, k=self.retrieve_k)
        target = next((c for c in candidates if c["paper_id"] == paper_id), None)
        if target is None:
            raise ValueError(
                f"paper_id {paper_id!r} not found in top-{self.retrieve_k} candidates"
            )

        passage = f"{target['title']} {target['abstract']}".strip()
        score, attn_info = self.reranker.score_with_attention(query, passage)

        # qp_attn shape: (n_layers, n_heads, q_tokens, p_tokens) or None.
        qp_attn = attn_info["query_passage_attention"]
        if qp_attn is not None:
            # Average over layers (dim 0) and heads (dim 1) -> (q_tokens, p_tokens).
            avg_attn = qp_attn.mean(dim=(0, 1))
            attention_weights: list[list[float]] = avg_attn.cpu().tolist()
        else:
            attention_weights = []

        return ExplainInfo(
            paper_id=paper_id,
            title=target["title"],
            tokens=attn_info.get("tokens"),
            attention_weights=attention_weights,
            score=float(score),
        )
