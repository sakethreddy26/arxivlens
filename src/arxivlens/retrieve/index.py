"""Dense retrieval over ArXiv papers via a FAISS index.

On Sol, the index lives at /scratch/spate472/mlrag/:
  index/index.faiss  — FAISS FlatIP index (384-dim MiniLM embeddings)
  index/meta.jsonl   — one JSON object per line in FAISS row order,
                       keys: paper_id (or id/arxiv_id fallback), title, abstract

FaissRetriever embeds the query with the same MiniLM model that built the
index (sentence-transformers/all-MiniLM-L6-v2), then does index.search() for
the top-k nearest neighbours. FAISS and sentence-transformers are lazy-imported
so the module is importable in CI without them installed.

Retriever vs. reranker — why two stages
---------------------------------------
The FAISS flat-IP search scales to millions of papers in milliseconds by using
dense vectors (precomputed offline). It cannot match cross-encoder quality
because query and passage are encoded independently. The retrieved top-k (50–100)
are then re-scored by CrossEncoderReranker which does see the query-passage
interaction — but only over the small candidate set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# _META_ID_KEYS: checked left-to-right; first present key wins.
_META_ID_KEYS: tuple[str, ...] = ("paper_id", "id", "arxiv_id")

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _extract_paper_id(meta_row: dict[str, Any]) -> str:
    """Return the paper id from a meta row, trying keys in _META_ID_KEYS order.

    Raises KeyError if none of the candidate keys are present.
    """
    for key in _META_ID_KEYS:
        if key in meta_row:
            return str(meta_row[key])
    raise KeyError(
        f"meta row has none of the expected id keys {_META_ID_KEYS!r}; "
        f"got keys: {list(meta_row.keys())!r}"
    )


class FaissRetriever:
    """Load a FAISS index + meta.jsonl and retrieve top-k candidates for a query.

    Args:
        index_path: path to index.faiss
        meta_path:  path to meta.jsonl (one JSON object per line, FAISS-row order)
        model_name: SentenceTransformer model name used to build the index
                    (default: "sentence-transformers/all-MiniLM-L6-v2")

    The number of rows in meta.jsonl MUST equal index.ntotal; the constructor
    asserts this at load time and raises RuntimeError with a clear message if not.

    retrieve(query, k) -> list[dict]  — each dict has at minimum:
        paper_id: str   (from "paper_id" / "id" / "arxiv_id" in meta)
        title:    str   (from "title" in meta, empty string if absent)
        abstract: str   (from "abstract" in meta, empty string if absent)
    The returned list is in decreasing similarity order (rank 1 first).
    """

    def __init__(
        self,
        index_path: str | Path,
        meta_path: str | Path,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        # Lazy imports: faiss and sentence_transformers are NOT imported at module
        # top level so that `from arxivlens.retrieve.index import FaissRetriever`
        # works in CI environments where neither package is installed.
        import faiss  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

        self._index = faiss.read_index(str(index_path))

        # Load all meta rows into memory; order must match FAISS row order.
        self._meta: list[dict[str, Any]] = []
        with open(meta_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._meta.append(json.loads(line))

        # Sanity-check: meta count must equal index size.
        if len(self._meta) != self._index.ntotal:
            raise RuntimeError(
                f"meta.jsonl has {len(self._meta)} rows but index.ntotal="
                f"{self._index.ntotal}; they must match (FAISS row order)"
            )

    def retrieve(self, query: str, k: int = 50) -> list[dict[str, Any]]:
        """Embed query and return the top-k nearest papers from the FAISS index.

        Args:
            query: natural-language query string.
            k:     number of candidates to retrieve (default 50).

        Returns:
            List of dicts in decreasing similarity order (rank 1 first).
            Each dict contains at minimum:
                paper_id (str), title (str), abstract (str).
            Any additional keys from the original meta row are also included.
            Results with a FAISS id of -1 (returned when the index has fewer
            than k entries) are silently skipped.
        """
        # Embed the query; normalize_embeddings=True matches the inner-product
        # index (cosine similarity via normalized dot-product).
        query_vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # shape: (1, 384)

        # index.search returns (distances, ids), both shape (1, k).
        _distances, faiss_ids = self._index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for fid in faiss_ids[0]:  # iterate over the k nearest neighbour ids
            if fid == -1:
                # FAISS returns -1 when the index has fewer than k entries.
                continue
            row = self._meta[fid]
            try:
                paper_id = _extract_paper_id(row)
            except KeyError:
                # Skip rows with no recognisable paper_id key — malformed meta.jsonl entry
                continue
            results.append(
                {
                    "paper_id": paper_id,
                    "title": str(row.get("title", "")),
                    "abstract": str(row.get("abstract", "")),
                    # Carry through any extra keys (e.g. categories, year).
                    **{k: v for k, v in row.items() if k not in ("title", "abstract")},
                }
            )
        return results
