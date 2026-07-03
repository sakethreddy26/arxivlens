# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build tools needed by some Python wheels (e.g. faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only torch first so pip doesn't pull the 2 GB CUDA wheel
# when it resolves requirements.txt later.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.7.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install the arxivlens package itself (editable installs don't copy well
# across stages, so we do a regular install into the default site-packages).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy installed Python packages and binaries from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (needed for the installed package to resolve correctly)
WORKDIR /app
COPY src/ src/
COPY configs/ configs/

# Make the src layout importable (belt-and-suspenders alongside the installed pkg)
ENV PYTHONPATH=/app/src

# Runtime defaults — override via `docker run -e` or compose environment:
#   INDEX_PATH   path to index.faiss   (required in production)
#   META_PATH    path to meta.jsonl    (required in production)
#   CHECKPOINT   path to a .pt file    (optional)
#   TOKENIZER    HF tokenizer name     (default: bert-base-uncased)
#   RETRIEVE_K   FAISS candidates cap  (default: 50)
ENV TOKENIZER=bert-base-uncased \
    RETRIEVE_K=50

EXPOSE 8000

CMD ["uvicorn", "arxivlens.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
