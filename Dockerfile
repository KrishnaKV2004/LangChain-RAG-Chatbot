# syntax=docker/dockerfile:1
#
# One image, two entrypoints: the API (default CMD) and the Streamlit UI
# (override the command). Both need the same code, so one build serves both.
#
#   docker build -t rag-agent .
#   docker run --env-file .env -p 8000:8000 rag-agent
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the HF model cache at a fixed, pre-populated path.
    HF_HOME=/models \
    # Chroma/telemetry off — no phoning home from a server container.
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Build toolchain is needed by a few wheels, then removed so it isn't shipped.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/*

# Bake the local embedding + reranker models into the image. Without this the
# first request downloads ~500 MB, which makes container start slow AND makes
# the app depend on HuggingFace being reachable at runtime.
# Build with --build-arg PREFETCH_MODELS=false if you use hosted embeddings
# (EMBEDDING__PROVIDER=openai) and RETRIEVAL__RERANK_ENABLED=false.
ARG PREFETCH_MODELS=true
RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
      python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder;\
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2');\
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

# `documents/` ships with the image: Warehouses.csv is the truck-leg whitelist
# the rate planner reads at runtime, not just reference material.
COPY app ./app
COPY frontend ./frontend
COPY documents ./documents

# Non-root, and the data dirs exist up front so a mounted volume inherits them.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data/chroma /app/data/cache \
 && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000 8501

# /health sits behind require_auth, so send the token when one is configured.
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=180s \
  CMD python -c "\
import os,urllib.request as u;\
t=os.getenv('API__AUTH_TOKEN');\
r=u.Request('http://127.0.0.1:8000/health');\
t and r.add_header('Authorization','Bearer '+t);\
u.urlopen(r,timeout=8)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
