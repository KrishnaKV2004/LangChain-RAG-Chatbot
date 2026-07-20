# Hybrid RAG AI Agent

An enterprise-grade Retrieval-Augmented Generation assistant for logistics
and general-knowledge questions. It combines **internal documents**
(ChromaDB), **live web search** (Tavily) and **LLM reasoning** into one
answer — behind a five-layer security stack that prevents leakage of
confidential company information.

Built with **LangChain · LangGraph · FastAPI · Streamlit · ChromaDB**.

## Features

- **Query routing** — an LLM classifier sends each question down the right
  path: `INTERNAL_ONLY`, `WEB_ONLY`, `HYBRID`, or `GENERAL_CHAT`.
- **9 document formats** — PDF, DOCX, TXT, Markdown, CSV, Excel, PowerPoint,
  HTML, JSON — with automatic chunking and citation-grade metadata
  (filename, page, section, creation date).
- **Retrieval funnel** — vector search → relevance threshold → MMR diversity
  → cross-encoder reranking → optional contextual compression.
- **Web search** — pluggable provider interface (Tavily default) with
  ranking, deduplication and page-text extraction; results cached with a
  24 h TTL and never permanently stored.
- **Five security layers** — prompt-injection detection (direct and
  indirect), sensitive-document detection, role-based permission validation,
  response leak scanning, PII redaction. See [docs/security.md](docs/security.md).
- **Hermes self-refinement** — a reviewer agent grades every draft answer
  (one committed answer, faithful to context, human tone, concise) and
  rewrites it when it falls short, before the security gate.
- **Grounded citations** — built programmatically from the exact sources in
  the LLM's context; fabrication is structurally impossible.
- **Observability** — structured logs (console or JSON) with per-stage
  latency, retrieval counts, cache hits and token usage.

## Architecture

```
User ─► Input Gate ─► Query Router ─► Retrieval / Web Search ─► Context Merge
        (security)    (LLM classify)   (ChromaDB)   (Tavily)     (security)
                                                                     │
User ◄─ Response Gate ◄─ Citation Builder ◄─ Answer Generation ◄─────┘
        (security)                            (LLM)
```

The workflow is a LangGraph state machine (`app/graph/`); every node is a
small, dependency-injected unit. `create_agent()` in `app/agents/agent.py`
is the single composition root. Full details in
[docs/architecture.md](docs/architecture.md).

## Quick start

Requires Python 3.9+ (3.11+ recommended).

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set OPENAI_API_KEY and TAVILY_API_KEY

# 1. Index your documents (put files under documents/ first)
python -m app.cli ingest

# 2. Start the API
uvicorn app.api.main:app --port 8000

# 3. Start the UI (new terminal)
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501 for the chat UI, http://localhost:8000/docs for
the API reference.

## Configuration

Everything is configurable through environment variables (or `.env`); nested
settings use `__` as the delimiter. Highlights (see `.env.example` for the
full list):

| Variable | Default | Purpose |
|---|---|---|
| `LLM__MODEL` | `gpt-4o` | answer-generation model |
| `LLM__ROUTER_MODEL` | `gpt-4o-mini` | query-classification model |
| `EMBEDDING__PROVIDER` | `openai` | `openai` \| `voyage` \| `cohere` \| `sentence_transformers` |
| `CHUNKING__CHUNK_SIZE` / `__CHUNK_OVERLAP` | 1000 / 200 | ingestion splitting |
| `RETRIEVAL__TOP_K` | 5 | chunks passed to the LLM |
| `RETRIEVAL__RERANK_ENABLED` | `true` | cross-encoder reranking |
| `WEB_SEARCH__CACHE_TTL_SECONDS` | 86400 | web-content cache TTL |
| `VECTOR_STORE__PERSIST_DIRECTORY` | `data/chroma` | ChromaDB location |
| `API__AUTH_TOKEN` | *(unset)* | set to require bearer auth |
| `LOGGING__JSON_FORMAT` | `false` | JSON logs for aggregators |

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Answer a question (full secured workflow) |
| POST | `/search` | Semantic search only (security-filtered) |
| POST | `/upload` | Upload + index one document |
| POST | `/reindex` | Rebuild or incrementally update the index |
| GET | `/health` | Liveness + vector-store statistics |
| GET/DELETE | `/cache` | Web-cache stats / clear |
| GET | `/docs` | Interactive OpenAPI documentation |

## CLI

```bash
python -m app.cli ingest            # incremental index of documents/
python -m app.cli rebuild           # drop and rebuild the index
python -m app.cli stats             # vector-store statistics (JSON)
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest                              # 156 tests, fully offline
```

Unit tests cover every module with fakes for externals (LLMs, web search);
integration tests run the real LangGraph workflow against a real ChromaDB
with scripted LLMs, plus the FastAPI endpoints via `TestClient`.

## Project structure

```
app/
├── agents/       # HybridRAGAgent facade + composition root
├── api/          # FastAPI (schemas, routes, auth, app factory)
├── cache/        # disk-backed TTL cache
├── chains/       # prompts, answer chain, citations, LLM factory
├── config/       # Pydantic settings tree
├── database/     # ChromaDB manager + indexing service
├── embeddings/   # provider factory (OpenAI/Voyage/Cohere/local)
├── graph/        # LangGraph state, nodes, workflow
├── loaders/      # 9 format loaders, chunking, ingestion
├── retrievers/   # retrieval funnel (similarity/MMR/rerank/compress)
├── routers/      # LLM query router
├── security/     # the five security layers
└── utils/        # logging, timing, exceptions
frontend/         # Streamlit UI (talks to the API over HTTP)
tests/            # unit + integration suites
docs/             # architecture and security documentation
legacy/           # the original Pinecone prototype (superseded)
```
