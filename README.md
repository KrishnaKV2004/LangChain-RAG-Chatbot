# Hybrid RAG AI Agent

An enterprise-grade Retrieval-Augmented Generation assistant for logistics
and general-knowledge questions. It combines **internal documents**
(ChromaDB), **live web search** (Tavily) and **LLM reasoning** into one
answer — behind a five-layer security stack that prevents leakage of
confidential company information.

Built with **LangChain · LangGraph · FastAPI · Streamlit · ChromaDB**.

## Features

- **Query routing** — an LLM classifier sends each question down the right
  path: `INTERNAL_ONLY`, `WEB_ONLY`, `HYBRID`, `GENERAL_CHAT`, or `RATES`.
- **Live freight rates** — the `RATES` route (and `POST /rates`) extracts an
  origin/destination/weight from natural language and fetches real carrier
  quotes from the 7LFreight API (air, LTL and LCL ocean), behind a JWT that is
  cached to respect the provider's daily login quota.
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
| POST | `/rates` | Live freight quotes for a structured shipment (7LFreight; 503 when unconfigured) |
| POST | `/upload` | Upload + index one document |
| POST | `/reindex` | Rebuild or incrementally update the index |
| GET | `/health` | Liveness + vector-store statistics |
| GET/DELETE | `/cache` | Web-cache stats / clear |
| GET | `/docs` | Interactive OpenAPI documentation |

## Deployment

```bash
cp .env.example .env          # fill in keys, and SET API__AUTH_TOKEN
docker compose up --build     # api :8000, ui :8501
```

Then index the documents once (the index is not built at boot):

```bash
docker compose run --rm api python -m app.cli rebuild
```

One image serves both services; the UI just overrides the command. What the
compose file gets right, and why each matters:

| Concern | Handling |
|---|---|
| **`data/cache` volume** | Holds the 7LFreight JWT. Losing it burns one of the **daily** login quota on every restart. |
| **`data/chroma` volume** | The vector index — expensive to rebuild. |
| **Health-check grace** | `start_period=180s`: startup loads the embedding + reranker models. Without it the container is killed mid-boot. |
| **Auth on `/health`** | Every route is behind `require_auth`, so the health probe sends the bearer token when one is set. |
| **Model pre-baking** | Models download at *build* time, so start-up is fast and runtime doesn't depend on HuggingFace. |
| **`.dockerignore`** | Keeps the 1.2 GB local `env/` out of the build context (0.4 MB context). |

Two things to set on whatever sits in front of the API:

- **Read timeout > 60s.** A door-to-door quote makes three live carrier calls
  (~20–40s). A default 30s proxy timeout will cut quotes off mid-flight.
- **`API__CORS_ORIGINS`** must list the real UI origin, not `localhost:8501`.

Slim image (no torch, ~600 MB instead of ~3.5 GB) if you use hosted embeddings:
set `EMBEDDING__PROVIDER=openai` and `RETRIEVAL__RERANK_ENABLED=false`, then
build with `--build-arg PREFETCH_MODELS=false`.

## CLI

```bash
python -m app.cli ingest            # incremental index of documents/
python -m app.cli rebuild           # drop and rebuild the index
python -m app.cli stats             # vector-store statistics (JSON)
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest                              # 219 tests, fully offline
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
├── rates/        # 7LFreight rate provider (air/LTL/ocean) + service
├── retrievers/   # retrieval funnel (similarity/MMR/rerank/compress)
├── routers/      # LLM query router
├── security/     # the five security layers
└── utils/        # logging, timing, exceptions
frontend/         # Streamlit UI (talks to the API over HTTP)
tests/            # unit + integration suites
docs/             # architecture and security documentation
legacy/           # the original Pinecone prototype (superseded)
```
