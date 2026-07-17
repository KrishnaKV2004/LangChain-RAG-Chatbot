"""HybridRAGAgent — the facade the API and UI consume.

``create_agent()`` is the **composition root**: the only place in the
codebase where concrete implementations are chosen and wired together.
Everything else (nodes, chains, services) receives its collaborators via
constructor injection, which is what keeps the whole system testable with
fakes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.cache.ttl_cache import TTLCache
from app.chains.answer import AnswerChain
from app.chains.citations import CitationBuilder, Citations
from app.chains.llm_factory import create_chat_model
from app.chains.prompts import PROTECTED_MARKERS
from app.config.settings import Settings
from app.database.chroma import ChromaManager
from app.database.indexer import IndexingService
from app.embeddings.factory import create_embeddings
from app.graph.nodes import GraphNodes
from app.graph.workflow import build_workflow
from app.loaders.ingestion import IngestionService
from app.retrievers.compression import SentenceCompressor
from app.retrievers.pipeline import RetrievalPipeline
from app.retrievers.reranker import CrossEncoderReranker
from app.routers.classifier import QueryRouter
from app.search.factory import create_search_provider
from app.search.service import WebSearchService
from app.security.guard import SecurityGuard
from app.security.models import UserContext
from app.utils.logging import configure_logging, get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Everything one chat turn produces, ready for JSON serialization."""

    answer: str
    route: str = "UNKNOWN"
    blocked: bool = False
    citations: Dict[str, Any] = field(default_factory=dict)
    #: Post-filter retrieved chunks (text preview + provenance) for the UI.
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    web_cache_hit: bool = False
    total_latency_ms: float = 0.0


class HybridRAGAgent:
    """High-level agent: one :meth:`chat` call runs the whole workflow."""

    def __init__(
        self,
        workflow: Any,
        manager: ChromaManager,
        indexer: IndexingService,
        web_cache: TTLCache,
        retrieval: Optional[RetrievalPipeline] = None,
        guard: Optional[SecurityGuard] = None,
    ) -> None:
        self._workflow = workflow
        # Exposed for the API layer (/upload, /reindex, /health, /cache,
        # /search — the latter two need direct retrieval + security access).
        self.manager = manager
        self.indexer = indexer
        self.web_cache = web_cache
        self.retrieval = retrieval
        self.guard = guard

    def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        user: Optional[UserContext] = None,
        force_route: Optional[str] = None,
    ) -> AgentResponse:
        """Answer one question through the full secured pipeline.

        ``force_route`` (a ``Route`` value) bypasses the LLM classifier —
        used by the UI's search-mode selector.
        """
        initial: Dict[str, Any] = {
            "query": query,
            "history": history or [],
            "user": user or UserContext(),
        }
        if force_route:
            initial["route"] = force_route
        with Timer() as timer:
            state = self._workflow.invoke(initial)

        citations: Citations = state.get("citations") or Citations()
        response = AgentResponse(
            answer=state.get("answer", ""),
            route=state.get("route", "UNKNOWN"),
            blocked=bool(state.get("blocked")),
            citations=citations.as_dict(),
            retrieved_chunks=[
                {
                    "text": doc.page_content[:500],
                    "filename": doc.metadata.get("filename"),
                    "page": doc.metadata.get("page"),
                    "section_title": doc.metadata.get("section_title"),
                }
                for doc in state.get("internal_docs", [])
            ],
            metrics=state.get("metrics", {}),
            token_usage=state.get("token_usage", {}),
            web_cache_hit=bool(state.get("web_cache_hit")),
            total_latency_ms=round(timer.elapsed_ms, 1),
        )
        logger.info(
            "chat_complete",
            route=response.route,
            blocked=response.blocked,
            latency_ms=response.total_latency_ms,
        )
        return response


def create_agent(settings: Settings) -> HybridRAGAgent:
    """Composition root: build and wire every component from configuration."""
    configure_logging(settings)
    settings.ensure_directories()

    # ---- Storage & retrieval ------------------------------------------- #
    embeddings = create_embeddings(settings)
    manager = ChromaManager(settings, embeddings)
    reranker = (
        CrossEncoderReranker(settings.retrieval.rerank_model)
        if settings.retrieval.rerank_enabled
        else None
    )
    compressor = (
        SentenceCompressor(embeddings)
        if settings.retrieval.compression_enabled
        else None
    )
    retrieval = RetrievalPipeline(manager, settings.retrieval, reranker, compressor)

    # ---- Web search ------------------------------------------------------ #
    web_cache = TTLCache(settings.cache.path, settings.web_search.cache_ttl_seconds)
    web_search = WebSearchService(
        create_search_provider(settings), web_cache, settings.web_search
    )

    # ---- Security --------------------------------------------------------- #
    guard = SecurityGuard(settings.security, protected_markers=PROTECTED_MARKERS)

    # ---- LLM chains --------------------------------------------------------- #
    router = QueryRouter(create_chat_model(settings, role="router"))
    answer_chain = AnswerChain(create_chat_model(settings, role="answer"))

    # ---- Ingestion (Layer-2 tagging happens before chunks are stored) ------ #
    ingestion = IngestionService(settings.chunking)
    indexer = IndexingService(
        ingestion,
        manager,
        settings.paths.documents_dir,
        chunk_processor=guard.tag_sensitivity,
    )

    # ---- Workflow ------------------------------------------------------------ #
    nodes = GraphNodes(router, retrieval, web_search, guard, answer_chain, CitationBuilder())
    workflow = build_workflow(nodes)

    logger.info("agent_ready", environment=settings.environment)
    return HybridRAGAgent(workflow, manager, indexer, web_cache, retrieval, guard)
