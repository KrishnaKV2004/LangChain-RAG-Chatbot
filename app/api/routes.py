"""REST endpoint handlers.

Thin layer: validate input (schemas), call the agent facade, shape the
response. No business logic lives here.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

import app as app_package
from app.agents.agent import HybridRAGAgent
from app.api.dependencies import get_agent, get_settings_dep, require_auth
from app.api.schemas import (
    CacheClearResponse,
    CacheStatsResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexReportResponse,
    RateLocationRequest,
    RateRequest,
    RateQuoteResponse,
    RatesResponse,
    ReindexRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.config.settings import Settings
from app.database.indexer import IndexReport
from app.loaders.registry import default_registry
from app.rates.models import FreightItem, Location, RateMode, RateQuery
from app.security.models import UserContext
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_auth)])

#: Upload size cap — a corrupt/malicious oversized file must not exhaust disk.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _report_response(report: IndexReport) -> IndexReportResponse:
    return IndexReportResponse(
        files_processed=report.files_processed,
        files_skipped=report.files_skipped,
        errors=report.errors,
        chunks_indexed=report.chunks_indexed,
        duration_ms=round(report.duration_ms, 1),
        rebuilt=report.rebuilt,
    )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: HybridRAGAgent = Depends(get_agent)) -> ChatResponse:
    """Answer a question through the full secured RAG workflow."""
    response = agent.chat(
        query=request.query,
        history=[turn.model_dump() for turn in request.history],
        user=UserContext(user_id=request.user_id, role=request.role),
        force_route=request.force_route,
    )
    return ChatResponse(**response.__dict__)


# --------------------------------------------------------------------------- #
# Semantic search (retrieval only, still security-filtered)
# --------------------------------------------------------------------------- #


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, agent: HybridRAGAgent = Depends(get_agent)) -> SearchResponse:
    """Search internal documents without generating an answer."""
    if agent.retrieval is None or agent.guard is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Search unavailable")

    verdict = agent.guard.validate_query(request.query)
    if not verdict.allowed:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Query rejected by security policy")

    result = agent.retrieval.retrieve(request.query, top_k=request.top_k)
    user = UserContext(user_id=request.user_id, role=request.role)

    # The same clearance rules apply to raw search as to chat.
    score_by_id = {
        chunk.document.metadata.get("chunk_id"): chunk for chunk in result.chunks
    }
    decision = agent.guard.filter_context(user, result.documents)

    hits = []
    for document in decision.documents:
        chunk = score_by_id.get(document.metadata.get("chunk_id"))
        hits.append(
            SearchHit(
                text=document.page_content,
                filename=document.metadata.get("filename"),
                page=document.metadata.get("page"),
                section_title=document.metadata.get("section_title"),
                similarity_score=chunk.similarity_score if chunk else None,
                rerank_score=chunk.rerank_score if chunk else None,
            )
        )
    return SearchResponse(results=hits, latency_ms=round(result.latency_ms, 1))


# --------------------------------------------------------------------------- #
# Freight rate quotes (7LFreight air / LTL / ocean)
# --------------------------------------------------------------------------- #


def _to_location(request: RateLocationRequest) -> Location:
    return Location(
        airport=request.airport,
        port=request.port,
        city=request.city,
        state=request.state,
        zipcode=request.zipcode,
        country=request.country,
        address1=request.address1,
        address2=request.address2,
    )


@router.post("/rates", response_model=RatesResponse)
def rates(request: RateRequest, agent: HybridRAGAgent = Depends(get_agent)) -> RatesResponse:
    """Get live carrier rate quotes for a structured shipment request.

    This is the deterministic counterpart to the chatbot's RATES route: the
    caller supplies an already-structured query (no NL extraction). Provider
    failures surface as 502 via the domain exception handler.
    """
    if agent.rates is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Rate quoting is not configured (missing 7LFreight credentials)",
        )
    query = RateQuery(
        mode=RateMode(request.mode),
        origin=_to_location(request.origin),
        destination=_to_location(request.destination),
        items=[
            FreightItem(
                weight=item.weight,
                qty=item.qty,
                weight_type=item.weight_type,
                length=item.length,
                width=item.width,
                height=item.height,
                dim_type=item.dim_type,
                commodity=item.commodity,
                freight_class=item.freight_class,
                hazmat=item.hazmat,
                stack=item.stack,
            )
            for item in request.items
        ],
        uom=request.uom,
        pickup_date=request.pickup_date,
        hazardous=request.hazardous,
    )
    outcome = agent.rates.get_rates(query)  # RateProviderError → 502 (handler)
    return RatesResponse(
        quotes=[
            RateQuoteResponse(
                **{key: value for key, value in quote.to_dict().items() if key != "breakdown"}
            )
            for quote in outcome.quotes
        ],
        cache_hit=outcome.cache_hit,
        latency_ms=round(outcome.latency_ms, 1),
    )


# --------------------------------------------------------------------------- #
# Document management
# --------------------------------------------------------------------------- #


@router.post("/upload", response_model=IndexReportResponse)
async def upload(
    file: UploadFile = File(...),
    agent: HybridRAGAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings_dep),
) -> IndexReportResponse:
    """Upload one document into the knowledge base and index it."""
    filename = Path(file.filename or "").name  # strip any path components
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")

    suffix = Path(filename).suffix.lower()
    if suffix not in default_registry.supported_suffixes():
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(default_registry.supported_suffixes())}",
        )

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    destination = Path(settings.paths.documents_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    logger.info("file_uploaded", file=filename, bytes=len(content))

    return _report_response(agent.indexer.index_file(destination))


@router.post("/reindex", response_model=IndexReportResponse)
def reindex(
    request: ReindexRequest, agent: HybridRAGAgent = Depends(get_agent)
) -> IndexReportResponse:
    """Re-index the documents directory (full rebuild by default)."""
    report = agent.indexer.rebuild() if request.rebuild else agent.indexer.index_directory()
    return _report_response(report)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthResponse)
def health(
    agent: HybridRAGAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    """Liveness + vector-store statistics (powers the UI sidebar)."""
    try:
        stats = agent.manager.stats()
        status_value = "ok"
    except Exception as exc:  # noqa: BLE001 — health must not 500
        logger.error("health_check_degraded", error=str(exc))
        stats = {"error": "vector store unavailable"}
        status_value = "degraded"
    return HealthResponse(
        status=status_value,
        version=app_package.__version__,
        environment=settings.environment,
        vector_store=stats,
    )


@router.get("/cache", response_model=CacheStatsResponse)
def cache_stats(agent: HybridRAGAgent = Depends(get_agent)) -> CacheStatsResponse:
    """Web-cache statistics."""
    return CacheStatsResponse(**agent.web_cache.stats())


@router.delete("/cache", response_model=CacheClearResponse)
def cache_clear(
    expired_only: bool = False, agent: HybridRAGAgent = Depends(get_agent)
) -> CacheClearResponse:
    """Clear the web cache (``?expired_only=true`` purges only stale entries)."""
    removed = (
        agent.web_cache.purge_expired() if expired_only else agent.web_cache.clear()
    )
    return CacheClearResponse(removed=removed, expired_only=expired_only)
