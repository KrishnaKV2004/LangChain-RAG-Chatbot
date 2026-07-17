"""Pydantic request/response models for the REST API.

Request models validate and bound all client input (lengths, ranges, enums)
before any business logic runs; response models define the exact JSON shape
so the OpenAPI docs at ``/docs`` stay accurate.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.security.models import Role

# --------------------------------------------------------------------------- #
# /chat
# --------------------------------------------------------------------------- #


class ChatTurn(BaseModel):
    """One prior message in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=50)
    #: In production the role should come from the auth layer; the field
    #: exists so trusted frontends can pass the authenticated user's role.
    role: Role = Role.EMPLOYEE
    user_id: str = Field(default="anonymous", max_length=128)
    #: Optional route override (UI "search mode"): skips the LLM classifier.
    force_route: Optional[
        Literal["INTERNAL_ONLY", "WEB_ONLY", "HYBRID", "GENERAL_CHAT"]
    ] = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    blocked: bool
    citations: Dict[str, object]
    retrieved_chunks: List[Dict[str, Optional[object]]]
    metrics: Dict[str, float]
    token_usage: Dict[str, int]
    web_cache_hit: bool
    total_latency_ms: float


# --------------------------------------------------------------------------- #
# /search
# --------------------------------------------------------------------------- #


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=20)
    role: Role = Role.EMPLOYEE
    user_id: str = Field(default="anonymous", max_length=128)


class SearchHit(BaseModel):
    text: str
    filename: Optional[str] = None
    page: Optional[int] = None
    section_title: Optional[str] = None
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchHit]
    latency_ms: float


# --------------------------------------------------------------------------- #
# /upload and /reindex
# --------------------------------------------------------------------------- #


class IndexReportResponse(BaseModel):
    files_processed: List[str]
    files_skipped: List[str]
    errors: Dict[str, str]
    chunks_indexed: int
    duration_ms: float
    rebuilt: bool


class ReindexRequest(BaseModel):
    #: ``true`` drops the collection first (removes deleted files from the
    #: index); ``false`` is an incremental upsert.
    rebuild: bool = True


# --------------------------------------------------------------------------- #
# /health and /cache
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    vector_store: Dict[str, object]


class CacheStatsResponse(BaseModel):
    path: str
    entries: int
    expired_entries: int
    size_bytes: int
    default_ttl_seconds: float


class CacheClearResponse(BaseModel):
    removed: int
    expired_only: bool


class ErrorResponse(BaseModel):
    error: str
