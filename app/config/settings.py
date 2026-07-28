"""Central application settings.

Every tunable of the Hybrid RAG system is defined here and can be overridden
through environment variables (or a ``.env`` file).  Nested groups use a
double-underscore delimiter, e.g.::

    LLM__MODEL=gpt-4o
    RETRIEVAL__TOP_K=8
    WEB_SEARCH__CACHE_TTL_SECONDS=3600

Secrets (API keys) are read from their conventional top-level names
(``OPENAI_API_KEY``, ``TAVILY_API_KEY``, ...) and wrapped in ``SecretStr`` so
they never leak into logs or reprs.

Usage::

    from app.config import get_settings

    settings = get_settings()
    print(settings.retrieval.top_k)
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Enumerations — constrain provider choices so typos fail fast at startup.
# --------------------------------------------------------------------------- #


class LLMProvider(str, Enum):
    """Supported chat-model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    """Supported embedding backends (switchable via ``EMBEDDING__PROVIDER``)."""

    OPENAI = "openai"
    VOYAGE = "voyage"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class SearchProvider(str, Enum):
    """Supported web-search providers. Tavily is the default; the rest are
    registered through the same interface (see ``app.search.base``)."""

    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"
    BRAVE = "brave"
    GOOGLE = "google"


# Sensible per-provider defaults so users only set a model when they want to
# deviate from the recommended one.
_DEFAULT_EMBEDDING_MODELS = {
    EmbeddingProvider.OPENAI: "text-embedding-3-large",
    EmbeddingProvider.VOYAGE: "voyage-3",
    EmbeddingProvider.COHERE: "embed-english-v3.0",
    EmbeddingProvider.SENTENCE_TRANSFORMERS: "sentence-transformers/all-MiniLM-L6-v2",
}


# --------------------------------------------------------------------------- #
# Nested settings groups
# --------------------------------------------------------------------------- #


class LLMSettings(BaseModel):
    """Chat-model configuration (answer generation + query routing)."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    # A cheaper/faster model is used for query classification because routing
    # is a short, low-stakes structured-output task.
    router_model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)
    # Any OpenAI-compatible endpoint (Ollama: http://localhost:11434/v1,
    # Groq: https://api.groq.com/openai/v1, LM Studio, OpenRouter, vLLM...).
    # When set, OPENAI_API_KEY becomes optional (local servers ignore it).
    base_url: Optional[str] = None


class EmbeddingSettings(BaseModel):
    """Embedding-model configuration."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    # ``None`` means "use the provider's recommended default model".
    model: Optional[str] = None
    batch_size: int = Field(default=64, gt=0)

    @property
    def resolved_model(self) -> str:
        """The model to actually load: explicit override or provider default."""
        return self.model or _DEFAULT_EMBEDDING_MODELS[self.provider]


class ChunkingSettings(BaseModel):
    """Document splitting parameters used during ingestion."""

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    # Chunks shorter than this (e.g. page headers) are dropped as noise.
    min_chunk_chars: int = Field(default=50, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


class RetrievalSettings(BaseModel):
    """Vector-retrieval behaviour."""

    # "similarity" or "mmr" — MMR trades a little relevance for diversity.
    search_type: str = Field(default="mmr", pattern="^(similarity|mmr)$")
    top_k: int = Field(default=5, gt=0)
    # Candidates fetched before MMR/reranking narrows them down.
    fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    # Similarity floor (cosine relevance 0..1); chunks below it are discarded.
    score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    # Cross-encoder reranking (Layer 2 of relevance filtering).
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = Field(default=5, gt=0)
    # LLM-free contextual compression (trim irrelevant sentences).
    compression_enabled: bool = False


class VectorStoreSettings(BaseModel):
    """ChromaDB persistence configuration."""

    persist_directory: Path = Path("data/chroma")
    documents_collection: str = "documents"
    cache_collection: str = "web_cache"
    # Cosine distance is the right metric for normalized text embeddings.
    distance_metric: str = Field(default="cosine", pattern="^(cosine|l2|ip)$")


class WebSearchSettings(BaseModel):
    """External web-search configuration."""

    provider: SearchProvider = SearchProvider.TAVILY
    max_results: int = Field(default=5, gt=0, le=20)
    timeout_seconds: int = Field(default=20, gt=0)
    # Webpages are cached (not permanently stored) for this long.
    cache_ttl_seconds: int = Field(default=86_400, gt=0)  # 24 h default
    # "basic" is faster/cheaper, "advanced" extracts more page content.
    search_depth: str = Field(default="advanced", pattern="^(basic|advanced)$")


class CacheSettings(BaseModel):
    """On-disk TTL cache used for web results and other transient data."""

    path: Path = Path("data/cache")
    default_ttl_seconds: int = Field(default=86_400, gt=0)


class RatesSettings(BaseModel):
    """7LFreight rate-quote provider configuration.

    7LFreight authenticates via username/password (see ``Settings.seven_l_*``)
    to obtain a short-lived JWT — not a static API key like Tavily.
    """

    enabled: bool = True
    base_url: str = "https://restapi.my7l.com"
    # Path prefix under base_url (7L versions its REST API at /api/v1).
    api_prefix: str = "/api/v1"
    timeout_seconds: int = Field(default=20, gt=0)
    # Login is rate-limited to a DAILY quota, so the JWT is cached on disk and
    # reused across restarts; refresh (not re-login) renews it near expiry.
    # Safety margin (seconds) subtracted from the token's expiry before it is
    # considered stale, so a call never races the clock.
    token_expiry_margin_seconds: int = Field(default=120, ge=0)
    # Quotes carry their own ValidFrom/ValidTo; this cache just avoids
    # hammering the API for the same lookup within a short window.
    cache_ttl_seconds: int = Field(default=300, gt=0)  # 5 minutes
    max_carriers_returned: int = Field(default=5, gt=0, le=20)


class HermesSettings(BaseModel):
    """Hermes — the self-refinement agent that reviews and improves draft
    answers (quality, faithfulness, human tone) before they leave the system."""

    enabled: bool = True
    # Review/revise passes per answer. Each pass costs one extra LLM call
    # only when the draft actually needs improvement.
    max_iterations: int = Field(default=2, ge=1, le=3)


class SecuritySettings(BaseModel):
    """Toggles and thresholds for the five security layers."""

    prompt_injection_detection: bool = True   # Layer 1
    sensitive_document_detection: bool = True  # Layer 2
    permission_validation: bool = True         # Layer 3
    response_scanning: bool = True             # Layer 4
    pii_detection: bool = True                 # Layer 5
    # Hard cap on user input length (defends against context stuffing).
    max_query_chars: int = Field(default=4_000, gt=0)
    # Metadata flag that marks a document confidential at ingestion time.
    confidential_metadata_key: str = "confidential"
    # PII findings in responses are redacted rather than blocking the answer.
    redact_pii_in_responses: bool = True


class APISettings(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65_536)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    # Optional static bearer token; when set, all endpoints require it.
    auth_token: Optional[SecretStr] = None


class PathSettings(BaseModel):
    """Filesystem layout."""

    documents_dir: Path = Path("documents")
    data_dir: Path = Path("data")


class LoggingSettings(BaseModel):
    """Structured-logging configuration."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    # JSON logs for production aggregation; pretty console logs for dev.
    json_format: bool = False


# --------------------------------------------------------------------------- #
# Root settings object
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Root of the configuration tree.

    Reads from environment variables and ``.env``; nested fields use ``__``
    as the delimiter (``LLM__TEMPERATURE=0.2``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",  # tolerate unrelated vars in the environment/.env
    )

    environment: str = Field(default="development", pattern="^(development|staging|production)$")

    # ---- API keys (top-level, conventional names) ------------------------- #
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    tavily_api_key: Optional[SecretStr] = None
    voyage_api_key: Optional[SecretStr] = None
    cohere_api_key: Optional[SecretStr] = None
    # 7LFreight uses username/password (not a static key) to obtain a JWT.
    # Field aliases because "7L_USERNAME" is not a legal Python identifier.
    seven_l_username: Optional[str] = Field(default=None, alias="7L_USERNAME")
    seven_l_password: Optional[SecretStr] = Field(default=None, alias="7L_PASSWORD")

    # ---- Nested groups ----------------------------------------------------- #
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    rates: RatesSettings = Field(default_factory=RatesSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    hermes: HermesSettings = Field(default_factory=HermesSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: APISettings = Field(default_factory=APISettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def ensure_directories(self) -> None:
        """Create all data directories the application writes to."""
        for directory in (
            self.paths.data_dir,
            self.paths.documents_dir,
            self.vector_store.persist_directory,
            self.cache.path,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read configuration (used by tests / reindex)."""
    get_settings.cache_clear()
    return get_settings()
