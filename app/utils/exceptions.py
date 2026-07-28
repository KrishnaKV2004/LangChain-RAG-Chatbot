"""Exception hierarchy for the Hybrid RAG system.

All custom exceptions derive from :class:`RAGChatbotError` so callers (the
FastAPI error handler in particular) can catch one base class and map it to a
clean JSON error response instead of leaking stack traces to clients.
"""

from typing import Any, Dict, Optional


class RAGChatbotError(Exception):
    """Base class for every error raised by this application.

    Attributes:
        message: Human-readable description (safe to show to API clients).
        details: Optional structured context for logs — never sent to clients.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(RAGChatbotError):
    """Invalid or missing configuration (e.g. absent API key for a provider)."""


class IngestionError(RAGChatbotError):
    """A document could not be ingested into the vector store."""


class DocumentLoadError(IngestionError):
    """A specific file could not be parsed by its loader."""


class EmbeddingError(RAGChatbotError):
    """The embedding backend failed or returned malformed vectors."""


class VectorStoreError(RAGChatbotError):
    """ChromaDB operation failed (add, query, delete, rebuild)."""


class RetrievalError(RAGChatbotError):
    """The retrieval pipeline failed (search, rerank, compression)."""


class WebSearchError(RAGChatbotError):
    """The web-search provider failed or timed out."""


class RateProviderError(RAGChatbotError):
    """The freight rate-quote provider failed, timed out, or rejected auth."""


class CacheError(RAGChatbotError):
    """The TTL cache could not be read or written."""


class LLMError(RAGChatbotError):
    """The chat model failed to produce a response."""


class SecurityViolationError(RAGChatbotError):
    """A request or response was blocked by one of the security layers.

    Attributes:
        layer: Which defense layer triggered (e.g. ``"prompt_injection"``).
    """

    def __init__(
        self,
        message: str,
        layer: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.layer = layer
