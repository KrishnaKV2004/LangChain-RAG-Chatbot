"""Shared utilities: structured logging, timing helpers, and exceptions."""

from app.utils.exceptions import (
    CacheError,
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    IngestionError,
    LLMError,
    RAGChatbotError,
    RetrievalError,
    SecurityViolationError,
    VectorStoreError,
    WebSearchError,
)
from app.utils.logging import configure_logging, get_logger
from app.utils.timing import Timer

__all__ = [
    "CacheError",
    "ConfigurationError",
    "DocumentLoadError",
    "EmbeddingError",
    "IngestionError",
    "LLMError",
    "RAGChatbotError",
    "RetrievalError",
    "SecurityViolationError",
    "VectorStoreError",
    "WebSearchError",
    "configure_logging",
    "get_logger",
    "Timer",
]
