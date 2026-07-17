"""Configuration package.

Exposes the application-wide :func:`get_settings` accessor so other modules
never instantiate settings objects directly (dependency injection friendly).
"""

from app.config.settings import (
    EmbeddingProvider,
    LLMProvider,
    SearchProvider,
    Settings,
    get_settings,
    reload_settings,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "SearchProvider",
    "Settings",
    "get_settings",
    "reload_settings",
]
