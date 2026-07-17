"""Structured logging built on ``structlog``.

Two output modes, selected via ``LOGGING__JSON_FORMAT``:

* ``json_format=False`` (default) — colorized, human-readable console lines
  for local development.
* ``json_format=True`` — one JSON object per line, ready for ingestion by
  log aggregators (Datadog, CloudWatch, ELK, ...).

Every log call accepts arbitrary key/value pairs, which is how the rest of
the codebase reports latency, retrieval counts, cache hits, etc.::

    log = get_logger(__name__)
    log.info("retrieval_complete", top_k=5, latency_ms=132.4, cache_hit=False)
"""

import logging
import sys
from typing import Optional

import structlog

from app.config.settings import Settings

_CONFIGURED = False


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure stdlib + structlog once for the whole process.

    Safe to call multiple times; only the first call takes effect (so the
    API server, ingestion CLI, and tests can all call it defensively).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if settings is None:
        # Imported lazily to avoid a circular import at module load time.
        from app.config.settings import get_settings

        settings = get_settings()

    level = getattr(logging, settings.logging.level)

    # Route stdlib logging (used by uvicorn, chromadb, httpx...) through the
    # same handler so third-party logs share our format and level.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors = [
        structlog.contextvars.merge_contextvars,  # request-scoped context
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.logging.json_format
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger (configures logging on first use)."""
    configure_logging()
    return structlog.get_logger(name)
