"""FastAPI application factory.

Run with::

    uvicorn app.api.main:app --host 0.0.0.0 --port 8000

The agent is built during startup (lifespan), not at import time, so tooling
can import this module without needing API keys. Tests inject a stub agent
via ``create_app(agent=...)``.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.agent import HybridRAGAgent, create_agent
from app.api.routes import router
from app.config.settings import Settings, get_settings
from app.utils.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    RAGChatbotError,
    RateProviderError,
    SecurityViolationError,
    WebSearchError,
)
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Exception class → HTTP status. Everything else in the hierarchy is a 500.
_STATUS_MAP = [
    (SecurityViolationError, 403),
    (DocumentLoadError, 400),
    (WebSearchError, 502),
    (RateProviderError, 502),
    (ConfigurationError, 500),
]


def create_app(
    settings: Optional[Settings] = None,
    agent: Optional[HybridRAGAgent] = None,
) -> FastAPI:
    """Build the API. ``agent`` is injectable for tests."""
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Build the real agent at startup unless a test injected one.
        if application.state.agent is None:
            application.state.agent = create_agent(settings)
        logger.info("api_started", port=settings.api.port)
        yield
        logger.info("api_stopped")

    application = FastAPI(
        title="Hybrid RAG AI Agent",
        description=(
            "Enterprise RAG assistant combining internal documents, web search "
            "and LLM reasoning behind a five-layer security stack."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.agent = agent

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(RAGChatbotError)
    async def domain_error_handler(request: Request, exc: RAGChatbotError) -> JSONResponse:
        """Map domain exceptions to clean JSON errors; details stay in logs."""
        status_code = next(
            (code for cls, code in _STATUS_MAP if isinstance(exc, cls)), 500
        )
        logger.error(
            "request_failed",
            path=request.url.path,
            error_type=type(exc).__name__,
            error=exc.message,
            details=exc.details,
        )
        return JSONResponse(status_code=status_code, content={"error": exc.message})

    application.include_router(router)
    return application


#: Module-level app for `uvicorn app.api.main:app`.
app = create_app()
