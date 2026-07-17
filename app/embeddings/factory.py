"""Embedding-provider factory.

The rest of the codebase depends only on the abstract
``langchain_core.embeddings.Embeddings`` interface; this factory is the single
place that knows which concrete provider to build.  Switching providers is a
pure configuration change::

    EMBEDDING__PROVIDER=sentence_transformers

Provider SDKs are imported lazily so only the configured one needs to be
installed.  Missing API keys fail fast at startup with an actionable message
instead of failing on the first user query.
"""

from typing import Callable, Dict

from langchain_core.embeddings import Embeddings

from app.config.settings import EmbeddingProvider, Settings
from app.utils.exceptions import ConfigurationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _require_key(settings: Settings, attribute: str, env_name: str) -> str:
    """Fetch a SecretStr API key or raise a configuration error."""
    secret = getattr(settings, attribute)
    if secret is None:
        raise ConfigurationError(
            f"{env_name} is required for the configured embedding provider. "
            f"Set it in your environment or .env file."
        )
    return secret.get_secret_value()


def _build_openai(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding.resolved_model,
        api_key=_require_key(settings, "openai_api_key", "OPENAI_API_KEY"),
        chunk_size=settings.embedding.batch_size,
    )


def _build_voyage(settings: Settings) -> Embeddings:
    try:
        from langchain_voyageai import VoyageAIEmbeddings
    except ImportError as exc:
        raise ConfigurationError(
            "EMBEDDING__PROVIDER=voyage requires: pip install langchain-voyageai"
        ) from exc

    return VoyageAIEmbeddings(
        model=settings.embedding.resolved_model,
        api_key=_require_key(settings, "voyage_api_key", "VOYAGE_API_KEY"),
        batch_size=settings.embedding.batch_size,
    )


def _build_cohere(settings: Settings) -> Embeddings:
    try:
        from langchain_cohere import CohereEmbeddings
    except ImportError as exc:
        raise ConfigurationError(
            "EMBEDDING__PROVIDER=cohere requires: pip install langchain-cohere"
        ) from exc

    return CohereEmbeddings(
        model=settings.embedding.resolved_model,
        cohere_api_key=_require_key(settings, "cohere_api_key", "COHERE_API_KEY"),
    )


def _build_sentence_transformers(settings: Settings) -> Embeddings:
    from app.embeddings.local import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(
        model_name=settings.embedding.resolved_model,
        batch_size=settings.embedding.batch_size,
    )


_BUILDERS: Dict[EmbeddingProvider, Callable[[Settings], Embeddings]] = {
    EmbeddingProvider.OPENAI: _build_openai,
    EmbeddingProvider.VOYAGE: _build_voyage,
    EmbeddingProvider.COHERE: _build_cohere,
    EmbeddingProvider.SENTENCE_TRANSFORMERS: _build_sentence_transformers,
}


def create_embeddings(settings: Settings) -> Embeddings:
    """Build the embedding backend selected by ``settings.embedding.provider``."""
    provider = settings.embedding.provider
    builder = _BUILDERS.get(provider)
    if builder is None:  # unreachable while the enum and dict stay in sync
        raise ConfigurationError(f"Unknown embedding provider: {provider}")

    embeddings = builder(settings)
    logger.info(
        "embeddings_initialized",
        provider=provider.value,
        model=settings.embedding.resolved_model,
    )
    return embeddings
