"""Chat-model factory.

Two logical roles are built from configuration:

* ``answer`` — the main generation model (``LLM__MODEL``),
* ``router`` — a cheaper/faster model for query classification
  (``LLM__ROUTER_MODEL``), always run at temperature 0 because routing is a
  deterministic classification task.

Like the embedding factory, provider SDKs are imported lazily and missing
keys fail fast with actionable messages.
"""

from langchain_core.language_models import BaseChatModel

from app.config.settings import LLMProvider, Settings
from app.utils.exceptions import ConfigurationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def create_chat_model(settings: Settings, role: str = "answer") -> BaseChatModel:
    """Build the chat model for ``role`` ("answer" or "router")."""
    is_router = role == "router"
    model_name = settings.llm.router_model if is_router else settings.llm.model
    temperature = 0.0 if is_router else settings.llm.temperature

    if settings.llm.provider is LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        base_url = settings.llm.base_url
        if settings.openai_api_key is not None:
            api_key = settings.openai_api_key.get_secret_value()
        elif base_url:
            # Local OpenAI-compatible servers (Ollama, LM Studio) accept any
            # placeholder key; hosted ones (Groq, OpenRouter) need a real one.
            api_key = "not-needed"
        else:
            raise ConfigurationError(
                "OPENAI_API_KEY is required for LLM__PROVIDER=openai "
                "(or set LLM__BASE_URL to use a local OpenAI-compatible server)."
            )
        model: BaseChatModel = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout_seconds,
            api_key=api_key,
            base_url=base_url,
        )
    elif settings.llm.provider is LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ConfigurationError(
                "LLM__PROVIDER=anthropic requires: pip install langchain-anthropic"
            ) from exc

        if settings.anthropic_api_key is None:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is required for LLM__PROVIDER=anthropic."
            )
        model = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=settings.llm.timeout_seconds,
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
    else:  # unreachable while the enum stays in sync
        raise ConfigurationError(f"Unknown LLM provider: {settings.llm.provider}")

    logger.info("chat_model_initialized", role=role, model=model_name)
    return model
