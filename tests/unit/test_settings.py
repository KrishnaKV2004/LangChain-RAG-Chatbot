"""Unit tests for the configuration system (app.config.settings)."""

import pytest
from pydantic import ValidationError

from app.config.settings import (
    EmbeddingProvider,
    EmbeddingSettings,
    ChunkingSettings,
    Settings,
)


class TestDefaults:
    """The system must boot with sane defaults and no environment at all."""

    def test_default_settings_load(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.llm.model == "gpt-4o"
        assert settings.retrieval.top_k == 5
        assert settings.web_search.cache_ttl_seconds == 86_400
        assert settings.security.prompt_injection_detection is True

    def test_embedding_model_falls_back_to_provider_default(self) -> None:
        emb = EmbeddingSettings(provider=EmbeddingProvider.VOYAGE)
        assert emb.resolved_model == "voyage-3"

    def test_embedding_model_explicit_override_wins(self) -> None:
        emb = EmbeddingSettings(provider=EmbeddingProvider.OPENAI, model="text-embedding-3-small")
        assert emb.resolved_model == "text-embedding-3-small"


class TestEnvironmentOverrides:
    """Nested settings must be overridable via `GROUP__FIELD` env vars."""

    def test_nested_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM__TEMPERATURE", "0.7")
        monkeypatch.setenv("RETRIEVAL__TOP_K", "9")
        settings = Settings(_env_file=None)
        assert settings.llm.temperature == 0.7
        assert settings.retrieval.top_k == 9

    def test_api_keys_are_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
        settings = Settings(_env_file=None)
        # SecretStr must not expose the key in repr/str (log-leak protection).
        assert "sk-super-secret" not in repr(settings)
        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-super-secret"


class TestValidation:
    """Invalid configuration must fail fast at startup, not at request time."""

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValidationError):
            ChunkingSettings(chunk_size=100, chunk_overlap=200)

    def test_invalid_search_type_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVAL__SEARCH_TYPE", "quantum")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_invalid_provider_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING__PROVIDER", "not-a-provider")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestLocalLLMEndpoint:
    """LLM__BASE_URL enables OpenAI-compatible servers without an API key."""

    def test_base_url_without_key_builds_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.chains.llm_factory import create_chat_model

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM__BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("LLM__MODEL", "llama3.1")
        model = create_chat_model(Settings(_env_file=None))
        assert model.model_name == "llama3.1"

    def test_no_key_and_no_base_url_still_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.chains.llm_factory import create_chat_model
        from app.utils.exceptions import ConfigurationError

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            create_chat_model(Settings(_env_file=None))
