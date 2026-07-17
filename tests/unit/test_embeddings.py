"""Unit tests for the embedding factory."""

import pytest

from app.config.settings import Settings
from app.embeddings.factory import create_embeddings
from app.utils.exceptions import ConfigurationError


class TestEmbeddingFactory:
    def test_openai_without_key_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(_env_file=None)
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            create_embeddings(settings)

    def test_openai_builds_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        embeddings = create_embeddings(settings)
        # The provider default model must be applied.
        assert embeddings.model == "text-embedding-3-large"

    def test_voyage_without_package_gives_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDING__PROVIDER", "voyage")
        settings = Settings(_env_file=None)
        # langchain-voyageai is an optional extra; either outcome is a clean
        # ConfigurationError (missing package or missing key), never a crash.
        with pytest.raises(ConfigurationError):
            create_embeddings(settings)
