"""Unit tests for the web-search service and Tavily provider (mocked APIs)."""

from pathlib import Path
from typing import List

import pytest

from app.cache.ttl_cache import TTLCache
from app.config.settings import Settings, WebSearchSettings
from app.search.base import BaseSearchProvider, SearchResult
from app.search.service import WebSearchService
from app.search.tavily import TavilySearchProvider
from app.utils.exceptions import WebSearchError


class FakeProvider(BaseSearchProvider):
    """Scripted provider that counts invocations (for cache-hit assertions)."""

    name = "fake"

    def __init__(self, results: List[SearchResult]) -> None:
        self._results = results
        self.calls = 0

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        self.calls += 1
        return list(self._results)


def result(url: str, title: str, score: float, content: str = "text") -> SearchResult:
    return SearchResult(title=title, url=url, content=content, score=score, provider="fake")


@pytest.fixture()
def cache(tmp_path: Path) -> TTLCache:
    return TTLCache(tmp_path / "cache", default_ttl_seconds=60)


def make_service(provider: BaseSearchProvider, cache: TTLCache, **overrides: object) -> WebSearchService:
    settings = WebSearchSettings(**overrides)
    return WebSearchService(provider, cache, settings)


class TestWebSearchService:
    def test_results_ranked_by_score(self, cache: TTLCache) -> None:
        provider = FakeProvider(
            [result("https://a.com", "A", 0.2), result("https://b.com", "B", 0.9)]
        )
        outcome = make_service(provider, cache).search("cargo rules")
        assert [r.url for r in outcome.results] == ["https://b.com", "https://a.com"]

    def test_deduplicates_url_variants_and_same_titles(self, cache: TTLCache) -> None:
        provider = FakeProvider(
            [
                result("https://www.iata.org/dg/", "IATA DG Rules", 0.9),
                result("http://iata.org/dg", "IATA DG Rules (mirror)", 0.8),
                result("https://mirror.example.com/x", "IATA DG Rules", 0.7),
                result("https://icao.int/rules", "ICAO Annex 18", 0.6),
            ]
        )
        outcome = make_service(provider, cache).search("dangerous goods")
        assert len(outcome.results) == 2
        assert {r.url for r in outcome.results} == {
            "https://www.iata.org/dg/",
            "https://icao.int/rules",
        }

    def test_second_call_hits_cache(self, cache: TTLCache) -> None:
        provider = FakeProvider([result("https://a.com", "A", 0.5)])
        service = make_service(provider, cache)
        first = service.search("lithium batteries")
        second = service.search("Lithium   Batteries  ")  # normalization applies
        assert first.cache_hit is False
        assert second.cache_hit is True
        assert provider.calls == 1
        assert second.results[0].url == "https://a.com"

    def test_expired_cache_refetches(self, cache: TTLCache) -> None:
        provider = FakeProvider([result("https://a.com", "A", 0.5)])
        service = make_service(provider, cache, cache_ttl_seconds=1)
        # Manipulate: set a very short TTL by writing through the service
        # with a patched settings value instead of sleeping a full second.
        service._settings = WebSearchSettings(cache_ttl_seconds=1)
        service._cache.set(service._cache_key("q", 5), [], ttl_seconds=0.01)
        import time

        time.sleep(0.05)
        outcome = service.search("q")
        assert outcome.cache_hit is False
        assert provider.calls == 1

    def test_long_content_is_clipped_at_sentence_boundary(self, cache: TTLCache) -> None:
        long_text = ("Sentence about freight. " * 400).strip()  # ~9.6k chars
        provider = FakeProvider([result("https://a.com", "A", 0.5, content=long_text)])
        outcome = make_service(provider, cache).search("freight")
        content = outcome.results[0].content
        assert len(content) <= 6000
        assert content.endswith(".")


class TestTavilyProvider:
    def test_parses_tavily_response(self) -> None:
        class StubClient:
            def search(self, **kwargs: object) -> dict:
                return {
                    "results": [
                        {
                            "title": "IATA Lithium Battery Guidance",
                            "url": "https://iata.org/lithium",
                            "content": "snippet",
                            "raw_content": "Full extracted page text.",
                            "score": 0.87,
                        },
                        {"url": "", "title": "no url — dropped"},
                    ]
                }

        provider = TavilySearchProvider(Settings(_env_file=None), client=StubClient())
        results = provider.search("lithium battery rules", max_results=5)
        assert len(results) == 1
        assert results[0].content == "Full extracted page text."
        assert results[0].score == 0.87
        assert results[0].provider == "tavily"

    def test_provider_errors_are_normalized(self) -> None:
        class FailingClient:
            def search(self, **kwargs: object) -> dict:
                raise RuntimeError("connection refused")

        provider = TavilySearchProvider(Settings(_env_file=None), client=FailingClient())
        with pytest.raises(WebSearchError, match="Tavily search failed"):
            provider.search("anything", max_results=3)

    def test_missing_api_key_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from app.utils.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
            TavilySearchProvider(Settings(_env_file=None))
