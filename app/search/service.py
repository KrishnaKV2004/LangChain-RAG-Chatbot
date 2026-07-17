"""Web-search service: provider + ranking + deduplication + TTL caching.

Pipeline per query:

1. **Cache lookup** — key is (provider, normalized query, max_results);
   a hit skips the network entirely and is logged as such.
2. **Provider search** — via the configured :class:`BaseSearchProvider`.
3. **Deduplicate** — by normalized URL first, then by near-identical titles
   (mirrors and trackers produce both kinds of duplicates).
4. **Rank** — provider relevance score, descending.
5. **Condense** — page text is clipped to a budget per result; deep
   summarization is the answer-generation LLM's job, which sees the query
   and can weigh the pages jointly.
6. **Cache store** — TTL-bound (default 24 h). Web content is never written
   to the vector store or any permanent storage.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.cache.ttl_cache import TTLCache
from app.config.settings import WebSearchSettings
from app.search.base import BaseSearchProvider, SearchResult
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: Character budget per result passed to the LLM (≈1.5k tokens).
_MAX_CONTENT_CHARS = 6000


@dataclass
class WebSearchOutcome:
    """Search results plus the observability fields callers log/display."""

    results: List[SearchResult] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.results


class WebSearchService:
    """Cached, deduplicated web search over a pluggable provider."""

    def __init__(
        self,
        provider: BaseSearchProvider,
        cache: TTLCache,
        settings: WebSearchSettings,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._settings = settings

    def search(self, query: str, max_results: Optional[int] = None) -> WebSearchOutcome:
        """Search the web, serving from cache when possible."""
        limit = max_results or self._settings.max_results
        cache_key = self._cache_key(query, limit)
        outcome = WebSearchOutcome()

        with Timer() as timer:
            cached = self._cache.get(cache_key)
            if cached is not None:
                outcome.results = [SearchResult.from_dict(item) for item in cached]
                outcome.cache_hit = True
            else:
                raw = self._provider.search(query, max_results=limit)
                results = self._deduplicate(raw)
                results.sort(key=lambda r: r.score, reverse=True)
                results = [self._condense(r) for r in results[:limit]]
                self._cache.set(
                    cache_key,
                    [r.to_dict() for r in results],
                    ttl_seconds=self._settings.cache_ttl_seconds,
                )
                outcome.results = results

        outcome.latency_ms = timer.elapsed_ms
        logger.info(
            "web_search_complete",
            provider=self._provider.name,
            results=len(outcome.results),
            cache_hit=outcome.cache_hit,
            latency_ms=round(outcome.latency_ms, 1),
        )
        return outcome

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _cache_key(self, query: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return f"search:{self._provider.name}:{limit}:{normalized}"

    @staticmethod
    def _deduplicate(results: List[SearchResult]) -> List[SearchResult]:
        """Drop same-page duplicates (URL variants, syndicated titles)."""
        seen_urls = set()
        seen_titles = set()
        unique: List[SearchResult] = []
        for result in results:
            url_key = re.sub(r"^https?://(www\.)?", "", result.url).rstrip("/").lower()
            title_key = re.sub(r"\W+", "", result.title.lower())
            if url_key in seen_urls or (title_key and title_key in seen_titles):
                continue
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            unique.append(result)
        return unique

    @staticmethod
    def _condense(result: SearchResult) -> SearchResult:
        """Clip page text to the per-result budget at a sentence boundary."""
        if len(result.content) > _MAX_CONTENT_CHARS:
            clipped = result.content[:_MAX_CONTENT_CHARS]
            # Cut at the last sentence end so the LLM never sees half words.
            last_stop = clipped.rfind(". ")
            if last_stop > _MAX_CONTENT_CHARS // 2:
                clipped = clipped[: last_stop + 1]
            result.content = clipped
        return result
