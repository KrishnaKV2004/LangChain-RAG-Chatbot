"""Tavily search provider (the default).

Tavily is purpose-built for LLM consumption: it returns cleaned page text
(``raw_content``) alongside a relevance score, so no separate scraping step
is needed for most pages.
"""

from typing import Any, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import Settings
from app.search.base import BaseSearchProvider, SearchResult
from app.utils.exceptions import ConfigurationError, WebSearchError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class _RetryableTavilyError(Exception):
    """Internal marker for transient failures worth retrying."""


class TavilySearchProvider(BaseSearchProvider):
    """Tavily implementation of the search interface."""

    name = "tavily"

    def __init__(self, settings: Settings, client: Optional[Any] = None) -> None:
        """``client`` is injectable for tests; production builds its own."""
        self._settings = settings.web_search
        if client is not None:
            self._client = client
            return

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ConfigurationError(
                "WEB_SEARCH__PROVIDER=tavily requires: pip install tavily-python"
            ) from exc

        if settings.tavily_api_key is None:
            raise ConfigurationError(
                "TAVILY_API_KEY is required for web search. "
                "Set it in your environment or .env file."
            )
        self._client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    def search(self, query: str, max_results: int) -> List[SearchResult]:
        try:
            response = self._search_with_retry(query, max_results)
        except Exception as exc:  # noqa: BLE001 — normalize provider errors
            raise WebSearchError(f"Tavily search failed: {exc}") from exc
        return self._parse(response)

    @retry(
        retry=retry_if_exception_type(_RetryableTavilyError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _search_with_retry(self, query: str, max_results: int) -> dict:
        """One Tavily call with retries on transient (5xx/network) failures."""
        try:
            return self._client.search(
                query=query,
                max_results=max_results,
                search_depth=self._settings.search_depth,
                # raw_content = extracted page text; the whole point of search.
                include_raw_content=True,
                include_answer=False,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            transient = any(hint in message for hint in ("timeout", "502", "503", "429"))
            if transient:
                raise _RetryableTavilyError(str(exc)) from exc
            raise

    def _parse(self, response: dict) -> List[SearchResult]:
        results: List[SearchResult] = []
        for item in response.get("results", []):
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=item.get("title") or url,
                    url=url,
                    # Prefer full extracted text; fall back to the snippet.
                    content=(item.get("raw_content") or item.get("content") or "").strip(),
                    score=float(item.get("score") or 0.0),
                    provider=self.name,
                    published_date=item.get("published_date"),
                )
            )
        logger.debug("tavily_parsed", results=len(results))
        return results
