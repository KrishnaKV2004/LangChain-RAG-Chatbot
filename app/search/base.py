"""Web-search provider interface.

Adding a provider (Google Custom Search, SerpAPI, Brave, DuckDuckGo, ...)
means implementing :class:`BaseSearchProvider` and registering the class in
``app.search.factory`` — the service layer, caching, deduplication and the
LangGraph node all stay untouched.
"""

import abc
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """One webpage returned by a provider, normalized across providers."""

    title: str
    url: str
    #: Extracted page text (what the LLM reads). Truncated by the service.
    content: str
    #: Provider's own relevance score, normalized to [0, 1] where available.
    score: float = 0.0
    provider: str = "unknown"
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form for caching."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(**data)


class BaseSearchProvider(abc.ABC):
    """Contract every web-search backend must fulfil."""

    #: Machine name, matching the ``SearchProvider`` enum value.
    name: str = "base"

    @abc.abstractmethod
    def search(self, query: str, max_results: int) -> List[SearchResult]:
        """Run a web search.

        Implementations must raise
        :class:`app.utils.exceptions.WebSearchError` on failure — callers
        never see provider-specific exceptions.
        """
