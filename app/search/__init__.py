"""Web-search provider interface and implementations (Tavily default)."""

from app.search.base import BaseSearchProvider, SearchResult
from app.search.factory import create_search_provider
from app.search.service import WebSearchOutcome, WebSearchService
from app.search.tavily import TavilySearchProvider

__all__ = [
    "BaseSearchProvider",
    "SearchResult",
    "TavilySearchProvider",
    "WebSearchOutcome",
    "WebSearchService",
    "create_search_provider",
]
