"""Search-provider factory.

Mirrors the embedding factory: the enum constrains the choice, this module
builds the concrete provider, and everything else depends on the interface.
Providers listed in the enum but not yet implemented raise a clear
configuration error instead of failing mysteriously downstream.
"""

from typing import Callable, Dict

from app.config.settings import SearchProvider, Settings
from app.search.base import BaseSearchProvider
from app.search.tavily import TavilySearchProvider
from app.utils.exceptions import ConfigurationError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_BUILDERS: Dict[SearchProvider, Callable[[Settings], BaseSearchProvider]] = {
    SearchProvider.TAVILY: TavilySearchProvider,
    # Extension point — implement BaseSearchProvider and register here:
    # SearchProvider.GOOGLE: GoogleCustomSearchProvider,
    # SearchProvider.SERPAPI: SerpAPIProvider,
    # SearchProvider.BRAVE: BraveSearchProvider,
    # SearchProvider.DUCKDUCKGO: DuckDuckGoProvider,
}


def create_search_provider(settings: Settings) -> BaseSearchProvider:
    """Build the provider selected by ``WEB_SEARCH__PROVIDER``."""
    provider = settings.web_search.provider
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ConfigurationError(
            f"Search provider '{provider.value}' is not implemented yet. "
            f"Available: {', '.join(p.value for p in _BUILDERS)}"
        )
    instance = builder(settings)
    logger.info("search_provider_initialized", provider=provider.value)
    return instance
