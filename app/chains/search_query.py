"""Web-search query refinement.

The raw user question makes a poor search string — a request like "the nearest
Air India cargo drop-off to 1500 Atlantic St, Union City, CA" is dominated by
the user's OWN address, so the search returns property listings instead of Air
India cargo info. This chain rewrites the question into a focused query
(intent + target entities, filler and origin dropped) with the cheap
router-tier LLM, exactly as :class:`QueryRouter` classifies it.

Robustness over elegance (same stance as the router/extractor): any failure
falls back to the raw question — a bad refinement must never break search.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.chains.prompts import SEARCH_QUERY_SYSTEM_PROMPT
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Never let a refinement balloon into a paragraph; a search query is short.
_MAX_QUERY_CHARS = 200


class SearchQueryRefiner:
    """Rewrites a natural-language question into a focused web-search query."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def refine(self, question: str) -> str:
        """Return a focused search query for ``question`` (never raises)."""
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=SEARCH_QUERY_SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ]
            )
            text = str(response.content).strip()
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            refined = first_line.strip("\"'").strip()
            if not refined or len(refined) > _MAX_QUERY_CHARS:
                return question
            logger.debug("search_query_refined", original=question[:80], refined=refined[:80])
            return refined
        except Exception as exc:  # noqa: BLE001 — refinement must never break search
            logger.warning("search_query_refine_failed", error=str(exc))
            return question
