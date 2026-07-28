"""LLM-based query router.

Classifies each question into one of four routes before any retrieval
happens, so the pipeline only does the work the question actually needs.
Robustness over elegance: the router model's reply is parsed defensively and
any failure (timeout, garbage output) falls back to ``HYBRID`` — the safest
route, because it consults every knowledge source.
"""

from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.chains.prompts import ROUTER_SYSTEM_PROMPT
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)


class Route(str, Enum):
    """Where the answer's knowledge should come from."""

    INTERNAL_ONLY = "INTERNAL_ONLY"
    WEB_ONLY = "WEB_ONLY"
    HYBRID = "HYBRID"
    GENERAL_CHAT = "GENERAL_CHAT"
    #: Live freight rate quote via the carrier rate API (see ``app.rates``).
    RATES = "RATES"


#: Fallback when classification fails — HYBRID consults every source, so a
#: misroute degrades to extra work, never to a wrong refusal.
_FALLBACK = Route.HYBRID


class QueryRouter:
    """Classifies queries with a small LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def classify(self, query: str) -> Route:
        """Return the route for ``query`` (never raises)."""
        with Timer() as timer:
            try:
                response = self._llm.invoke(
                    [
                        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                        HumanMessage(content=query),
                    ]
                )
                route = self._parse(str(response.content))
            except Exception as exc:  # noqa: BLE001 — router must never fail
                logger.warning("router_failed_using_fallback", error=str(exc))
                route = _FALLBACK

        logger.info("query_routed", route=route.value, latency_ms=round(timer.elapsed_ms, 1))
        return route

    @staticmethod
    def _parse(text: str) -> Route:
        """Find a route name anywhere in the reply; unknown → fallback.

        Order matters: INTERNAL_ONLY/WEB_ONLY are checked before HYBRID so a
        rambling reply mentioning several names resolves to the most specific
        one it *started* with.
        """
        upper = text.upper()
        for route in (
            Route.INTERNAL_ONLY,
            Route.WEB_ONLY,
            Route.GENERAL_CHAT,
            Route.RATES,
            Route.HYBRID,
        ):
            if route.value in upper:
                return route
        logger.warning("router_unparseable_reply", reply=text[:80])
        return _FALLBACK
