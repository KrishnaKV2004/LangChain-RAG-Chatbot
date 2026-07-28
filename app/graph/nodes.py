"""LangGraph node implementations.

Each node is a small method on :class:`GraphNodes` that reads from the state
and returns a partial update. All collaborators are injected, so every node
is unit-testable with fakes and the graph wiring stays trivial.

Node map (see ``workflow.py`` for the edges):

    input_gate → classify → retrieve → web_search → merge_context
              → generate → build_citations → validate_response
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.documents import Document

from app.chains.answer import AnswerChain
from app.chains.citations import CitationBuilder, Citations
from app.chains.rate_extraction import RateExtractor
from app.graph.state import AgentState
from app.rates.models import RateQuery, RateQuote
from app.retrievers.pipeline import RetrievalPipeline
from app.routers.classifier import QueryRouter, Route
from app.search.service import WebSearchService
from app.security.guard import SecurityGuard
from app.security.models import UserContext
from app.utils.exceptions import RateProviderError, WebSearchError
from app.utils.logging import get_logger
from app.utils.timing import Timer

if TYPE_CHECKING:  # imports only for typing — avoid an import cycle / heavy deps
    from app.agents.hermes import HermesRefiner
    from app.rates.seven_l import SevenLRates

logger = get_logger(__name__)


def _metrics(state: AgentState, name: str, value: float) -> Dict[str, float]:
    """Copy-and-update the metrics dict (nodes run sequentially)."""
    metrics = dict(state.get("metrics", {}))
    metrics[name] = round(value, 1)
    return metrics


class GraphNodes:
    """All workflow nodes with their injected dependencies."""

    def __init__(
        self,
        router: QueryRouter,
        retrieval: RetrievalPipeline,
        web_search: WebSearchService,
        guard: SecurityGuard,
        answer_chain: AnswerChain,
        citation_builder: CitationBuilder,
        refiner: "Optional[HermesRefiner]" = None,
        rate_extractor: Optional[RateExtractor] = None,
        rate_service: "Optional[SevenLRates]" = None,
    ) -> None:
        self._router = router
        self._retrieval = retrieval
        self._web_search = web_search
        self._guard = guard
        self._answer_chain = answer_chain
        self._citations = citation_builder
        self._refiner = refiner
        # Optional: present only when 7LFreight credentials are configured.
        self._rate_extractor = rate_extractor
        self._rate_service = rate_service

    # ------------------------------------------------------------------ #
    # 1. Input gate — security checkpoint 1
    # ------------------------------------------------------------------ #

    def input_gate(self, state: AgentState) -> Dict[str, Any]:
        verdict = self._guard.validate_query(state["query"])
        if not verdict.allowed:
            return {
                "blocked": True,
                "refusal": self._guard.refusal_message,
                "answer": self._guard.refusal_message,
                "citations": Citations(),
            }
        return {"blocked": False}

    # ------------------------------------------------------------------ #
    # 2. Query classification
    # ------------------------------------------------------------------ #

    def classify(self, state: AgentState) -> Dict[str, Any]:
        # A pre-set route (the UI's "search mode" override) skips the LLM.
        forced = state.get("route")
        if forced:
            return {"route": forced, "metrics": _metrics(state, "routing_ms", 0.0)}
        with Timer() as timer:
            route = self._router.classify(state["query"])
        return {
            "route": route.value,
            "metrics": _metrics(state, "routing_ms", timer.elapsed_ms),
        }

    # ------------------------------------------------------------------ #
    # 3. Document retrieval (internal knowledge)
    # ------------------------------------------------------------------ #

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        result = self._retrieval.retrieve(state["query"])
        return {
            "internal_docs": result.documents,  # security-filtered in merge
            "metrics": _metrics(state, "retrieval_ms", result.latency_ms),
        }

    # ------------------------------------------------------------------ #
    # 4. Web search — failures degrade gracefully to "no web context"
    # ------------------------------------------------------------------ #

    def web_search(self, state: AgentState) -> Dict[str, Any]:
        try:
            outcome = self._web_search.search(state["query"])
        except WebSearchError as exc:
            logger.warning("web_search_degraded", error=exc.message)
            return {
                "web_docs": [],
                "web_cache_hit": False,
                "metrics": _metrics(state, "web_search_ms", 0.0),
            }

        documents = [
            Document(
                page_content=result.content,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "provider": result.provider,
                    # Web content is public by definition; without this the
                    # sensitivity default (internal) would hide it from guests.
                    "sensitivity": "public",
                },
            )
            for result in outcome.results
        ]
        return {
            "web_docs": documents,
            "web_cache_hit": outcome.cache_hit,
            "metrics": _metrics(state, "web_search_ms", outcome.latency_ms),
        }

    # ------------------------------------------------------------------ #
    # 4b. Rate quoting (RATES route): NL → RateQuery, then the carrier API
    # ------------------------------------------------------------------ #

    def extract_rate_query(self, state: AgentState) -> Dict[str, Any]:
        """Parse the user's message into a structured :class:`RateQuery`.

        When rates aren't configured, or the message lacks a required field,
        this ships a message straight to the user and no API call is made.
        """
        if self._rate_extractor is None or self._rate_service is None:
            return {
                "answer": "Live freight rate quoting isn't configured on this "
                "assistant right now.",
                "rate_quotes": [],
            }
        with Timer() as timer:
            extraction = self._rate_extractor.extract(
                state["query"], state.get("history", [])
            )
        metrics = _metrics(state, "rate_extraction_ms", timer.elapsed_ms)
        if not extraction.ok:
            # Missing origin/destination/weight — ask, don't guess.
            return {
                "answer": extraction.clarification,
                "rate_quotes": [],
                "metrics": metrics,
            }
        return {"rate_query": extraction.query, "metrics": metrics}

    def fetch_rates(self, state: AgentState) -> Dict[str, Any]:
        """Call the rate service and format the answer DETERMINISTICALLY.

        The answer is built from the real quotes (exact figures + currency),
        never generated by the LLM — so prices can't be hallucinated or
        rounded, and there is no injection surface. When the provider fails or
        returns nothing, we say so honestly instead of estimating a price.
        """
        query = state.get("rate_query")
        if query is None or self._rate_service is None:
            return {"rate_docs": [], "rate_quotes": [],
                    "answer": "I couldn't parse a shippable request from that."}
        try:
            outcome = self._rate_service.get_rates(query)
        except RateProviderError as exc:
            is_auth = exc.details.get("kind") == "auth"
            logger.warning(
                "rate_lookup_degraded", error=exc.message, kind="auth" if is_auth else "transient"
            )
            message = (
                self._rate_config_message(query)
                if is_auth
                else self._rate_unavailable_message(query)
            )
            return {
                "rate_docs": [],
                "rate_quotes": [],
                "answer": message,
                "metrics": _metrics(state, "rate_lookup_ms", 0.0),
            }
        return {
            "rate_docs": [self._rate_document(quote) for quote in outcome.quotes],
            "rate_quotes": [quote.to_dict() for quote in outcome.quotes],
            "answer": self._format_rate_answer(query, outcome.quotes),
            "metrics": _metrics(state, "rate_lookup_ms", outcome.latency_ms),
        }

    # ------------------------------------------------------------------ #
    # Deterministic rate presentation (no LLM → exact figures, no estimates)
    # ------------------------------------------------------------------ #

    @classmethod
    def _format_rate_answer(cls, query: "RateQuery", quotes: List[RateQuote]) -> str:
        lane = cls._lane(query)
        mode = query.mode.value
        if not quotes:
            return (
                f"I couldn't find any live {mode} freight rates for {lane} right now. "
                "Please double-check the origin, destination and weight — or try again "
                "shortly, as carriers don't always return a rate for every lane."
            )
        lines = [f"Here are the live {mode} freight quotes for {lane}, cheapest first:", ""]
        for quote in quotes:  # already sorted cheapest-first by the service
            code = f" ({quote.carrier_code})" if quote.carrier_code else ""
            # Currency as an ISO code ("USD 1,200.00"), never a "$" — a bare "$"
            # is rendered as broken LaTeX math by the markdown chat UI.
            price = f"{quote.currency} {quote.price:,.2f}"
            transit = f" · about {quote.transit_days} days" if quote.transit_days is not None else ""
            lines.append(f"- {quote.carrier_name}{code}: {price}{transit}")
        return "\n".join(lines)

    @classmethod
    def _rate_unavailable_message(cls, query: "RateQuery") -> str:
        return (
            f"The rate service is temporarily unavailable, so I couldn't pull live "
            f"{query.mode.value} quotes for {cls._lane(query)} right now. Please try again "
            "in a moment. I won't guess at a price."
        )

    @classmethod
    def _rate_config_message(cls, query: "RateQuery") -> str:
        # Auth was rejected — retrying won't help; an administrator must fix the
        # freight-provider credentials. We don't estimate a price.
        return (
            "I can't pull live freight rates right now because the rate provider isn't "
            "accepting our credentials. This is a configuration issue on our side — please "
            "ask an administrator to check the 7LFreight login (7L_USERNAME / 7L_PASSWORD). "
            "I won't guess at a price."
        )

    @staticmethod
    def _lane(query: "RateQuery") -> str:
        return f"{query.origin.label()} → {query.destination.label()}"

    @staticmethod
    def _rate_document(quote: RateQuote) -> Document:
        """Render one quote as a Document so it flows through the answer chain."""
        code = f" ({quote.carrier_code})" if quote.carrier_code else ""
        lines = [
            f"Carrier: {quote.carrier_name}{code}",
            f"Lane: {quote.origin} -> {quote.destination} ({quote.mode})",
            f"Price: {quote.price:.2f} {quote.currency}",
        ]
        if quote.transit_days is not None:
            lines.append(f"Transit: {quote.transit_days} days")
        if quote.valid_from or quote.valid_to:
            lines.append(f"Valid: {quote.valid_from or '?'} to {quote.valid_to or '?'}")
        if quote.remarks:
            lines.append(f"Remarks: {quote.remarks}")
        return Document(
            page_content="\n".join(lines),
            metadata={
                "carrier": quote.carrier_name,
                "carrier_code": quote.carrier_code,
                "mode": quote.mode,
                "provider": quote.provider,
                "price": quote.price,
                "currency": quote.currency,
                "transit_days": quote.transit_days,
                # Rate quotes are public, not confidential company documents.
                "sensitivity": "public",
            },
        )

    # ------------------------------------------------------------------ #
    # 5. Context merge — security checkpoint 2 (Layers 2+3 + poisoning scan)
    # ------------------------------------------------------------------ #

    def merge_context(self, state: AgentState) -> Dict[str, Any]:
        user = state.get("user") or UserContext()
        with Timer() as timer:
            internal = self._guard.filter_context(user, state.get("internal_docs", []))
            web = self._guard.filter_context(user, state.get("web_docs", []))
            # Rate quotes are public data, but a carrier's free-text "remarks"
            # field is still untrusted input, so run them through the same scan.
            rate = self._guard.filter_context(user, state.get("rate_docs", []))
        return {
            "internal_docs": internal.documents,
            "web_docs": web.documents,
            "rate_docs": rate.documents,
            "denied_texts": internal.denied_texts + web.denied_texts + rate.denied_texts,
            "dropped_injected": internal.dropped_injected
            + web.dropped_injected
            + rate.dropped_injected,
            "denied_by_permission": internal.denied_by_permission
            + web.denied_by_permission
            + rate.denied_by_permission,
            "metrics": _metrics(state, "security_filter_ms", timer.elapsed_ms),
        }

    # ------------------------------------------------------------------ #
    # 6. Answer generation
    # ------------------------------------------------------------------ #

    def generate(self, state: AgentState) -> Dict[str, Any]:
        general_chat = state.get("route") == Route.GENERAL_CHAT.value
        result = self._answer_chain.generate(
            query=state["query"],
            internal_docs=state.get("internal_docs", []),
            web_docs=state.get("web_docs", []),
            history=state.get("history", []),
            general_chat=general_chat,
            rate_docs=state.get("rate_docs", []),
        )
        return {
            "answer": result.answer,
            "token_usage": result.token_usage,
            "metrics": _metrics(state, "generation_ms", result.latency_ms),
        }

    # ------------------------------------------------------------------ #
    # 6b. Corrective fallback — internal dead-end escalates to web search
    # ------------------------------------------------------------------ #

    def escalate_to_web(self, state: AgentState) -> Dict[str, Any]:
        """An INTERNAL_ONLY answer admitted defeat: retry with web context.

        The route flips to HYBRID and the flow loops back through
        web_search → merge_context → generate. ``web_fallback_used`` makes
        this a one-shot escalation, never an infinite loop.
        """
        logger.info("web_fallback_triggered", query=state["query"][:80])
        return {
            "route": Route.HYBRID.value,
            "web_fallback_used": True,
            "metrics": _metrics(state, "web_fallback", 1.0),
        }

    # ------------------------------------------------------------------ #
    # 7. Hermes refinement — self-improve the draft before it ships
    # ------------------------------------------------------------------ #

    def refine(self, state: AgentState) -> Dict[str, Any]:
        if self._refiner is None:
            return {}
        result = self._refiner.refine(
            query=state["query"],
            draft=state.get("answer", ""),
            internal_docs=state.get("internal_docs", []),
            web_docs=state.get("web_docs", []),
        )
        # Hermes tokens count toward the turn's total.
        usage = dict(state.get("token_usage", {}))
        for key, value in result.token_usage.items():
            usage[key] = usage.get(key, 0) + value

        metrics = _metrics(state, "refinement_ms", result.latency_ms)
        metrics["refinement_iterations"] = float(result.iterations)
        return {"answer": result.answer, "token_usage": usage, "metrics": metrics}

    # ------------------------------------------------------------------ #
    # 8. Citation builder — from the exact documents the LLM saw
    # ------------------------------------------------------------------ #

    def build_citations(self, state: AgentState) -> Dict[str, Any]:
        citations = self._citations.build(
            state.get("internal_docs", []),
            state.get("web_docs", []),
            state.get("rate_docs", []),
        )
        return {"citations": citations}

    # ------------------------------------------------------------------ #
    # 9. Response validation — security checkpoint 3 (Layers 4+5)
    # ------------------------------------------------------------------ #

    def validate_response(self, state: AgentState) -> Dict[str, Any]:
        with Timer() as timer:
            decision = self._guard.validate_response(
                state.get("answer", ""), state.get("denied_texts", [])
            )
        update: Dict[str, Any] = {
            "answer": decision.answer,
            "metrics": _metrics(state, "response_scan_ms", timer.elapsed_ms),
        }
        if decision.blocked:
            # A blocked answer must not carry citations that hint at what
            # the suppressed content was.
            update["blocked"] = True
            update["citations"] = Citations()
        return update
