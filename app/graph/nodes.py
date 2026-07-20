"""LangGraph node implementations.

Each node is a small method on :class:`GraphNodes` that reads from the state
and returns a partial update. All collaborators are injected, so every node
is unit-testable with fakes and the graph wiring stays trivial.

Node map (see ``workflow.py`` for the edges):

    input_gate → classify → retrieve → web_search → merge_context
              → generate → build_citations → validate_response
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.agents.hermes import HermesRefiner
from app.chains.answer import AnswerChain
from app.chains.citations import CitationBuilder, Citations
from app.graph.state import AgentState
from app.retrievers.pipeline import RetrievalPipeline
from app.routers.classifier import QueryRouter, Route
from app.search.service import WebSearchService
from app.security.guard import SecurityGuard
from app.security.models import UserContext
from app.utils.exceptions import WebSearchError
from app.utils.logging import get_logger
from app.utils.timing import Timer

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
        refiner: Optional[HermesRefiner] = None,
    ) -> None:
        self._router = router
        self._retrieval = retrieval
        self._web_search = web_search
        self._guard = guard
        self._answer_chain = answer_chain
        self._citations = citation_builder
        self._refiner = refiner

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
    # 5. Context merge — security checkpoint 2 (Layers 2+3 + poisoning scan)
    # ------------------------------------------------------------------ #

    def merge_context(self, state: AgentState) -> Dict[str, Any]:
        user = state.get("user") or UserContext()
        with Timer() as timer:
            internal = self._guard.filter_context(user, state.get("internal_docs", []))
            web = self._guard.filter_context(user, state.get("web_docs", []))
        return {
            "internal_docs": internal.documents,
            "web_docs": web.documents,
            "denied_texts": internal.denied_texts + web.denied_texts,
            "dropped_injected": internal.dropped_injected + web.dropped_injected,
            "denied_by_permission": internal.denied_by_permission
            + web.denied_by_permission,
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
        )
        return {
            "answer": result.answer,
            "token_usage": result.token_usage,
            "metrics": _metrics(state, "generation_ms", result.latency_ms),
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
            state.get("internal_docs", []), state.get("web_docs", [])
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
