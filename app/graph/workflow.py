"""LangGraph workflow assembly.

Routing logic lives here as conditional edges; behaviour lives in the nodes.

           ┌──────────── blocked ────────────► END
    input_gate
           └► classify ─┬─ INTERNAL_ONLY ─► retrieve ──────────┐
                        ├─ HYBRID ────────► retrieve ─► web_search ─┐
                        ├─ WEB_ONLY ──────► web_search ─────────┤
                        ├─ RATES ─► extract_rate_query ─► fetch_rates ─► build_citations
                        │                └─ (needs info) ─► build_citations
                        └─ GENERAL_CHAT ──────────────────► generate
                                        merge_context ─► generate
                                                              │
                            validate_response ◄─ build_citations
"""

from langgraph.graph import END, START, StateGraph

from app.chains.answer import is_non_answer
from app.graph.nodes import GraphNodes
from app.graph.state import AgentState
from app.routers.classifier import Route


def _after_refine(state: AgentState) -> str:
    """Corrective-fallback gate: an INTERNAL_ONLY reply that admits it
    couldn't answer gets one retry with web context before it ships.

    Evaluated after Hermes so the check sees the answer in its final,
    condensed form (a verbose draft refusal could otherwise slip past)."""
    if (
        state.get("route") == Route.INTERNAL_ONLY.value
        and not state.get("web_fallback_used")
        and is_non_answer(state.get("answer", ""))
    ):
        return "escalate"
    return "citations"


def build_workflow(nodes: GraphNodes):
    """Compile the full agent graph from a node collection."""
    graph = StateGraph(AgentState)

    graph.add_node("input_gate", nodes.input_gate)
    graph.add_node("classify", nodes.classify)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("extract_rate_query", nodes.extract_rate_query)  # RATES route
    graph.add_node("fetch_rates", nodes.fetch_rates)
    graph.add_node("merge_context", nodes.merge_context)
    graph.add_node("generate", nodes.generate)
    graph.add_node("escalate_to_web", nodes.escalate_to_web)  # corrective fallback
    graph.add_node("refine", nodes.refine)  # Hermes self-improvement pass
    graph.add_node("build_citations", nodes.build_citations)
    graph.add_node("validate_response", nodes.validate_response)

    graph.add_edge(START, "input_gate")

    # Blocked input short-circuits the whole pipeline.
    graph.add_conditional_edges(
        "input_gate",
        lambda state: "blocked" if state.get("blocked") else "ok",
        {"blocked": END, "ok": "classify"},
    )

    # Route → which knowledge sources to consult.
    graph.add_conditional_edges(
        "classify",
        lambda state: state["route"],
        {
            Route.INTERNAL_ONLY.value: "retrieve",
            Route.HYBRID.value: "retrieve",
            Route.WEB_ONLY.value: "web_search",
            Route.GENERAL_CHAT.value: "generate",
            Route.RATES.value: "extract_rate_query",
        },
    )

    # RATES: extraction either yields a query (→ fetch) or needs more info from
    # the user (→ straight to citations with the clarification already set).
    # fetch_rates formats the answer deterministically from the real quotes, so
    # it goes straight to citations — bypassing the LLM (which would otherwise
    # estimate a price) and Hermes (which would rewrite exact figures).
    graph.add_conditional_edges(
        "extract_rate_query",
        lambda state: "fetch" if state.get("rate_query") else "answer",
        {"fetch": "fetch_rates", "answer": "build_citations"},
    )
    graph.add_edge("fetch_rates", "build_citations")

    # HYBRID continues from retrieval into web search; INTERNAL goes straight
    # to the merge/security stage.
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "web" if state["route"] == Route.HYBRID.value else "merge",
        {"web": "web_search", "merge": "merge_context"},
    )

    graph.add_edge("web_search", "merge_context")
    graph.add_edge("merge_context", "generate")
    graph.add_edge("generate", "refine")
    # Corrective fallback: an internal-only "I can't answer that" loops back
    # through web search once; everything else proceeds to citations.
    graph.add_conditional_edges(
        "refine",
        _after_refine,
        {"escalate": "escalate_to_web", "citations": "build_citations"},
    )
    graph.add_edge("escalate_to_web", "web_search")
    graph.add_edge("build_citations", "validate_response")
    graph.add_edge("validate_response", END)

    return graph.compile()
