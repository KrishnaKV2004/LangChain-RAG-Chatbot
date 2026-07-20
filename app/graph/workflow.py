"""LangGraph workflow assembly.

Routing logic lives here as conditional edges; behaviour lives in the nodes.

           ┌──────────── blocked ────────────► END
    input_gate
           └► classify ─┬─ INTERNAL_ONLY ─► retrieve ──────────┐
                        ├─ HYBRID ────────► retrieve ─► web_search ─┐
                        ├─ WEB_ONLY ──────► web_search ─────────┤
                        └─ GENERAL_CHAT ──────────────────► generate
                                        merge_context ─► generate
                                                              │
                            validate_response ◄─ build_citations
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import GraphNodes
from app.graph.state import AgentState
from app.routers.classifier import Route


def build_workflow(nodes: GraphNodes):
    """Compile the full agent graph from a node collection."""
    graph = StateGraph(AgentState)

    graph.add_node("input_gate", nodes.input_gate)
    graph.add_node("classify", nodes.classify)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("merge_context", nodes.merge_context)
    graph.add_node("generate", nodes.generate)
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
        },
    )

    # HYBRID continues from retrieval into web search; INTERNAL goes straight
    # to the merge/security stage.
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "web" if state["route"] == Route.HYBRID.value else "merge",
        {"web": "web_search", "merge": "merge_context"},
    )

    graph.add_edge("web_search", "merge_context")
    graph.add_edge("merge_context", "generate")
    # Hermes reviews/improves the draft; security still validates the result.
    graph.add_edge("generate", "refine")
    graph.add_edge("refine", "build_citations")
    graph.add_edge("build_citations", "validate_response")
    graph.add_edge("validate_response", END)

    return graph.compile()
