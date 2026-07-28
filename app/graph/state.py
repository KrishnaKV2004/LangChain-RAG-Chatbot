"""LangGraph state definition.

One :class:`AgentState` dictionary flows through every node. Nodes read what
they need and return only the keys they update (LangGraph merges partial
updates into the state).
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document

from app.chains.citations import Citations
from app.rates.models import RateQuery
from app.security.models import UserContext


class AgentState(TypedDict, total=False):
    """Everything the workflow knows about one request."""

    # ---- Input ---------------------------------------------------------- #
    query: str
    history: List[Dict[str, str]]
    user: UserContext

    # ---- Routing / security gate ---------------------------------------- #
    route: str                    # Route enum value
    blocked: bool                 # set by the input gate or response validator
    refusal: Optional[str]        # message to return when blocked
    rate_query: RateQuery         # structured freight request (RATES route)

    # ---- Retrieved context ------------------------------------------------ #
    internal_docs: List[Document]     # after security filtering
    web_docs: List[Document]          # web results as Documents
    rate_docs: List[Document]         # freight rate quotes as Documents (RATES route)
    rate_quotes: List[Dict]           # same quotes, structured, for the API/UI
    denied_texts: List[str]           # for Layer 4's leak check
    dropped_injected: int
    denied_by_permission: int
    web_cache_hit: bool
    #: Set once the corrective fallback (internal non-answer → web retry)
    #: has run, so the loop can never trigger twice.
    web_fallback_used: bool

    # ---- Output ------------------------------------------------------------ #
    answer: str
    citations: Citations
    token_usage: Dict[str, int]

    # ---- Observability ------------------------------------------------------ #
    metrics: Dict[str, float]     # per-stage latency in ms
