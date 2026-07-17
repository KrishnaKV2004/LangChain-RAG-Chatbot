"""Answer-generation chain.

Assembles the message list (system prompt with data-wrapped context →
conversation history → current question), invokes the LLM, and reports token
usage alongside the text.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.chains.prompts import build_system_prompt
from app.utils.exceptions import LLMError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: History beyond this many messages is dropped (oldest first) to bound cost.
_MAX_HISTORY_MESSAGES = 10


@dataclass
class GenerationResult:
    """LLM output plus the metrics the UI displays."""

    answer: str
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


class AnswerChain:
    """Generates the final answer from context + history + question."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def generate(
        self,
        query: str,
        internal_docs: Optional[List[Document]] = None,
        web_docs: Optional[List[Document]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        general_chat: bool = False,
    ) -> GenerationResult:
        """Produce an answer grounded in the supplied context."""
        messages = self._build_messages(query, internal_docs, web_docs, history, general_chat)

        with Timer() as timer:
            try:
                response = self._llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 — normalize provider errors
                raise LLMError(f"Answer generation failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None) or {}
        result = GenerationResult(
            answer=str(response.content).strip(),
            latency_ms=timer.elapsed_ms,
            token_usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        logger.info(
            "answer_generated",
            latency_ms=round(result.latency_ms, 1),
            tokens=result.token_usage.get("total_tokens", 0),
        )
        return result

    @staticmethod
    def _build_messages(
        query: str,
        internal_docs: Optional[List[Document]],
        web_docs: Optional[List[Document]],
        history: Optional[List[Dict[str, str]]],
        general_chat: bool,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [
            SystemMessage(
                content=build_system_prompt(internal_docs, web_docs, general_chat)
            )
        ]
        # History arrives as [{"role": "user"|"assistant", "content": ...}]
        # (JSON-friendly for the API); only the recent tail is kept.
        for turn in (history or [])[-_MAX_HISTORY_MESSAGES:]:
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=turn.get("content", "")))
            elif turn.get("role") == "assistant":
                messages.append(AIMessage(content=turn.get("content", "")))
        messages.append(HumanMessage(content=query))
        return messages
