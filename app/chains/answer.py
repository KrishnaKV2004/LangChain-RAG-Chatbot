"""Answer-generation chain.

Assembles the message list (system prompt with data-wrapped context →
conversation history → current question), invokes the LLM, and reports token
usage alongside the text.

The model is instructed to deliberate inside ``<thinking>`` tags and put the
user-facing reply inside ``<answer>`` tags; only the answer is returned.
This keeps chain-of-thought rambling ("...actually, a better answer is...")
structurally out of the response instead of relying on the model's restraint.
"""

import re
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

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
# Matches a thinking block even when truncation cut off its closing tag.
_THINKING_RE = re.compile(r"<thinking>.*?(?:</thinking>|$)", re.S | re.I)


#: Refusal-phrasing families that mark a reply as "I couldn't answer". Used
#: by the graph's corrective fallback: an internal-only non-answer triggers a
#: web retry. Regexes (not fixed strings) so wording variants can't slip by.
_NON_ANSWER_PATTERNS = (
    re.compile(r"\bi\s+(?:can'?t|cannot|can\s+not)\b", re.I),
    re.compile(r"\b(?:i'?m|i\s+am)\s+(?:unable|not\s+able)\b", re.I),
    re.compile(r"\bunable\s+to\s+(?:determine|find|tell|answer|provide|locate)\b", re.I),
    # "don't have (that/the/enough/any) information/data/details/answer"
    re.compile(
        r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not)\s+have\b[^.!?]{0,60}"
        r"\b(?:information|data|details?|specifics|answer)\b",
        re.I,
    ),
    re.compile(r"\bno\s+(?:information|data|details?)\b", re.I),
    re.compile(r"\b(?:information|data|details?)\s+(?:is|are)?\s*not\s+available\b", re.I),
    re.compile(r"\bnot\s+available\s+in\s+(?:the|my|our)\b", re.I),
    # "the documents don't contain / provide / include / specify / mention / list ..."
    re.compile(
        r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not)\s+"
        r"(?:contain|provide|include|specify|mention|list)\b",
        re.I,
    ),
    re.compile(r"\bnot\s+(?:specified|mentioned|listed|provided)\s+in\b", re.I),
)

#: Long answers that merely *include* a hedge are real answers; genuine
#: refusals are short (checked post-Hermes, which condenses verbose ones).
_NON_ANSWER_MAX_CHARS = 500


def is_non_answer(text: str) -> bool:
    """True when the reply admits it couldn't answer the question."""
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) > _NON_ANSWER_MAX_CHARS:
        return False
    return any(pattern.search(stripped) for pattern in _NON_ANSWER_PATTERNS)


def extract_final_answer(text: str) -> str:
    """Return only the user-facing part of the model's output.

    Preference order: the ``<answer>`` block; otherwise the text with any
    thinking block removed; otherwise (nothing left) the raw text — a model
    that ignored the tag protocol entirely must still produce a reply.
    """
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    stripped = _THINKING_RE.sub("", text).replace("<answer>", "").strip()
    return stripped or text.strip()


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
        rate_docs: Optional[List[Document]] = None,
    ) -> GenerationResult:
        """Produce an answer grounded in the supplied context."""
        messages = self._build_messages(
            query, internal_docs, web_docs, history, general_chat, rate_docs
        )

        with Timer() as timer:
            try:
                response = self._llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001 — normalize provider errors
                raise LLMError(f"Answer generation failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None) or {}
        result = GenerationResult(
            answer=extract_final_answer(str(response.content)),
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
        rate_docs: Optional[List[Document]] = None,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [
            SystemMessage(
                content=build_system_prompt(
                    internal_docs, web_docs, rate_docs, general_chat=general_chat
                )
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
