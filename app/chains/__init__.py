"""Prompt templates and LLM chains for answer generation and citations."""

from app.chains.answer import AnswerChain, GenerationResult
from app.chains.citations import CitationBuilder, Citations
from app.chains.llm_factory import create_chat_model
from app.chains.prompts import PROTECTED_MARKERS, build_system_prompt

__all__ = [
    "AnswerChain",
    "CitationBuilder",
    "Citations",
    "GenerationResult",
    "PROTECTED_MARKERS",
    "build_system_prompt",
    "create_chat_model",
]
