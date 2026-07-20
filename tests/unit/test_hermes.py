"""Unit tests for the Hermes self-refinement agent."""

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.agents.hermes import HermesRefiner
from app.config.settings import HermesSettings
from tests.unit.test_router_chains import ScriptedChatModel, scripted


def make_refiner(model: ScriptedChatModel, max_iterations: int = 2) -> HermesRefiner:
    return HermesRefiner(model, HermesSettings(max_iterations=max_iterations))


class TestHermesRefiner:
    def test_approved_draft_is_kept_unchanged(self) -> None:
        result = make_refiner(scripted("APPROVED")).refine(
            "Which airport is closest to Denver?", "Denver International Airport (DEN)."
        )
        assert result.answer == "Denver International Airport (DEN)."
        assert result.improved is False
        assert result.iterations == 1

    def test_bad_draft_is_rewritten_then_approved(self) -> None:
        calls: List[str] = []

        def responder(messages: List[BaseMessage]) -> str:
            calls.append(messages[-1].content)
            if len(calls) == 1:
                return "The closest listed airport to Denver is DEN."
            return "APPROVED"

        refiner = make_refiner(ScriptedChatModel(responder=responder))
        result = refiner.refine(
            "Which airport is closest to Denver?",
            "It could be BJC, but a better answer would be DEN, or maybe SLC.",
        )
        assert result.answer == "The closest listed airport to Denver is DEN."
        assert result.improved is True
        assert result.iterations == 2
        # Second review must see the *revised* draft, not the original.
        assert "closest listed airport" in calls[1]

    def test_iteration_budget_is_respected(self) -> None:
        # Reviewer never approves; the loop must stop at max_iterations.
        counter = {"n": 0}

        def responder(messages: List[BaseMessage]) -> str:
            counter["n"] += 1
            return f"Revision number {counter['n']}."

        result = make_refiner(
            ScriptedChatModel(responder=responder), max_iterations=3
        ).refine("q", "draft")
        assert result.iterations == 3
        assert result.answer == "Revision number 3."

    def test_reviewer_failure_keeps_the_draft(self) -> None:
        class ExplodingModel(ScriptedChatModel):
            def _generate(self, *args: object, **kwargs: object):
                raise RuntimeError("api down")

        result = make_refiner(ExplodingModel(responder=lambda m: "")).refine(
            "q", "The original draft."
        )
        assert result.answer == "The original draft."
        assert result.improved is False

    def test_context_is_included_in_the_review(self) -> None:
        seen: List[str] = []

        def responder(messages: List[BaseMessage]) -> str:
            seen.append(messages[-1].content)
            return "APPROVED"

        make_refiner(ScriptedChatModel(responder=responder)).refine(
            "closest airport?",
            "DEN.",
            internal_docs=[
                Document(page_content="DEN Denver International", metadata={"filename": "a.pdf"})
            ],
        )
        assert "DEN Denver International" in seen[0]
        assert "DRAFT ANSWER:\nDEN." in seen[0]

    def test_thinking_tags_in_revision_are_stripped(self) -> None:
        refiner = make_refiner(
            scripted("<thinking>weighing...</thinking><answer>Clean revision.</answer>"),
            max_iterations=1,
        )
        result = refiner.refine("q", "messy draft")
        assert result.answer == "Clean revision."

    def test_empty_draft_short_circuits(self) -> None:
        counter = {"n": 0}

        def responder(messages: List[BaseMessage]) -> str:
            counter["n"] += 1
            return "APPROVED"

        result = make_refiner(ScriptedChatModel(responder=responder)).refine("q", "  ")
        assert counter["n"] == 0
        assert result.iterations == 0

    def test_token_usage_accumulates_across_iterations(self) -> None:
        counter = {"n": 0}

        def responder(messages: List[BaseMessage]) -> str:
            counter["n"] += 1
            return "APPROVED" if counter["n"] == 2 else "Better answer."

        result = make_refiner(ScriptedChatModel(responder=responder)).refine("q", "draft")
        # ScriptedChatModel reports 15 total tokens per call; two calls ran.
        assert result.token_usage["total_tokens"] == 30
