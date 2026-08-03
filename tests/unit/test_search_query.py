"""Unit tests for the web-search query refiner (scripted LLM)."""

from langchain_core.outputs import ChatResult

from app.chains.search_query import SearchQueryRefiner
from tests.unit.test_router_chains import ScriptedChatModel

VERBOSE = "What is the address of the nearest air india cargo dropoff to 1500 Atlantic st, Union City, CA?"


def refiner(reply: str) -> SearchQueryRefiner:
    return SearchQueryRefiner(ScriptedChatModel(responder=lambda messages: reply))


def test_returns_focused_query() -> None:
    out = refiner("Air India cargo drop-off San Francisco Bay Area address").refine(VERBOSE)
    assert out == "Air India cargo drop-off San Francisco Bay Area address"


def test_strips_quotes_and_extra_lines() -> None:
    out = refiner('"Dubai weather today"\n(here is your query)').refine("weather in Dubai?")
    assert out == "Dubai weather today"


def test_empty_reply_falls_back_to_raw() -> None:
    assert refiner("   ").refine(VERBOSE) == VERBOSE


def test_overlong_reply_falls_back_to_raw() -> None:
    assert refiner("word " * 60).refine(VERBOSE) == VERBOSE


def test_llm_failure_falls_back_to_raw() -> None:
    class Exploding(ScriptedChatModel):
        def _generate(self, *args: object, **kwargs: object) -> ChatResult:
            raise RuntimeError("api down")

    out = SearchQueryRefiner(Exploding(responder=lambda m: "")).refine(VERBOSE)
    assert out == VERBOSE
