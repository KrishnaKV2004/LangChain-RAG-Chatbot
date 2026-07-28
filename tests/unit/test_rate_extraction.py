"""Unit tests for the NL → RateQuery extraction chain (scripted LLM)."""

import json

import pytest

from app.chains.rate_extraction import RateExtractor
from app.rates.models import RateMode
from tests.unit.test_router_chains import ScriptedChatModel


def extractor(reply: str) -> RateExtractor:
    return RateExtractor(ScriptedChatModel(responder=lambda messages: reply))


def payload(**overrides) -> str:
    base = {
        "mode": "air",
        "origin": {"airport": "SFO"},
        "destination": {"airport": "ORD"},
        "items": [{"weight": 200, "length": 48, "width": 40, "height": 48}],
        "uom": "US",
        "ready": True,
        "clarification": None,
    }
    base.update(overrides)
    return json.dumps(base)


class TestExtraction:
    def test_air_query_extracted(self) -> None:
        result = extractor(payload()).extract("air freight 200kg SFO to ORD")
        assert result.ok
        assert result.query.mode is RateMode.AIR
        assert result.query.origin.airport == "SFO"
        assert result.query.destination.airport == "ORD"
        assert result.query.items[0].weight == 200.0

    def test_reply_wrapped_in_code_fence_is_parsed(self) -> None:
        result = extractor("Sure!\n```json\n" + payload() + "\n```").extract("q")
        assert result.ok and result.query.destination.airport == "ORD"

    def test_ltl_query_extracted_with_class(self) -> None:
        raw = payload(
            mode="ltl",
            origin={"city": "Fremont", "state": "CA", "zipcode": "94538"},
            destination={"city": "Chicago", "state": "IL", "zipcode": "60601"},
            items=[{"weight": 500, "qty": 2, "freight_class": "100",
                    "length": 48, "width": 40, "height": 48}],
        )
        result = extractor(raw).extract("ltl quote")
        assert result.ok and result.query.mode is RateMode.LTL
        assert result.query.items[0].freight_class == "100"
        assert result.query.origin.label() == "Fremont, CA, 94538"

    def test_ocean_query_extracted(self) -> None:
        raw = payload(mode="ocean", origin={"port": "USOAK"}, destination={"port": "INNSA"},
                      items=[{"weight": 1000, "length": 120, "width": 100, "height": 100}])
        result = extractor(raw).extract("ocean quote")
        assert result.ok and result.query.mode is RateMode.OCEAN
        assert result.query.origin.port == "USOAK"

    def test_metric_uom_detected(self) -> None:
        result = extractor(payload(uom="METRIC")).extract("200 kg SFO to ORD")
        assert result.query.uom == "METRIC"

    def test_mode_inferred_from_airports_when_model_omits_it(self) -> None:
        # A terse follow-up drops the mode but keeps the airports → infer AIR.
        result = extractor(payload(mode="")).extract("SFO to ORD, 200 kg, 48x40x48")
        assert result.ok and result.query.mode is RateMode.AIR

    def test_mode_inferred_from_city_for_domestic_follow_up(self) -> None:
        # Models the screenshot's "10 kgs" follow-up: lane carried over, no mode.
        raw = payload(mode="", origin={"city": "San Francisco", "state": "CA", "zipcode": "94107"},
                      destination={"city": "Denver", "state": "CO", "zipcode": "80202"},
                      items=[{"weight": 10, "length": 40, "width": 40, "height": 40}])
        result = extractor(raw).extract("10 kgs")
        assert result.ok and result.query.mode is RateMode.LTL


class TestClarification:
    def test_missing_weight_asks(self) -> None:
        result = extractor(payload(items=[])).extract("air freight SFO to ORD")
        assert not result.ok
        assert "weight" in result.clarification.lower()

    def test_air_without_dimensions_asks(self) -> None:
        # 7L air rates 400 without L×W×H, so we must ask rather than fail the call.
        result = extractor(payload(items=[{"weight": 200}])).extract("air SFO to ORD 200 kg")
        assert not result.ok
        assert "dimension" in result.clarification.lower()

    def test_clarification_acknowledges_the_known_lane(self) -> None:
        # The key UX fix: don't re-ask for the origin/destination already given.
        result = extractor(payload(items=[])).extract("air freight SFO to ORD")
        assert "SFO" in result.clarification and "ORD" in result.clarification
        assert "weight" in result.clarification.lower()

    def test_missing_endpoint_asks_even_if_model_said_ready(self) -> None:
        # Model claims ready=true but a required airport is absent; we still ask.
        result = extractor(payload(destination={"airport": None})).extract("quote from SFO")
        assert not result.ok
        assert "airport" in result.clarification.lower()

    def test_model_clarification_used_only_when_nothing_specific_missing(self) -> None:
        # Full valid query, but the model wants to confirm an optional detail:
        # here (and only here) its own clarification is used verbatim.
        raw = payload(ready=False, clarification="Do you want expedited service?")
        result = extractor(raw).extract("air freight SFO to ORD, 200 kg")
        assert result.clarification == "Do you want expedited service?"

    def test_invalid_mode_and_no_lane_asks_generic(self) -> None:
        # No mode AND no address to infer from → the generic ask.
        raw = payload(mode="teleport", origin={}, destination={}, items=[])
        result = extractor(raw).extract("beam it over")
        assert not result.ok and result.clarification

    def test_garbage_reply_yields_clarification(self) -> None:
        result = extractor("I have no idea what you mean").extract("???")
        assert not result.ok and result.clarification

    def test_llm_exception_yields_clarification(self) -> None:
        from langchain_core.outputs import ChatResult

        class Exploding(ScriptedChatModel):
            def _generate(self, *args: object, **kwargs: object) -> ChatResult:
                raise RuntimeError("api down")

        result = RateExtractor(Exploding(responder=lambda m: "")).extract("quote")
        assert not result.ok and result.clarification
