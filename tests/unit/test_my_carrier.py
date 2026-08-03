"""Unit tests for the MyCarrier LTL client and door-to-door planning.

The client runs against a real ``httpx.Client`` backed by ``MockTransport``, so
body building and response parsing are exercised for real — only the network is
faked. Response fixtures mirror the live shapes exactly (nested ``price[]``,
mixed available/rejected).
"""

from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from app.config.settings import Settings
from app.rates.door_to_door import bid, nearest_warehouse, plan_door_to_door
from app.rates.models import FreightItem, Location, RateMode, RateQuery, RateQuote, RateResult
from app.rates.my_carrier import MyCarrierRates
from app.utils.exceptions import RateProviderError

# Shape copied from a live response: rejected + available, prices nested in price[].
RATING_RESPONSE = {
    "data": {
        "rates": [
            {
                "quoteStatus": "rejected",
                "rejectReason": "Carrier Service Level Error",
                "currencyCode": "USD",
                "carrier": {"name": "SEFL", "scac": "SEFL"},
                "price": [{"serviceType": {"code": "GSNOON"}, "grossPrice": "0", "netPrice": "0"}],
            },
            {
                "quoteStatus": "available",
                "quoteId": "CXG72EDGC",
                "currencyCode": "USD",
                "quoteExpiryDate": "08/03/2026 00:00:00",
                "carrier": {"name": "SEFL", "scac": "SEFL"},
                "price": [
                    {
                        "serviceType": {"code": "STD", "description": "Standard", "serviceDays": "6"},
                        "quoteId": "CXG72EDGC",
                        "grossPrice": "214.51",
                        "netPrice": "214.51",
                        "priceBreakUp": [{"code": "FSC", "price": "41.41"}],
                    }
                ],
            },
            {
                "quoteStatus": "available",
                "currencyCode": "USD",
                "carrier": {"name": "SAIA", "scac": "SAIA"},
                "price": [
                    {
                        "serviceType": {"code": "STD", "description": "Standard", "serviceDays": "4"},
                        "netPrice": "128.29",
                    },
                    {
                        "serviceType": {"code": "GUR", "description": "Guaranteed", "serviceDays": "4"},
                        "netPrice": "398.29",
                    },
                ],
            },
        ]
    }
}


class FakeRating:
    """Captures the posted body and replays a scripted response."""

    def __init__(self, response: Dict[str, Any] = None, status: int = 200) -> None:
        self.response = response if response is not None else RATING_RESPONSE
        self.status = status
        self.bodies: List[Dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"title": "validation error"})
        return httpx.Response(200, json=self.response)


def client(server: FakeRating, **env: str) -> MyCarrierRates:
    http = httpx.Client(transport=httpx.MockTransport(server.handler))
    return MyCarrierRates(Settings(_env_file=None), cache=None, client=http)


def ltl_query(**overrides: Any) -> RateQuery:
    item = overrides.pop("item", FreightItem(weight=100, length=40, width=30, height=20,
                                             freight_class="250"))
    return RateQuery(
        mode=RateMode.LTL,
        origin=Location(city="Union City", state="CA", zipcode="94587"),
        destination=Location(city="Denver", state="CO", zipcode="80249"),
        items=[item],
        **overrides,
    )


class TestParsing:
    def test_only_available_quotes_parsed_cheapest_first(self) -> None:
        result = client(FakeRating()).get_rates(ltl_query())
        assert [q.price for q in result.quotes] == [128.29, 214.51, 398.29]
        assert all(q.provider == "mycarrier" for q in result.quotes)

    def test_carrier_and_service_level_in_name(self) -> None:
        quotes = client(FakeRating()).get_rates(ltl_query()).quotes
        cheapest = quotes[0]
        assert cheapest.carrier_name == "SAIA — Standard"
        assert cheapest.carrier_code == "SAIA"
        assert cheapest.transit_days == 4

    def test_quote_id_expiry_and_breakdown_kept(self) -> None:
        quotes = client(FakeRating()).get_rates(ltl_query()).quotes
        sefl = next(q for q in quotes if q.carrier_code == "SEFL")
        assert sefl.rate_id == "CXG72EDGC"
        assert sefl.valid_to == "08/03/2026 00:00:00"
        assert sefl.breakdown["priceBreakUp"] == [{"code": "FSC", "price": "41.41"}]

    def test_zero_priced_rows_dropped(self) -> None:
        server = FakeRating({"data": {"rates": [
            {"quoteStatus": "available", "carrier": {"name": "X", "scac": "X"},
             "price": [{"serviceType": {"code": "STD"}, "netPrice": "0"}]}]}})
        assert client(server).get_rates(ltl_query()).is_empty


class TestRequestBody:
    def test_dimensions_are_ints(self) -> None:
        """MyCarrier types dimensions as Int32 — floats are a 400."""
        server = FakeRating()
        client(server).get_rates(ltl_query())
        dims = server.bodies[0]["data"]["shipment"]["shipmentLineItems"][0]["dimensions"]
        for key in ("height", "length", "width", "weight", "quantity"):
            assert isinstance(dims[key], int), f"{key} must be int, got {type(dims[key])}"

    def test_metric_converted_to_inches_and_pounds(self) -> None:
        server = FakeRating()
        item = FreightItem(weight=100, length=100, width=80, height=90)  # kg / cm
        client(server).get_rates(ltl_query(item=item, uom="METRIC"))
        dims = server.bodies[0]["data"]["shipment"]["shipmentLineItems"][0]["dimensions"]
        assert dims["weight"] == 220           # 100 kg → lb
        assert dims["length"] == 39            # 100 cm → in
        assert dims["heightUOM"] == "INCHES" and dims["weightUOM"] == "LB"

    def test_total_weight_split_per_piece(self) -> None:
        server = FakeRating()
        item = FreightItem(weight=400, qty=4, weight_type="total", length=40, width=30, height=20)
        client(server).get_rates(ltl_query(item=item))
        dims = server.bodies[0]["data"]["shipment"]["shipmentLineItems"][0]["dimensions"]
        assert dims["weight"] == 100 and dims["quantity"] == 4

    def test_stops_carry_city_state_zip(self) -> None:
        server = FakeRating()
        client(server).get_rates(ltl_query())
        pickup, drop = server.bodies[0]["data"]["shipment"]["stops"]
        assert (pickup["stopType"], pickup["postalCode"]) == ("PICKUP", "94587")
        assert (drop["stopType"], drop["postalCode"]) == ("DROP", "80249")
        assert pickup["country"]["alpha2Code"] == "US"

    def test_street_lines_are_sent_when_known(self) -> None:
        server = FakeRating()
        query = ltl_query()
        query.origin.address1, query.origin.address2 = "1500 Atlantic St", "Suite A"
        client(server).get_rates(query)
        pickup = server.bodies[0]["data"]["shipment"]["stops"][0]
        assert pickup["addressLine1"] == "1500 Atlantic St"
        assert pickup["addressLine2"] == "Suite A"

    def test_street_lines_are_blank_when_unknown(self) -> None:
        """No street given → send empty strings, never a fabricated address."""
        server = FakeRating()
        client(server).get_rates(ltl_query())      # city/state/zip only
        for stop in server.bodies[0]["data"]["shipment"]["stops"]:
            assert stop["addressLine1"] == "" and stop["addressLine2"] == ""

    def test_missing_zip_raises(self) -> None:
        query = ltl_query()
        query.destination = Location(city="Denver", state="CO")  # no zip
        with pytest.raises(RateProviderError, match="postal code"):
            client(FakeRating()).get_rates(query)

    def test_http_error_normalized(self) -> None:
        with pytest.raises(RateProviderError, match="400"):
            client(FakeRating(status=400)).get_rates(ltl_query())


# --------------------------------------------------------------------------- #
# Warehouse whitelist + door-to-door
# --------------------------------------------------------------------------- #

CSV = str(Path("documents/Warehouses.csv"))


class TestWarehouseLookup:
    def test_picks_bay_area_warehouse_for_sfo(self) -> None:
        warehouse = nearest_warehouse("94128", CSV)  # SFO
        assert warehouse["city"] == "Union City" and warehouse["state"] == "CA"

    def test_picks_denver_area_warehouse_for_den(self) -> None:
        warehouse = nearest_warehouse("80249", CSV)  # DEN
        assert warehouse["city"] == "Aurora" and warehouse["state"] == "CO"

    def test_missing_csv_returns_none(self) -> None:
        assert nearest_warehouse("94128", "does/not/exist.csv") is None


class FakeSevenL:
    """Air leg + airport resolution, and (like the real client) LTL bids too."""

    name = "7lfreight"

    def __init__(self, air_price: float = 108.43, fail: bool = False,
                 truck_price: float = None) -> None:
        self.air_price = air_price
        self.fail = fail
        self.truck_price = truck_price

    def airport_location(self, code: str):
        # Mirrors the real client: unknown code → None, never an exception.
        return {
            "SFO": Location(city="San Francisco", state="CA", zipcode="94128"),
            "DEN": Location(city="Denver", state="CO", zipcode="80249"),
        }.get(code)

    def get_rates(self, query: RateQuery) -> RateResult:
        if self.fail:
            raise RateProviderError("7L down")
        if query.mode is RateMode.LTL:
            if self.truck_price is None:
                return RateResult()
            return RateResult(quotes=[RateQuote(
                carrier_name="Estes", carrier_code="EXLA", origin=query.origin.label(),
                destination=query.destination.label(), price=self.truck_price,
                currency="USD", mode="ltl", provider="7lfreight", transit_days=3)])
        return RateResult(quotes=[RateQuote(
            carrier_name="United Cargo — General", carrier_code="UA", origin="SFO",
            destination="DEN", price=self.air_price, currency="USD", mode="air",
            provider="7lfreight")])


class FakeMyCarrier:
    name = "mycarrier"

    def __init__(self, price: float = 129.32, fail: bool = False) -> None:
        self.price = price
        self.fail = fail
        self.lanes: List[str] = []

    def get_rates(self, query: RateQuery) -> RateResult:
        if self.fail:
            raise RateProviderError("MyCarrier down")
        self.lanes.append(f"{query.origin.label()}>{query.destination.label()}")
        return RateResult(quotes=[RateQuote(
            carrier_name="FWDN — Standard", carrier_code="FWDN",
            origin=query.origin.label(), destination=query.destination.label(),
            price=self.price, currency="USD", mode="ltl", provider="mycarrier",
            transit_days=1)])


def air_query() -> RateQuery:
    return RateQuery(mode=RateMode.AIR, origin=Location(airport="SFO"),
                     destination=Location(airport="DEN"),
                     items=[FreightItem(weight=100, length=40, width=30, height=20)])


class TestDoorToDoor:
    def test_three_legs_totalled(self) -> None:
        mc = FakeMyCarrier(price=100.0)
        plan = plan_door_to_door(air_query(), FakeSevenL(air_price=50.0), mc, CSV)
        assert len(plan.legs) == 3 and plan.complete
        assert plan.total == 250.0          # 100 truck + 50 air + 100 truck
        assert plan.currency == "USD"

    def test_truck_legs_run_warehouse_to_airport_and_back(self) -> None:
        mc = FakeMyCarrier()
        plan_door_to_door(air_query(), FakeSevenL(), mc, CSV)
        # Leg 1 starts at the warehouse and ends at the airport; leg 3 reverses.
        assert mc.lanes[0].startswith("Union City") and "94128" in mc.lanes[0]
        assert mc.lanes[1].startswith("Denver") and "Aurora" in mc.lanes[1]

    def test_failed_truck_leg_gives_partial_subtotal(self) -> None:
        plan = plan_door_to_door(air_query(), FakeSevenL(air_price=50.0),
                                 FakeMyCarrier(fail=True), CSV)
        assert not plan.complete
        assert plan.total == 50.0                       # air only
        assert sum(leg.quote is None for leg in plan.legs) == 2
        assert all(leg.note for leg in plan.legs if leg.quote is None)

    def test_no_legs_priced_has_no_total(self) -> None:
        plan = plan_door_to_door(air_query(), FakeSevenL(fail=True),
                                 FakeMyCarrier(fail=True), CSV)
        assert plan.total is None and not plan.complete

    def test_without_my_carrier_truck_legs_are_not_priced(self) -> None:
        # MyCarrier owns the truck legs; 7L is not substituted even though it
        # can quote LTL.
        plan = plan_door_to_door(air_query(), FakeSevenL(truck_price=90.0), None, CSV)
        assert not plan.complete
        assert plan.legs[0].quote is None and plan.legs[2].quote is None
        assert plan.legs[1].source == "7LFreight"


class TestLegOwnership:
    """MyCarrier prices warehouse↔airport; 7LFreight prices airport↔airport."""

    def test_truck_legs_come_from_my_carrier_only(self) -> None:
        # 7L would undercut at 90 vs MyCarrier's 129.32, but must not be asked.
        plan = plan_door_to_door(
            air_query(), FakeSevenL(air_price=50.0, truck_price=90.0),
            FakeMyCarrier(price=129.32), CSV
        )
        for leg in (plan.legs[0], plan.legs[2]):
            assert leg.source == "MyCarrier"
            assert leg.quote.price == 129.32
            assert leg.providers == ("MyCarrier",)
        assert plan.total == 308.64                      # 129.32 + 50 + 129.32

    def test_air_leg_comes_from_7l_only(self) -> None:
        plan = plan_door_to_door(air_query(), FakeSevenL(), FakeMyCarrier(), CSV)
        assert plan.legs[1].source == "7LFreight"
        assert plan.legs[1].providers == ("7LFreight",)

    def test_unsupported_gateway_is_refused(self) -> None:
        query = air_query()
        query.origin = Location(airport="SNA")           # not in Airports.pdf
        plan = plan_door_to_door(query, FakeSevenL(), FakeMyCarrier(), CSV)
        air = plan.legs[1]
        assert air.quote is None and "not a supported gateway" in air.note


class TestSingleLaneCompetition:
    """An explicit LTL-only lane CAN be quoted by both, so they compete there."""

    def test_cheapest_across_providers_wins(self) -> None:
        quotes = bid(ltl_query(), [FakeSevenL(truck_price=90.0), FakeMyCarrier(price=129.32)])
        assert [q.price for q in quotes] == [90.0, 129.32]     # sorted cheapest-first
        assert quotes[0].provider == "7lfreight"

    def test_one_provider_failing_does_not_sink_the_lane(self) -> None:
        quotes = bid(ltl_query(), [FakeSevenL(fail=True), FakeMyCarrier(price=129.32)])
        assert [q.provider for q in quotes] == ["mycarrier"]

    def test_no_providers_returns_nothing(self) -> None:
        assert bid(ltl_query(), [None]) == []
