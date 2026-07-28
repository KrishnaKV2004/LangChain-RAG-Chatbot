"""Unit tests for the 7LFreight rate client (mocked HTTP).

The client is exercised through a real ``httpx.Client`` backed by a
``MockTransport``, so request building, auth/token flow, retries, response
parsing AND the caching/sort/cap all run for real — only the network is faked.
"""

import json
import time
from pathlib import Path
from typing import List

import httpx
import pytest

from app.cache.ttl_cache import TTLCache
from app.config.settings import Settings
from app.rates.models import FreightItem, Location, RateMode, RateQuery
from app.rates.seven_l import SevenLRates
from app.utils.exceptions import ConfigurationError, RateProviderError


# --------------------------------------------------------------------------- #
# A scripted 7LFreight server
# --------------------------------------------------------------------------- #


class FakeSevenL:
    """Routes MockTransport requests by path; records calls for assertions."""

    def __init__(self) -> None:
        self.calls: List[str] = []            # paths hit, in order
        self.params: dict = {}                # path -> last query params
        self.login_count = 0
        self.login_status = 200               # set 401/403 to simulate bad creds
        self.fail_air_times = 0               # transient 503s to emit first
        self.air_status = 200                 # terminal air status
        self.air_body = None                  # override air response body
        self.unauthorized_first_air = False   # emit one 401 then succeed

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        self.params[path] = dict(request.url.params)

        if path.endswith("/login"):
            self.login_count += 1
            if self.login_status != 200:
                return httpx.Response(
                    self.login_status,
                    json={"error": {"code": self.login_status, "message": "Invalid credentials."}},
                )
            return httpx.Response(
                200, json={"data": {"accessToken": "tok-123", "exp": time.time() + 3600}}
            )
        if path.endswith("/refreshtoken"):
            return httpx.Response(200, json={"data": {"accessToken": "tok-refreshed"}})

        if path.endswith(("/international/airrates", "/expediteair/completeair")):
            if self.unauthorized_first_air:
                self.unauthorized_first_air = False
                return httpx.Response(401, json={"error": "token expired"})
            if self.fail_air_times > 0:
                self.fail_air_times -= 1
                return httpx.Response(503, text="service unavailable")
            if self.air_body is not None:
                return httpx.Response(self.air_status, **self.air_body)
            return httpx.Response(
                self.air_status,
                json={
                    "data": {
                        "results": [
                            {"Name": "FastAir", "SCAC": "FAIR", "Total": "2500.00",
                             "Currency": "USD", "TransitDays": 2},
                            {"Name": "SlowAir", "SCAC": "SAIR", "Total": 1800.5,
                             "TransitDays": 5},
                        ]
                    }
                },
            )
        if path.endswith("/database/ltlaccount"):
            return httpx.Response(200, json={"data": {"results": [
                {"CarrierHash": "h1", "Name": "C1"}, {"CarrierHash": "h2", "Name": "C2"}]}})
        if path.endswith("/ltl/ltlrates"):
            return httpx.Response(200, json={"data": {"results": [
                {"Name": "C-LTL", "Total": 1200, "TransitDays": 3, "RateId": "r1"}]}})
        if path.endswith("/database/lclaccount"):
            return httpx.Response(200, json={"data": {"results": [{"CarrierHash": "o1"}]}})
        if path.endswith("/lcl/oceanrates"):
            return httpx.Response(200, json={"data": {"results": [
                {"Name": "OceanCo", "Total": 900, "Currency": "USD"}]}})
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("7L_USERNAME", "user")
    monkeypatch.setenv("7L_PASSWORD", "pass")
    return Settings(_env_file=None)


def make_client(settings: Settings, server: FakeSevenL, cache: TTLCache = None) -> SevenLRates:
    http = httpx.Client(transport=httpx.MockTransport(server.handler))
    return SevenLRates(settings, cache=cache, client=http)


def air_query(dest_country: str = "US") -> RateQuery:
    return RateQuery(
        mode=RateMode.AIR,
        origin=Location(airport="SFO", country="US"),
        destination=Location(airport="ORD", country=dest_country),
        items=[FreightItem(weight=200, length=48, width=40, height=40)],
    )


def ltl_query() -> RateQuery:
    return RateQuery(
        mode=RateMode.LTL,
        origin=Location(city="Fremont", state="CA", zipcode="94538"),
        destination=Location(city="Chicago", state="IL", zipcode="60601"),
        items=[FreightItem(weight=500, freight_class="100")],
    )


def ocean_query() -> RateQuery:
    return RateQuery(
        mode=RateMode.OCEAN,
        origin=Location(port="USOAK"),
        destination=Location(port="INNSA"),
        items=[FreightItem(weight=1000, weight_type="total")],
    )


@pytest.fixture()
def cache(tmp_path: Path) -> TTLCache:
    return TTLCache(tmp_path / "cache", 60)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


class TestAuth:
    def test_missing_credentials_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("7L_USERNAME", raising=False)
        monkeypatch.delenv("7L_PASSWORD", raising=False)
        with pytest.raises(ConfigurationError, match="7L_USERNAME"):
            SevenLRates(Settings(_env_file=None))

    def test_login_once_and_token_reused_across_calls(self, settings: Settings, tmp_path: Path) -> None:
        server = FakeSevenL()
        client = make_client(settings, server, TTLCache(tmp_path / "c", 3600))
        client.get_rates(air_query())
        client.get_rates(air_query(dest_country="IN"))  # different lane, not cached
        assert server.login_count == 1  # token cached, not re-issued

    def test_401_triggers_relogin_and_retry(self, settings: Settings, tmp_path: Path) -> None:
        server = FakeSevenL()
        server.unauthorized_first_air = True
        result = make_client(settings, server, TTLCache(tmp_path / "c", 3600)).get_rates(air_query())
        assert result.quotes              # succeeded on the retry
        assert server.login_count == 2    # initial login + forced re-login

    def test_invalid_credentials_raise_auth_error(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.login_status = 401         # the real API's response to bad creds
        with pytest.raises(RateProviderError) as excinfo:
            make_client(settings, server).get_rates(air_query())
        # Marked as auth so callers don't tell the user to "try again".
        assert excinfo.value.details.get("kind") == "auth"


# --------------------------------------------------------------------------- #
# Per-mode parsing
# --------------------------------------------------------------------------- #


class TestAirRates:
    def test_parses_and_normalizes(self, settings: Settings) -> None:
        quotes = make_client(settings, FakeSevenL()).get_rates(air_query()).quotes
        assert {q.carrier_name for q in quotes} == {"FastAir", "SlowAir"}
        fast = next(q for q in quotes if q.carrier_name == "FastAir")
        assert fast.price == 2500.0 and fast.currency == "USD" and fast.transit_days == 2
        assert fast.carrier_code == "FAIR" and fast.mode == "air" and fast.provider == "7lfreight"
        assert fast.origin == "SFO" and fast.destination == "ORD"

    def test_domestic_uses_completeair_international_uses_intl(self, settings: Settings) -> None:
        server = FakeSevenL()
        make_client(settings, server).get_rates(air_query(dest_country="US"))
        assert any(c.endswith("/expediteair/completeair") for c in server.calls)

        server2 = FakeSevenL()
        make_client(settings, server2).get_rates(air_query(dest_country="IN"))
        assert any(c.endswith("/international/airrates") for c in server2.calls)

    def test_missing_airport_raises(self, settings: Settings) -> None:
        q = air_query()
        q.destination = Location(airport=None)
        with pytest.raises(RateProviderError, match="airport"):
            make_client(settings, FakeSevenL()).get_rates(q)

    def test_freight_info_serialized_without_class_for_air(self, settings: Settings) -> None:
        server = FakeSevenL()
        make_client(settings, server).get_rates(air_query())
        path = next(c for c in server.calls if "air" in c)
        freight = json.loads(server.params[path]["freightInfo"])
        assert freight[0]["weight"] == "200" and "class" not in freight[0]

    def test_parses_real_nested_vendor_airline_shape(self, settings: Settings) -> None:
        # The real 7L air response nests quotes under Vendors.Airline[lane] with
        # the price at TotalCharges.Total (not a flat {Name, Total} row).
        server = FakeSevenL()
        server.air_body = {"json": {"data": {"results": {"Vendors": {
            "Pickup": [], "Delivery": [],
            "Airline": {"SFO-ORD": [
                {"AirlineName": "United Cargo", "AirlineCode": "UA",
                 "TariffInformation": {"ValidFrom": "2026-01-01", "ValidTo": "2026-12-31",
                                       "Remarks": "Airport to Airport"},
                 "ServiceInformation": {"ServiceType": "General"},
                 "TotalCharges": {"Subtotal": 159, "Screening": 10, "Tax": 9.94, "Total": 195.44}},
                {"AirlineName": "United Cargo", "AirlineCode": "UA",
                 "ServiceInformation": {"ServiceType": "Guaranteed"},
                 "TotalCharges": {"Total": 250.0}},
            ]}}}}}}
        quotes = make_client(settings, server).get_rates(air_query()).quotes
        assert len(quotes) == 2
        cheapest = quotes[0]  # sorted cheapest-first
        assert cheapest.price == 195.44 and cheapest.currency == "USD"
        assert cheapest.carrier_code == "UA"
        assert "United Cargo" in cheapest.carrier_name and "General" in cheapest.carrier_name
        assert cheapest.valid_to == "2026-12-31" and cheapest.remarks == "Airport to Airport"


class TestLtlRates:
    def test_looks_up_carriers_then_quotes_each(self, settings: Settings) -> None:
        server = FakeSevenL()
        quotes = make_client(settings, server).get_rates(ltl_query()).quotes
        assert server.calls.count("/api/v1/database/ltlaccount") == 1
        assert server.calls.count("/api/v1/ltl/ltlrates") == 2   # one per carrier hash
        assert len(quotes) == 2 and all(q.mode == "ltl" for q in quotes)
        assert quotes[0].rate_id == "r1"

    def test_freight_info_includes_class_for_ltl(self, settings: Settings) -> None:
        server = FakeSevenL()
        make_client(settings, server).get_rates(ltl_query())
        freight = json.loads(server.params["/api/v1/ltl/ltlrates"]["freightInfo"])
        assert freight[0]["class"] == "100"

    def test_missing_zip_raises(self, settings: Settings) -> None:
        q = ltl_query()
        q.origin = Location(city="Fremont", state="CA")  # no zipcode
        with pytest.raises(RateProviderError, match="city"):
            make_client(settings, FakeSevenL()).get_rates(q)


class TestOceanRates:
    def test_uses_lcl_account_and_ports(self, settings: Settings) -> None:
        server = FakeSevenL()
        quotes = make_client(settings, server).get_rates(ocean_query()).quotes
        assert any(c.endswith("/database/lclaccount") for c in server.calls)
        assert any(c.endswith("/lcl/oceanrates") for c in server.calls)
        assert quotes[0].carrier_name == "OceanCo" and quotes[0].mode == "ocean"


# --------------------------------------------------------------------------- #
# Resilience / error normalization
# --------------------------------------------------------------------------- #


class TestResilience:
    def test_transient_5xx_is_retried_then_succeeds(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.fail_air_times = 2  # two 503s, then the real 200
        result = make_client(settings, server).get_rates(air_query())
        assert result.quotes
        assert sum(c.endswith("completeair") for c in server.calls) == 3

    def test_persistent_5xx_becomes_rate_provider_error(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.fail_air_times = 99
        with pytest.raises(RateProviderError):
            make_client(settings, server).get_rates(air_query())

    def test_4xx_is_normalized(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.air_status = 400
        server.air_body = {"json": {"error": "bad request"}}
        with pytest.raises(RateProviderError, match="400"):
            make_client(settings, server).get_rates(air_query())

    def test_non_json_body_is_normalized(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.air_body = {"text": "not json at all"}
        with pytest.raises(RateProviderError, match="non-JSON"):
            make_client(settings, server).get_rates(air_query())

    def test_row_without_total_is_skipped(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.air_body = {"json": {"data": {"results": [
            {"Name": "NoPrice"}, {"Name": "HasPrice", "Total": 100}]}}}
        quotes = make_client(settings, server).get_rates(air_query()).quotes
        assert [q.carrier_name for q in quotes] == ["HasPrice"]


# --------------------------------------------------------------------------- #
# Caching / sorting / capping (folded into the client)
# --------------------------------------------------------------------------- #


class TestCachingAndOrdering:
    def test_quotes_sorted_cheapest_first(self, settings: Settings) -> None:
        server = FakeSevenL()
        server.air_body = {"json": {"data": {"results": [
            {"Name": "Pricey", "Total": 3000},
            {"Name": "Cheap", "Total": 900},
            {"Name": "Mid", "Total": 1500}]}}}
        quotes = make_client(settings, server).get_rates(air_query()).quotes
        assert [q.price for q in quotes] == [900, 1500, 3000]

    def test_capped_to_max_carriers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("7L_USERNAME", "user")
        monkeypatch.setenv("7L_PASSWORD", "pass")
        monkeypatch.setenv("RATES__MAX_CARRIERS_RETURNED", "3")
        server = FakeSevenL()
        server.air_body = {"json": {"data": {"results": [
            {"Name": f"C{i}", "Total": 1000 + i} for i in range(10)]}}}
        quotes = make_client(Settings(_env_file=None), server).get_rates(air_query()).quotes
        assert len(quotes) == 3

    def test_second_identical_lookup_hits_cache(self, settings: Settings, cache: TTLCache) -> None:
        server = FakeSevenL()
        client = make_client(settings, server, cache)
        first = client.get_rates(air_query())
        second = client.get_rates(air_query())  # same lane → served from cache
        assert first.cache_hit is False and second.cache_hit is True
        assert sum(c.endswith("completeair") for c in server.calls) == 1  # network hit once

    def test_different_lane_is_a_cache_miss(self, settings: Settings, cache: TTLCache) -> None:
        server = FakeSevenL()
        client = make_client(settings, server, cache)
        client.get_rates(air_query())
        client.get_rates(RateQuery(mode=RateMode.AIR, origin=Location(airport="JFK"),
                                   destination=Location(airport="LAX"),
                                   items=[FreightItem(weight=200)]))
        assert sum(c.endswith("completeair") for c in server.calls) == 2
