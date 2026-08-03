"""Unit tests for street-address standardization (mocked Census geocoder)."""

from pathlib import Path

import httpx
import pytest

from app.cache.ttl_cache import TTLCache
from app.rates.address import make_address_resolver, standardize_street
from app.rates.models import Location

# Shape copied from a live Census response.
MATCH = {
    "result": {
        "addressMatches": [
            {
                "matchedAddress": "1500 ATLANTIC ST, UNION CITY, CA, 94587",
                "coordinates": {"x": -122.0423, "y": 37.6017},
                "addressComponents": {
                    "streetName": "ATLANTIC", "suffixType": "ST",
                    "city": "UNION CITY", "state": "CA", "zip": "94587",
                },
            }
        ]
    }
}
NO_MATCH = {"result": {"addressMatches": []}}


class FakeGeocoder:
    def __init__(self, payload=None, status: int = 200, boom: bool = False) -> None:
        self.payload = payload if payload is not None else MATCH
        self.status = status
        self.boom = boom
        self.queries = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            self.queries.append(request.url.params.get("address"))
            if self.boom:
                raise httpx.ConnectError("network down")
            return httpx.Response(self.status, json=self.payload)

        return httpx.Client(transport=httpx.MockTransport(handler))


class TestStandardizeStreet:
    def test_corrects_and_completes_the_address(self) -> None:
        geo = FakeGeocoder()
        got = standardize_street("1500 atlantic street", "Union City", "CA", client=geo.client())
        assert got["address1"] == "1500 Atlantic St"      # spelling normalized
        assert (got["city"], got["state"], got["zipcode"]) == ("Union City", "CA", "94587")

    def test_secondary_unit_is_preserved(self) -> None:
        """The geocoder drops "Suite A"; delivery needs it, so we keep it."""
        geo = FakeGeocoder()
        got = standardize_street("4835 Decatur Blvd Suite A", "Indianapolis", "IN",
                                 client=geo.client())
        assert got["address2"] == "Suite A"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1411 S 47th Ave #140", "#140"), ("5 Main St Apt 3B", "Apt 3B"),
         ("9 Oak Rd, Unit 12", "Unit 12"), ("1500 Atlantic St", None)],
    )
    def test_unit_variants(self, raw: str, expected) -> None:
        got = standardize_street(raw, "X", "CA", client=FakeGeocoder().client())
        assert got["address2"] == expected

    @pytest.mark.parametrize(
        ("matched", "expected"),
        [
            # str.title() corrupts both of these; a driver has to read them.
            ("152 SELIG DR SW, ATLANTA, GA, 30336", "152 Selig Dr SW"),
            ("1411 S 47TH AVE, PHOENIX, AZ, 85043", "1411 S 47th Ave"),
            ("100 NE 1ST ST, MIAMI, FL, 33132", "100 NE 1st St"),
            ("9 W 3RD AVE, DENVER, CO, 80223", "9 W 3rd Ave"),
            ("1500 ATLANTIC ST, UNION CITY, CA, 94587", "1500 Atlantic St"),
        ],
    )
    def test_directionals_and_ordinals_are_not_mangled(self, matched: str, expected: str) -> None:
        payload = {"result": {"addressMatches": [
            {"matchedAddress": matched, "addressComponents": {
                "city": "X", "state": "CA", "zip": "00000"}}]}}
        got = standardize_street("whatever", client=FakeGeocoder(payload).client())
        assert got["address1"] == expected

    def test_no_match_returns_none(self) -> None:
        got = standardize_street("999999 Nowhere Rd", client=FakeGeocoder(NO_MATCH).client())
        assert got is None

    def test_network_failure_returns_none(self) -> None:
        got = standardize_street("1500 Atlantic St", client=FakeGeocoder(boom=True).client())
        assert got is None

    def test_blank_address_skips_the_call(self) -> None:
        geo = FakeGeocoder()
        assert standardize_street("  ", client=geo.client()) is None
        assert geo.queries == []

    def test_result_is_cached(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "c", 60)
        geo = FakeGeocoder()
        for _ in range(2):
            standardize_street("1500 Atlantic St", "Union City", "CA",
                               cache=cache, client=geo.client())
        assert len(geo.queries) == 1

    def test_misses_are_cached_too(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "c", 60)
        geo = FakeGeocoder(NO_MATCH)
        for _ in range(2):
            standardize_street("999999 Nowhere Rd", cache=cache, client=geo.client())
        assert len(geo.queries) == 1


class TestAddressResolver:
    """One hook: street → geocoder, otherwise the city/zip fallback."""

    @staticmethod
    def _zip_resolver(**expected):
        calls = []

        def resolve(city=None, state=None, zipcode=None):
            calls.append({"city": city, "state": state, "zipcode": zipcode})
            return Location(city="Union City", state="CA", zipcode="94587")

        resolve.calls = calls
        return resolve

    def test_street_bypasses_the_zip_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "app.rates.address.standardize_street",
            lambda *a, **k: {"address1": "1500 Atlantic St", "address2": None,
                             "city": "Union City", "state": "CA", "zipcode": "94587"},
        )
        zips = self._zip_resolver()
        got = make_address_resolver(zips)(address1="1500 Atlantic St", city="Union City", state="CA")
        assert got["address1"] == "1500 Atlantic St"
        assert zips.calls == []                    # geocoder answered it

    def test_falls_back_to_zip_lookup_without_a_street(self) -> None:
        zips = self._zip_resolver()
        got = make_address_resolver(zips)(city="Union City", state="CA")
        assert got["zipcode"] == "94587"
        assert zips.calls == [{"city": "Union City", "state": "CA", "zipcode": None}]
        # No street was given, so none may be invented.
        assert got["address1"] is None and got["address2"] is None

    def test_no_street_means_no_geocoder_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A street is only ever standardized, never conjured from a city."""
        calls = []
        monkeypatch.setattr(
            "app.rates.address.standardize_street",
            lambda *a, **k: calls.append(a) or {"address1": "123 Invented St"},
        )
        got = make_address_resolver(self._zip_resolver())(city="Union City", state="CA")
        assert calls == []
        assert got["address1"] is None

    def test_falls_back_when_the_street_cannot_be_matched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.rates.address.standardize_street", lambda *a, **k: None)
        zips = self._zip_resolver()
        got = make_address_resolver(zips)(address1="999999 Nowhere Rd", city="Union City", state="CA")
        assert got["zipcode"] == "94587"           # city/state still resolved
        assert len(zips.calls) == 1

    def test_returns_none_when_nothing_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rates.address.standardize_street", lambda *a, **k: None)
        assert make_address_resolver(lambda **kw: None)(city="Nowhere", state="ZZ") is None
