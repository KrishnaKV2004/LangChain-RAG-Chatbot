"""7LFreight rate client — the whole feature in one file.

Given a :class:`RateQuery` it returns a :class:`RateResult` (quotes + cache
hit + latency). It owns everything the rest of the app is kept unaware of:

* **Auth** — 7L issues a short-lived JWT from username/password and caps
  ``/login`` to a *daily* quota, so the token is cached on disk (via the
  injected :class:`~app.cache.ttl_cache.TTLCache`) and renewed with
  ``/refreshtoken``; only a hard failure falls back to ``/login``. A ``401``
  mid-request forces one re-login and retry.
* **Carriers** — LTL/ocean rates are quoted per carrier (a ``CarrierHash``
  from ``/database/ltlaccount`` / ``/database/lclaccount``, cached). Air needs
  no carrier hash.
* **Caching / sorting** — identical lookups are served from a short-TTL cache;
  quotes come back cheapest-first, capped to ``max_carriers_returned``.

All failures surface as :class:`~app.utils.exceptions.RateProviderError`;
transient ones (network, 5xx, 429) are retried first.
"""

import json
import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.cache.ttl_cache import TTLCache
from app.config.settings import Settings
from app.rates.models import Location, RateMode, RateQuery, RateQuote, RateResult
from app.utils.exceptions import ConfigurationError, RateProviderError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: Keep the cached token far longer than its own lifetime so a stale entry can
#: still seed a ``/refreshtoken`` call instead of a quota-consuming login.
_TOKEN_RETENTION_SECONDS = 30 * 24 * 3600
#: Carrier lists rarely change; cache them for an hour.
_CARRIER_CACHE_SECONDS = 3600
#: LTL requires a pickup date; default this many days out when unspecified.
_DEFAULT_PICKUP_OFFSET_DAYS = 3


class _RetryableRateError(Exception):
    """Internal marker for transient failures worth retrying (5xx / network)."""


class SevenLRates:
    """The 7LFreight rate client (auth + carriers + fetch + cache, in one)."""

    name = "7lfreight"

    def __init__(
        self,
        settings: Settings,
        cache: Optional[TTLCache] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """``cache`` persists the token, carrier lists and quotes (prefixed
        keys). ``client`` is injectable for tests; production builds its own."""
        self._settings = settings.rates
        self._username = settings.seven_l_username
        self._password = (
            settings.seven_l_password.get_secret_value()
            if settings.seven_l_password is not None
            else None
        )
        if not self._username or not self._password:
            raise ConfigurationError(
                "7L_USERNAME and 7L_PASSWORD are required for freight rates. "
                "Set them in your environment or .env file."
            )
        self._base = settings.rates.base_url.rstrip("/") + settings.rates.api_prefix
        self._cache = cache
        self._token_key = f"7l:token:{self._username}"
        self._mem_token: Optional[Dict[str, Any]] = None
        self._client = client or httpx.Client(timeout=settings.rates.timeout_seconds)

    # ------------------------------------------------------------------ #
    # Public entry point: cache → fetch → sort → cap
    # ------------------------------------------------------------------ #

    def get_rates(self, query: RateQuery) -> RateResult:
        """Return cheapest-first quotes for ``query``, cached where possible.

        Raises :class:`RateProviderError` on failure; the caller decides how to
        degrade.
        """
        cache_key = f"rates:{self.name}:{query.cache_key()}"
        result = RateResult()
        with Timer() as timer:
            cached = self._cache.get(cache_key) if self._cache is not None else None
            if cached is not None:
                result.quotes = [RateQuote.from_dict(item) for item in cached]
                result.cache_hit = True
            else:
                quotes = self._fetch(query)
                quotes.sort(key=lambda q: q.price)  # cheapest first
                quotes = quotes[: self._settings.max_carriers_returned]
                if self._cache is not None:
                    self._cache.set(
                        cache_key,
                        [q.to_dict() for q in quotes],
                        ttl_seconds=self._settings.cache_ttl_seconds,
                    )
                result.quotes = quotes
        result.latency_ms = timer.elapsed_ms
        logger.info(
            "rate_lookup_complete",
            mode=query.mode.value,
            lane=f"{query.origin.label()}>{query.destination.label()}",
            quotes=len(result.quotes),
            cache_hit=result.cache_hit,
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    def _fetch(self, query: RateQuery) -> List[RateQuote]:
        """Dispatch to the right endpoint by mode, normalizing all errors."""
        try:
            if query.mode is RateMode.AIR:
                return self._air_rates(query)
            if query.mode is RateMode.LTL:
                return self._ltl_rates(query)
            if query.mode is RateMode.OCEAN:
                return self._ocean_rates(query)
            raise RateProviderError(f"Unsupported rate mode: {query.mode}")
        except RateProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize provider errors
            raise RateProviderError(f"7LFreight rate lookup failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Per-mode rate requests
    # ------------------------------------------------------------------ #

    def _air_rates(self, query: RateQuery) -> List[RateQuote]:
        origin, destination = query.origin, query.destination
        if not origin.airport or not destination.airport:
            raise RateProviderError(
                "Air rates require origin and destination airport IATA codes"
            )
        # Different countries ⇒ international endpoint; same country ⇒ domestic.
        international = (origin.country or "US").upper() != (destination.country or "US").upper()
        path = "/international/airrates" if international else "/expediteair/completeair"
        params = {
            "originAirport": origin.airport.upper(),
            "destinationAirport": destination.airport.upper(),
            "UOM": query.uom,
            "freightInfo": self._freight_info(query, include_class=False),
        }
        data = self._get(path, params)
        return self._parse_quotes(self._flatten_entries(data), query, RateMode.AIR)

    def _ltl_rates(self, query: RateQuery) -> List[RateQuote]:
        origin, destination = query.origin, query.destination
        for location, role in ((origin, "origin"), (destination, "destination")):
            if not (location.city and location.state and location.zipcode):
                raise RateProviderError(f"LTL rates require {role} city, state and zipcode")
        pickup_date = query.pickup_date or self._default_pickup_date()
        freight = self._freight_info(query, include_class=True)
        quotes: List[RateQuote] = []
        for carrier_hash in self._carrier_hashes(RateMode.LTL):
            params = {
                "carrierHash": carrier_hash,
                "originCity": origin.city,
                "originState": origin.state,
                "originZipcode": origin.zipcode,
                "originCountry": origin.country,
                "destinationCity": destination.city,
                "destinationState": destination.state,
                "destinationZipcode": destination.zipcode,
                "destinationCountry": destination.country,
                "UOM": query.uom,
                "strictResult": "false",   # returns a RateId so a quote can be saved
                "harmonizedCharges": "true",
                "pickupDate": pickup_date,
                "freightInfo": freight,
            }
            quotes.extend(self._per_carrier_quotes("/ltl/ltlrates", params, query, RateMode.LTL))
        return quotes

    def _ocean_rates(self, query: RateQuery) -> List[RateQuote]:
        origin, destination = query.origin, query.destination
        if not origin.port or not destination.port:
            raise RateProviderError(
                "Ocean rates require origin and destination UN/LOCODE ports"
            )
        freight = self._freight_info(query, include_class=False)
        quotes: List[RateQuote] = []
        for carrier_hash in self._carrier_hashes(RateMode.OCEAN):
            params = {
                "carrierHash": carrier_hash,
                "originPort": origin.port.upper(),
                "destinationPort": destination.port.upper(),
                "UOM": query.uom,
                "hazardous": "true" if query.hazardous else "false",
                "freightInfo": freight,
            }
            quotes.extend(self._per_carrier_quotes("/lcl/oceanrates", params, query, RateMode.OCEAN))
        return quotes

    def _per_carrier_quotes(
        self, path: str, params: Dict[str, Any], query: RateQuery, mode: RateMode
    ) -> List[RateQuote]:
        """One carrier's rate call; a single carrier failing never aborts the lane."""
        try:
            data = self._get(path, params)
        except RateProviderError as exc:
            logger.warning(
                "carrier_rate_failed", mode=mode.value,
                carrier_hash=str(params.get("carrierHash", ""))[:8], error=str(exc),
            )
            return []
        return self._parse_quotes(self._flatten_entries(data), query, mode)

    def resolve_place(
        self,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zipcode: Optional[str] = None,
    ) -> Optional[Location]:
        """Standardize a partial US address via ``/tools/zipcodes``.

        Fills in whatever the user left out — a zip from "Union City, CA", or
        the canonical city/state from a bare "94587" — so a partial address is
        completed from authoritative data instead of being re-asked for (or
        guessed at by the LLM).

        Returns ``None`` when nothing matches confidently; the caller then asks.
        """
        state = (state or "").strip().upper() or None
        city = (city or "").strip() or None
        zipcode = (zipcode or "").strip() or None

        # A zip is the most specific key. City search needs TWO tokens ("Fremont"
        # alone returns nothing, "Fremont, CA" works), so always pair with state.
        terms = [zipcode] if zipcode else []
        if city and state:
            terms.append(f"{city}, {state}")
        elif city and " " in city:
            terms.append(city)  # multi-word city can match on its own

        for term in terms:
            rows = [
                row
                for row in self._zipcode_search(term)
                if str(row.get("Country", "")).upper() == "US"
            ]
            if not rows:
                continue
            if zipcode:
                rows = [r for r in rows if str(r.get("Zipcode", "")).strip() == zipcode] or rows
            if state:
                narrowed = [r for r in rows if str(r.get("StateAbbr", "")).upper() == state]
                if narrowed:
                    rows = narrowed
            if city:
                upper = city.upper()
                exact = [
                    r for r in rows
                    if upper in (str(r.get("City", "")).upper(), str(r.get("CityAlias", "")).upper())
                ]
                if exact:
                    rows = exact
            # Without a zip or state to pin it down, the same city name in more
            # than one state is genuinely ambiguous (Union City NJ vs CA) — ask
            # rather than invent a lane. Several zips within ONE state is fine:
            # any of them standardizes the city for rating.
            if not (zipcode or state) and len({str(r.get("StateAbbr", "")).upper() for r in rows}) > 1:
                continue
            row = rows[0]
            return Location(
                city=str(row.get("City", "")).title() or None,
                state=str(row.get("StateAbbr", "")).upper() or None,
                zipcode=str(row.get("Zipcode", "")).strip() or None,
                country="US",
            )
        return None

    def _zipcode_search(self, term: str) -> List[Dict[str, Any]]:
        """Cached ``/tools/zipcodes`` lookup (postal data barely changes)."""
        cache_key = f"7l:zip:{term.lower()}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            rows = [r for r in self._results(self._get("/tools/zipcodes", {"search": term}))
                    if isinstance(r, dict)]
        except RateProviderError as exc:
            logger.warning("zipcode_lookup_failed", term=term, error=str(exc))
            return []
        if self._cache is not None:
            self._cache.set(cache_key, rows, ttl_seconds=_CARRIER_CACHE_SECONDS)
        return rows

    def airport_location(self, code: str) -> Optional[Location]:
        """Resolve an IATA code to a city/state/zip via ``/tools/airports``.

        Needed because the truck legs of a door-to-door quote are priced on
        city+state+postal code, while the air leg only knows "SFO".
        """
        code = code.strip().upper()
        cache_key = f"7l:airport:{code}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return Location(**cached)
        try:
            data = self._get("/tools/airports", {"search": code})
        except RateProviderError as exc:
            logger.warning("airport_lookup_failed", code=code, error=str(exc))
            return None

        for row in self._results(data):
            if not isinstance(row, dict) or str(row.get("AirportCode", "")).upper() != code:
                continue
            # Name looks like "San Francisco, CA"; Zipcode like "94128".
            name = str(row.get("Name") or "")
            city, _, state = name.partition(",")
            location = Location(
                city=city.strip() or None,
                state=state.strip() or None,
                zipcode=str(row.get("Zipcode") or "").strip() or None,
                country="US",
            )
            if self._cache is not None:
                self._cache.set(cache_key, location.__dict__, ttl_seconds=_CARRIER_CACHE_SECONDS)
            return location
        return None

    def _carrier_hashes(self, mode: RateMode) -> List[str]:
        cache_key = f"7l:carriers:{mode.value}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        if mode is RateMode.LTL:
            data = self._get("/database/ltlaccount", {"carrierType[]": "LTL"})
        else:  # ocean uses the LCL account list, not the LTL one
            data = self._get("/database/lclaccount", {})
        hashes = [
            str(value)
            for value in (self._ci_get(row, ["CarrierHash"]) for row in self._results(data))
            if value
        ][: self._settings.max_carriers_returned]
        if not hashes:
            raise RateProviderError(
                f"No {mode.value} carriers are configured on the 7LFreight account"
            )
        if self._cache is not None:
            self._cache.set(cache_key, hashes, ttl_seconds=_CARRIER_CACHE_SECONDS)
        return hashes

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    #: Keys whose presence marks a dict as a priced quote entry.
    _PRICE_MARKERS = ("TotalCharges", "Total", "TotalCharge", "GrandTotal", "NetCharge")

    def _parse_quotes(
        self, entries: List[Dict[str, Any]], query: RateQuery, mode: RateMode
    ) -> List[RateQuote]:
        quotes: List[RateQuote] = []
        for entry in entries:
            price = self._extract_price(entry)
            if price is None:
                continue  # not a usable quote
            tariff = entry.get("TariffInformation")
            tariff = tariff if isinstance(tariff, dict) else {}
            service = entry.get("ServiceInformation")
            service = service if isinstance(service, dict) else {}

            carrier = self._ci_str(entry, ["AirlineName", "CarrierName", "Carrier", "Name"]) or "Unknown carrier"
            # 7L returns several service levels per carrier — keep them distinct.
            service_type = self._ci_str(service, ["ServiceType", "ServiceLevel"])
            name = f"{carrier} — {service_type}" if service_type else carrier
            transit = self._ci_num(entry, ["TransitDays", "TransitTime", "Transit", "Days"])
            quotes.append(
                RateQuote(
                    carrier_name=name,
                    carrier_code=self._ci_str(entry, ["AirlineCode", "SCAC", "CarrierCode", "Code"]) or "",
                    origin=query.origin.label(),
                    destination=query.destination.label(),
                    price=float(price),
                    currency=self._ci_str(entry, ["Currency", "CurrencyCode"])
                    or self._ci_str(tariff, ["Currency"]) or "USD",
                    mode=mode.value,
                    provider=self.name,
                    transit_days=int(transit) if transit is not None else None,
                    rate_id=self._ci_str(entry, ["RateId", "rateId", "QuoteNumber"]),
                    valid_from=self._ci_str(entry, ["ValidFrom"]) or self._ci_str(tariff, ["ValidFrom"]),
                    valid_to=self._ci_str(entry, ["ValidTo"]) or self._ci_str(tariff, ["ValidTo"]),
                    remarks=self._ci_str(tariff, ["Remarks"])
                    or self._ci_str(entry, ["Remarks", "Notes", "Message"]),
                    breakdown=dict(entry),
                )
            )
        return quotes

    def _extract_price(self, entry: Dict[str, Any]) -> Optional[float]:
        """The total price, whether nested under ``TotalCharges`` or flat."""
        charges = entry.get("TotalCharges")
        if isinstance(charges, dict):
            nested = self._ci_num(charges, ["Total", "GrandTotal", "Amount", "NetCharge"])
            if nested is not None:
                return nested
        return self._ci_num(entry, ["Total", "TotalCharge", "GrandTotal", "Amount", "Price", "NetCharge"])

    @classmethod
    def _flatten_entries(cls, data: Any) -> List[Dict[str, Any]]:
        """Collect every priced quote dict, wherever 7L nests it.

        Rate responses vary by mode — a flat ``data.results`` list, or deeply
        nested (``data.results.Vendors.Airline["SFO-DEN"][...]``). Rather than
        hard-code each shape, walk the whole ``results`` tree and pick out the
        dicts that carry a price.
        """
        payload = data.get("data") if isinstance(data, dict) else None
        root = payload.get("results") if isinstance(payload, dict) else payload
        entries: List[Dict[str, Any]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if any(marker in node for marker in cls._PRICE_MARKERS):
                    entries.append(node)  # a quote — don't descend into its charges
                    return
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(root)
        return entries

    @staticmethod
    def _results(data: Any) -> List[Dict[str, Any]]:
        """Pull the ``data.results`` list out of a 7L envelope (carrier lookup)."""
        payload = data.get("data") if isinstance(data, dict) else None
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, list):
                return results
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #

    def _token(self) -> str:
        """Return a usable access token, refreshing or logging in as needed."""
        entry = self._mem_token
        if entry is None and self._cache is not None:
            entry = self._cache.get(self._token_key)
        if entry:
            self._mem_token = entry
            exp = entry.get("exp")
            margin = self._settings.token_expiry_margin_seconds
            if exp is None or time.time() < exp - margin:
                return entry["accessToken"]
            refreshed = self._refresh(entry["accessToken"])
            if refreshed:
                return refreshed
        return self._login()

    def _login(self) -> str:
        data = self._request_json(
            "POST", "/login",
            json_body={"username": self._username, "password": self._password},
            authed=False,
        )
        token, exp = self._extract_token(data)
        if not token:
            raise RateProviderError("7LFreight login returned no access token")
        self._store_token(token, exp)
        logger.info("seven_l_login_ok")
        return token

    def _refresh(self, old_token: str) -> Optional[str]:
        try:
            data = self._request_json(
                "POST", "/refreshtoken", json_body={"token": old_token}, authed=False
            )
        except RateProviderError as exc:
            logger.info("seven_l_refresh_failed_will_login", error=str(exc))
            return None
        token, exp = self._extract_token(data)
        if not token:
            return None
        self._store_token(token, exp)
        logger.info("seven_l_token_refreshed")
        return token

    def _store_token(self, token: str, exp: Optional[float]) -> None:
        entry = {"accessToken": token, "exp": exp}
        self._mem_token = entry
        if self._cache is not None:
            self._cache.set(self._token_key, entry, ttl_seconds=_TOKEN_RETENTION_SECONDS)

    def _extract_token(self, data: Any) -> "tuple[Optional[str], Optional[float]]":
        payload = data.get("data") if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            payload = data if isinstance(data, dict) else {}
        token = payload.get("accessToken") or payload.get("token")
        exp = self._parse_exp(payload.get("exp") or payload.get("expiresAt"))
        return token, exp

    @staticmethod
    def _parse_exp(exp: Any) -> Optional[float]:
        """Normalize 7L's ``exp`` (epoch s, epoch ms, or ISO-8601) to epoch seconds."""
        if exp is None or isinstance(exp, bool):
            return None
        if isinstance(exp, (int, float)):
            value = float(exp)
            return value / 1000.0 if value > 1e11 else value  # ms → s
        if isinstance(exp, str):
            text = exp.strip()
            try:
                value = float(text)
                return value / 1000.0 if value > 1e11 else value
            except ValueError:
                pass
            try:
                from datetime import datetime

                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        return self._request_json("GET", path, params=params)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        authed: bool = True,
    ) -> Any:
        token = self._token() if authed else None
        response = self._send(method, path, token, params, json_body)
        if authed and response.status_code == 401:
            # A rejected token: force one fresh login, then retry the call once.
            logger.info("seven_l_token_rejected_relogin")
            self._mem_token = None
            token = self._login()
            response = self._send(method, path, token, params, json_body)
        return self._parse_response(response)

    @retry(
        retry=retry_if_exception_type(_RetryableRateError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _send(
        self,
        method: str,
        path: str,
        token: Optional[str],
        params: Optional[Dict[str, Any]],
        json_body: Optional[Dict[str, Any]],
    ) -> httpx.Response:
        """One HTTP call, retried on transient (network / 5xx / 429) failures."""
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = self._client.request(
                method, f"{self._base}{path}", headers=headers, params=params,
                json=json_body, timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise _RetryableRateError(str(exc)) from exc
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise _RetryableRateError(f"{response.status_code} from 7LFreight")
        return response

    def _parse_response(self, response: httpx.Response) -> Any:
        if response.status_code in (401, 403):
            # A rejected credential is a configuration problem, not a transient
            # blip — mark it so callers don't tell the user to "try again".
            raise RateProviderError(
                f"7LFreight authentication failed ({self._error_detail(response)}). "
                "Check 7L_USERNAME / 7L_PASSWORD.",
                details={"kind": "auth"},
            )
        if response.status_code >= 400:
            raise RateProviderError(
                f"7LFreight API error {response.status_code}: {self._error_detail(response)}"
            )
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RateProviderError(
                f"7LFreight returned a non-JSON response: {response.text[:200]}"
            ) from exc

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return response.text[:200]
        if isinstance(body, dict):
            return str(body.get("error") or body.get("message") or body)[:200]
        return str(body)[:200]

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _freight_info(self, query: RateQuery, *, include_class: bool) -> str:
        if not query.items:
            raise RateProviderError("At least one freight item (with weight) is required")
        return json.dumps([item.to_seven_l(include_class=include_class) for item in query.items])

    @staticmethod
    def _default_pickup_date() -> str:
        from datetime import date, timedelta

        return (date.today() + timedelta(days=_DEFAULT_PICKUP_OFFSET_DAYS)).isoformat()

    @staticmethod
    def _ci_get(row: Dict[str, Any], names: List[str]) -> Any:
        """Case-insensitive first-present lookup (7L casing varies by endpoint)."""
        lowered = {str(key).lower(): value for key, value in row.items()}
        for name in names:
            value = lowered.get(name.lower())
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _ci_str(cls, row: Dict[str, Any], names: List[str]) -> Optional[str]:
        value = cls._ci_get(row, names)
        return str(value) if value is not None else None

    @classmethod
    def _ci_num(cls, row: Dict[str, Any], names: List[str]) -> Optional[float]:
        value = cls._ci_get(row, names)
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            return None
