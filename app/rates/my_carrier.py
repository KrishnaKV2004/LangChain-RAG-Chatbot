"""MyCarrier LTL rating client (``POST /feature/rating``).

Same contract as :class:`~app.rates.seven_l.SevenLRates` — a
:class:`RateQuery` in, a :class:`RateResult` out — so the graph and the API
don't care which provider priced a leg.

Two differences from 7L worth knowing:

* **No auth.** The account is identified by ``customer_email`` +
  ``location_id`` in the body; there is no login or token.
* **One call fans out to ~40 carriers** and returns a mix of ``available`` and
  ``rejected`` quotes, each with several service levels (STD / GUR / GSNOON).
  Only ``available`` ones are real prices.
"""

import json
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
from app.rates.models import FreightItem, RateQuery, RateQuote, RateResult
from app.utils.exceptions import RateProviderError
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: Metric → US conversions; we always send INCHES/LB rather than guess at
#: MyCarrier's metric UOM enum values.
_KG_TO_LB = 2.20462
_CM_TO_IN = 0.393701


class _RetryableError(Exception):
    """Transient failure worth retrying (network / 5xx / 429)."""


class MyCarrierRates:
    """MyCarrier implementation of the rate lookup."""

    name = "mycarrier"

    def __init__(
        self,
        settings: Settings,
        cache: Optional[TTLCache] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._settings = settings.mycarrier
        self._max_returned = settings.rates.max_carriers_returned
        self._cache_ttl = settings.rates.cache_ttl_seconds
        self._cache = cache
        self._client = client or httpx.Client(timeout=settings.mycarrier.timeout_seconds)

    # ------------------------------------------------------------------ #

    def get_rates(self, query: RateQuery) -> RateResult:
        """Cheapest-first LTL quotes for ``query``, cached where possible."""
        cache_key = f"rates:{self.name}:{query.cache_key()}"
        result = RateResult()
        with Timer() as timer:
            cached = self._cache.get(cache_key) if self._cache is not None else None
            if cached is not None:
                result.quotes = [RateQuote.from_dict(item) for item in cached]
                result.cache_hit = True
            else:
                quotes = self._parse(self._post(self._body(query)), query)
                quotes.sort(key=lambda q: q.price)
                quotes = quotes[: self._max_returned]
                if self._cache is not None:
                    self._cache.set(
                        cache_key, [q.to_dict() for q in quotes], ttl_seconds=self._cache_ttl
                    )
                result.quotes = quotes
        result.latency_ms = timer.elapsed_ms
        logger.info(
            "rate_lookup_complete",
            provider=self.name,
            lane=f"{query.origin.label()}>{query.destination.label()}",
            quotes=len(result.quotes),
            cache_hit=result.cache_hit,
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    # ------------------------------------------------------------------ #
    # Request
    # ------------------------------------------------------------------ #

    def _body(self, query: RateQuery) -> Dict[str, Any]:
        origin, destination = query.origin, query.destination
        for location, role in ((origin, "origin"), (destination, "destination")):
            if not (location.city and location.state and location.zipcode):
                raise RateProviderError(
                    f"MyCarrier LTL needs {role} city, state and postal code"
                )
        if not query.items:
            raise RateProviderError("At least one freight item (with weight) is required")

        return {
            "customerName": self._settings.customer_name,
            "customerId": "",
            "customerEmail": self._settings.customer_email,
            "locationId": self._settings.location_id,
            "data": {
                "preferredCarriers": "",
                "accessorialServices": "",
                "shipment": {
                    "shipmentId": "",
                    "directionOverride": "THIRD_PARTY",
                    "shipmentType": "LTL",
                    "stops": [
                        self._stop(1, "PICKUP", origin, query.pickup_date or ""),
                        self._stop(2, "DROP", destination),
                    ],
                    "shipmentLineItems": [
                        self._line_item(index, item, query.uom)
                        for index, item in enumerate(query.items, start=1)
                    ],
                },
            },
        }

    @staticmethod
    def _stop(number: int, stop_type: str, location, date: Optional[str] = None) -> Dict[str, Any]:
        stop: Dict[str, Any] = {
            "stopNumber": number,
            "stopType": stop_type,
            "name": "",
            "addressLine1": location.address1 or "",
            "addressLine2": location.address2 or "",
            "city": location.city,
            "stateProvince": location.state,
            "postalCode": location.zipcode,
            "country": {
                "alpha2Code": location.country or "US",
                "name": "Canada" if (location.country or "US").upper() == "CA" else "United States",
            },
            "contactEmail": "",
            "contactName": "",
            "contactPhone": "",
            "note": "",
        }
        if date is not None:
            stop["schedule"] = {"date": date}
        return stop

    @classmethod
    def _line_item(cls, index: int, item: FreightItem, uom: str) -> Dict[str, Any]:
        metric = uom.upper().startswith("MET")
        weight = item.weight * (_KG_TO_LB if metric else 1.0)
        if item.weight_type == "total" and item.qty > 1:
            weight = weight / item.qty  # MyCarrier weight is per piece
        # MyCarrier types every dimension/weight as Int32 — a float 400s.
        # ponytail: whole inches/pounds is how LTL is tendered anyway; if a
        # fractional weight ever matters, ask them for a decimal-typed field.
        dim = (lambda v: int(round(v * (_CM_TO_IN if metric else 1.0))) if v else 0)
        return {
            "lineItemId": str(index),
            "name": "",
            "class": item.freight_class or "100",
            "description": item.commodity or "goods",
            "associatedPickupStopNumber": 1,
            "associatedDropStopNumber": 2,
            "nmfcItemCode": "",
            "nmfcSubCode": "",
            "commodityType": "",
            "harmonizedCode": "",
            "isHazmat": item.hazmat,
            "hazmat": {
                "identification_number": "",
                "packingGroupId": "",
                "classId": "",
                "emergencyContact": {"name": "", "phone": "", "email": ""},
            },
            "unitValue": 0,
            "unitValueCurrency": "USD",
            "dimensions": {
                "height": dim(item.height),
                "heightUOM": "INCHES",
                "length": dim(item.length),
                "lengthUOM": "INCHES",
                "width": dim(item.width),
                "widthUOM": "INCHES",
                "weight": max(1, int(round(weight))),
                "weightUOM": "LB",
                "stackable": item.stack,
                "quantity": item.qty,
                "packageType": item.dim_type or "PLT",
            },
        }

    # ------------------------------------------------------------------ #
    # Response
    # ------------------------------------------------------------------ #

    def _parse(self, payload: Any, query: RateQuery) -> List[RateQuote]:
        data = payload.get("data") if isinstance(payload, dict) else None
        rates = data.get("rates") if isinstance(data, dict) else None
        if not isinstance(rates, list):
            return []

        quotes: List[RateQuote] = []
        rejected = 0
        for rate in rates:
            if not isinstance(rate, dict):
                continue
            if rate.get("quoteStatus") != "available":
                rejected += 1
                continue
            carrier = rate.get("carrier") if isinstance(rate.get("carrier"), dict) else {}
            for price in rate.get("price", []) or []:
                if not isinstance(price, dict):
                    continue
                amount = self._num(price.get("netPrice") or price.get("grossPrice"))
                if amount is None or amount <= 0:
                    continue
                service = price.get("serviceType") if isinstance(price.get("serviceType"), dict) else {}
                label = service.get("description") or service.get("code") or ""
                name = carrier.get("name") or carrier.get("scac") or "Unknown carrier"
                quotes.append(
                    RateQuote(
                        carrier_name=f"{name} — {label}" if label else name,
                        carrier_code=str(carrier.get("scac") or ""),
                        origin=query.origin.label(),
                        destination=query.destination.label(),
                        price=amount,
                        currency=str(rate.get("currencyCode") or "USD"),
                        mode=query.mode.value,
                        provider=self.name,
                        transit_days=self._int(service.get("serviceDays")),
                        rate_id=str(price.get("quoteId") or rate.get("quoteId") or "") or None,
                        valid_to=str(rate.get("quoteExpiryDate") or "") or None,
                        breakdown={"priceBreakUp": price.get("priceBreakUp") or []},
                    )
                )
        logger.debug("mycarrier_parsed", available=len(quotes), rejected=rejected)
        return quotes

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _post(self, body: Dict[str, Any]) -> Any:
        response = self._send(body)
        if response.status_code >= 400:
            raise RateProviderError(
                f"MyCarrier API error {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RateProviderError(
                f"MyCarrier returned a non-JSON response: {response.text[:200]}"
            ) from exc

    @retry(
        retry=retry_if_exception_type(_RetryableError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _send(self, body: Dict[str, Any]) -> httpx.Response:
        try:
            response = self._client.post(
                f"{self._settings.base_url.rstrip('/')}/feature/rating",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise _RetryableError(str(exc)) from exc
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise _RetryableError(f"{response.status_code} from MyCarrier")
        return response

    # ------------------------------------------------------------------ #

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            return float(str(value).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            return None

    @classmethod
    def _int(cls, value: Any) -> Optional[int]:
        number = cls._num(value)
        return int(number) if number is not None else None
