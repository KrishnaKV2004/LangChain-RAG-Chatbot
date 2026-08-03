"""Door-to-door pricing: truck + air + truck, totalled.

A city-to-city air shipment is really three priced legs:

    warehouse ──truck──► origin airport ──air──► destination airport ──truck──► warehouse
      (MyCarrier)                        (7LFreight)                        (MyCarrier)

The warehouse CSV (``documents/Warehouses.csv``) is the whitelist of allowed
truck-leg endpoints: only those locations can start or end a leg. The total is
the sum of the cheapest quote on each leg; a leg that can't be priced is
reported as such and the total is flagged partial rather than silently wrong.
"""

import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rates.models import SUPPORTED_AIRPORTS, Location, RateMode, RateQuery, RateQuote
from app.utils.exceptions import RateProviderError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


#: Machine provider name → the label shown to users as the rate's source.
PROVIDER_LABELS = {"7lfreight": "7LFreight", "mycarrier": "MyCarrier"}



def provider_label(name: Optional[str]) -> str:
    return PROVIDER_LABELS.get((name or "").lower(), name or "unknown")


@dataclass
class Leg:
    """One priced (or unpriceable) segment of the journey."""

    label: str
    quote: Optional[RateQuote] = None
    note: Optional[str] = None
    #: How many quotes this leg was chosen from, and which providers bid.
    compared: int = 0
    providers: tuple = ()

    @property
    def source(self) -> str:
        """Which provider the winning quote came from."""
        return provider_label(self.quote.provider) if self.quote else "—"


@dataclass
class DoorToDoorPlan:
    legs: List[Leg] = field(default_factory=list)
    currency: str = "USD"

    @property
    def priced(self) -> List[Leg]:
        return [leg for leg in self.legs if leg.quote is not None]

    @property
    def total(self) -> Optional[float]:
        if not self.priced:
            return None
        return round(sum(leg.quote.price for leg in self.priced), 2)

    @property
    def complete(self) -> bool:
        """True when every leg priced — a total below this is partial."""
        return bool(self.legs) and len(self.priced) == len(self.legs)


# --------------------------------------------------------------------------- #
# Warehouse whitelist
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def load_warehouses(csv_path: str) -> tuple:
    """Truck-capable, operational warehouses with a parseable US ZIP."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("warehouse_csv_missing", path=str(path))
        return ()
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if "truck" not in (row.get("modesHandled") or "").lower():
                continue
            if (row.get("operationalStatus") or "").lower() != "operational":
                continue
            match = _ZIP_RE.findall(row.get("address") or "")
            if not match:
                continue  # non-US postal code (CA) — MyCarrier body expects a ZIP
            rows.append({
                "id": row.get("id"),
                "name": row.get("name"),
                "city": row.get("city"),
                "state": row.get("state"),
                "zipcode": match[-1],
            })
    logger.info("warehouses_loaded", count=len(rows), path=str(path))
    return tuple(rows)


def nearest_warehouse(zipcode: str, csv_path: str) -> Optional[Dict[str, Any]]:
    """Warehouse closest to ``zipcode`` by shared ZIP prefix.

    ponytail: ZIP-prefix proximity (94128 → 94587 shares "94") instead of
    geocoding. The CSV has lat/long, so swap in haversine against real airport
    coordinates if this ever mispicks.
    """
    warehouses = load_warehouses(csv_path)
    if not warehouses or not zipcode:
        return None

    def score(row: Dict[str, Any]) -> tuple:
        other = row["zipcode"]
        shared = 0
        for a, b in zip(zipcode, other):
            if a != b:
                break
            shared += 1
        return (-shared, abs(int(other) - int(zipcode)))

    return min(warehouses, key=score)


def _location(row: Dict[str, Any]) -> Location:
    return Location(city=row["city"], state=row["state"], zipcode=row["zipcode"], country="US")


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def plan_door_to_door(query: RateQuery, seven_l, my_carrier, csv_path: str) -> DoorToDoorPlan:
    """Price the truck + air + truck chain for an airport-to-airport ``query``.

    Provider per leg is fixed: **MyCarrier** prices the warehouse↔airport truck
    legs, **7LFreight** prices the airport↔airport air leg. Within a leg the
    cheapest of that provider's carriers wins.

    Both endpoints must be supported: airports from
    :data:`SUPPORTED_AIRPORTS`, warehouses from the CSV whitelist.

    Every provider call is issued concurrently, so a 3-leg quote costs roughly
    one call's latency instead of three.
    """
    plan = DoorToDoorPlan()
    origin_code = (query.origin.airport or "").upper()
    dest_code = (query.destination.airport or "").upper()
    origin_airport = _airport(seven_l, origin_code)
    dest_airport = _airport(seven_l, dest_code)

    # MyCarrier owns the truck legs; 7LFreight owns the air leg.
    truck_providers = [my_carrier] if my_carrier is not None else []
    unsupported = [
        code for code in (origin_code, dest_code) if code and code not in SUPPORTED_AIRPORTS
    ]
    air_note = None
    if unsupported:
        air_note = f"{', '.join(unsupported)} is not a supported gateway airport"
    elif seven_l is None:
        air_note = "air rating is not configured"

    specs = [
        _truck_spec(f"Pickup: warehouse → {origin_code}", origin_airport, query,
                    csv_path, truck_providers, to_airport=True),
        _Spec(f"Air: {origin_code} → {dest_code}", query,
              [] if air_note else [seven_l], note=air_note),
        _truck_spec(f"Delivery: {dest_code} → warehouse", dest_airport, query,
                    csv_path, truck_providers, to_airport=False),
    ]

    # Fan every (leg, provider) call out at once. ponytail: threads because the
    # provider clients are blocking httpx; no async rewrite needed for 5 calls.
    calls = [
        (index, provider)
        for index, spec in enumerate(specs)
        for provider in spec.providers
    ]
    results: Dict[tuple, Any] = {}
    if calls:
        with ThreadPoolExecutor(max_workers=min(6, len(calls))) as pool:
            futures = {
                pool.submit(_quote, provider, specs[index].query): (index, provider)
                for index, provider in calls
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    for index, spec in enumerate(specs):
        plan.legs.append(_pick(spec, [results.get((index, p)) for p in spec.providers]))

    priced = plan.priced
    if priced:
        plan.currency = priced[0].quote.currency
    logger.info(
        "door_to_door_planned",
        legs=len(plan.legs),
        priced=len(priced),
        total=plan.total,
        complete=plan.complete,
        sources=[leg.source for leg in priced],
    )
    return plan


@dataclass
class _Spec:
    """A leg to price: its label, the query, and who may bid on it."""

    label: str
    query: Optional[RateQuery]
    providers: List[Any] = field(default_factory=list)
    note: Optional[str] = None


def _airport(seven_l, code: str) -> Optional[Location]:
    """Resolve a gateway to city/state/zip; a lookup blip must not sink the plan."""
    if not (seven_l and code):
        return None
    try:
        return seven_l.airport_location(code)
    except Exception as exc:  # noqa: BLE001 — degrade to "unresolved", not a crash
        logger.warning("airport_resolve_failed", code=code, error=str(exc))
        return None


def _truck_spec(label, airport: Optional[Location], query, csv_path, providers, *, to_airport: bool) -> _Spec:
    if not providers:
        return _Spec(label, None, [], "truck rating is not configured")
    if airport is None or not airport.zipcode:
        return _Spec(label, None, [], "could not resolve the airport's postal code")
    warehouse = nearest_warehouse(airport.zipcode, csv_path)
    if warehouse is None:
        return _Spec(label, None, [], "no supported warehouse found for this lane")

    hub = _location(warehouse)
    origin, destination = (hub, airport) if to_airport else (airport, hub)
    return _Spec(
        f"{label} ({warehouse['name']}, {warehouse['city']}, {warehouse['state']})",
        RateQuery(
            mode=RateMode.LTL,
            origin=origin,
            destination=destination,
            items=query.items,
            uom=query.uom,
            pickup_date=query.pickup_date,
        ),
        list(providers),
    )


def bid(query: RateQuery, providers: List[Any]) -> List[RateQuote]:
    """Every provider's quotes for one lane, concurrently, cheapest first.

    Used for a single-leg quote (e.g. an explicit LTL-only request) where both
    providers can price the same lane, so they genuinely compete.
    """
    live = [p for p in providers if p is not None]
    if not live:
        return []
    with ThreadPoolExecutor(max_workers=len(live)) as pool:
        bids = list(pool.map(lambda p: _quote(p, query), live))
    return sorted((quote for group in bids for quote in group), key=lambda q: q.price)


def _quote(provider, query: RateQuery) -> List[RateQuote]:
    """One provider's quotes for one leg; a failure is an empty bid, not a crash."""
    try:
        return list(provider.get_rates(query).quotes)
    except RateProviderError as exc:
        logger.warning(
            "leg_provider_failed", provider=getattr(provider, "name", "?"), error=exc.message
        )
        return []
    except Exception as exc:  # noqa: BLE001 — one provider must not sink the quote
        logger.warning("leg_provider_error", provider=getattr(provider, "name", "?"), error=str(exc))
        return []


def _pick(spec: _Spec, bids: List[Optional[List[RateQuote]]]) -> Leg:
    """Cheapest quote across every provider that bid on this leg."""
    if spec.note:
        return Leg(spec.label, note=spec.note)
    quotes = [quote for bid in bids if bid for quote in bid]
    providers = tuple(sorted({provider_label(q.provider) for q in quotes}))
    if not quotes:
        return Leg(spec.label, note="no carrier returned a rate", providers=providers)
    return Leg(
        spec.label,
        quote=min(quotes, key=lambda q: q.price),
        compared=len(quotes),
        providers=providers,
    )
