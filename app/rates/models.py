"""Freight rate types shared by the client, the graph and the API.

Plain data — no behaviour beyond serialization and a couple of small helpers.
The 7LFreight client ([app.rates.seven_l]) turns a :class:`RateQuery` into a
list of :class:`RateQuote`, wrapped in a :class:`RateResult`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RateMode(str, Enum):
    """Which freight service a quote is for (also used in cache keys/logs)."""

    AIR = "air"      # airport-to-airport (IATA codes)
    LTL = "ltl"      # less-than-truckload, door-to-door (city/state/zip)
    OCEAN = "ocean"  # LCL ocean freight, port-to-port (UN/LOCODE)


@dataclass
class FreightItem:
    """One line of the shipment's freight description.

    Field names mirror the 7LFreight ``freightInfo[]`` schema. Dimensions and
    freight class are optional (LTL needs a ``class``; air/ocean do not).
    """

    weight: float
    qty: int = 1
    #: "each" (per-piece) or "total" (whole line).
    weight_type: str = "each"
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    #: Packaging: CTN|PLT|CRT|CON|CYL|DRM|ENV|BOX|BDL (7L codes).
    dim_type: str = "PLT"
    commodity: str = "General freight"
    #: LTL only — NMFC freight class 50..500. Ignored by air/ocean.
    freight_class: Optional[str] = None
    hazmat: bool = False
    stack: bool = False

    def to_seven_l(self, *, include_class: bool) -> Dict[str, Any]:
        """Render as a 7LFreight ``freightInfo`` object (string-typed numerics)."""
        item: Dict[str, Any] = {
            "qty": str(self.qty),
            "weight": str(self.weight),
            "weightType": self.weight_type,
            "dimType": self.dim_type,
            "commodity": self.commodity,
        }
        for key, value in (("length", self.length), ("width", self.width), ("height", self.height)):
            if value is not None:
                item[key] = str(value)
        if include_class:
            item["class"] = self.freight_class or "100"
            item["hazmat"] = self.hazmat
            item["stack"] = self.stack
        return item


@dataclass
class Location:
    """An origin or destination, addressed however its mode requires.

    * air   → :attr:`airport` (3-letter IATA)
    * ocean → :attr:`port` (UN/LOCODE, e.g. ``USOAK``)
    * ltl   → :attr:`city` / :attr:`state` / :attr:`zipcode` / :attr:`country`
    """

    airport: Optional[str] = None
    port: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: str = "US"

    def label(self) -> str:
        """Short human string for logs, answers and cache keys."""
        if self.airport:
            return self.airport.upper()
        if self.port:
            return self.port.upper()
        parts = [p for p in (self.city, self.state, self.zipcode) if p]
        return ", ".join(parts) if parts else "unknown"


@dataclass
class RateQuery:
    """A parsed, normalized rate request — the output of the extraction step."""

    mode: RateMode
    origin: Location
    destination: Location
    items: List[FreightItem] = field(default_factory=list)
    #: 7L unit-of-measure selector: US | METRIC | MIXED.
    uom: str = "US"
    #: LTL only — pickup date (YYYY-MM-DD); the client defaults it when absent.
    pickup_date: Optional[str] = None
    #: Ocean only — whether the cargo is hazardous.
    hazardous: bool = False

    def cache_key(self) -> str:
        """Stable key for the rate cache (order-independent within a lane)."""
        items = "|".join(
            f"{i.qty}x{i.weight}{i.weight_type}:{i.dim_type}:{i.freight_class or '-'}"
            f":{i.length or '-'}x{i.width or '-'}x{i.height or '-'}"
            for i in self.items
        )
        return (
            f"{self.mode.value}:{self.origin.label()}>{self.destination.label()}"
            f":{self.uom}:{self.pickup_date or '-'}:{int(self.hazardous)}:{items}"
        ).lower()


@dataclass
class RateQuote:
    """One carrier's quote, normalized across modes.

    ``breakdown`` keeps the raw per-tariff figures for transparency/debugging.
    """

    carrier_name: str
    carrier_code: str
    origin: str
    destination: str
    price: float
    currency: str
    mode: str = RateMode.AIR.value
    provider: str = "7lfreight"
    transit_days: Optional[int] = None
    #: 7L RateId — needed to later save the quote (LTL) via /ltl/rate/save.
    rate_id: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    remarks: Optional[str] = None
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form for caching and the API response."""
        return {
            "carrier_name": self.carrier_name,
            "carrier_code": self.carrier_code,
            "origin": self.origin,
            "destination": self.destination,
            "price": self.price,
            "currency": self.currency,
            "mode": self.mode,
            "provider": self.provider,
            "transit_days": self.transit_days,
            "rate_id": self.rate_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "remarks": self.remarks,
            "breakdown": self.breakdown,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateQuote":
        return cls(**data)


@dataclass
class RateResult:
    """Quotes plus the observability fields callers log/display."""

    quotes: List[RateQuote] = field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.quotes
