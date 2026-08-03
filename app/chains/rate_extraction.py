"""Natural-language → :class:`RateQuery` extraction.

The RATES route can't hit the carrier API with raw user phrasing — it needs a
normalized lane, weight and mode. This chain uses the cheap router-tier LLM to
turn "air freight 200 kg from SFO to Chicago" into a structured
:class:`~app.rates.models.RateQuery`, exactly as :class:`QueryRouter` turns a
question into a route.

Robustness over elegance (same stance as the router): the model's JSON is
parsed defensively and any failure yields a *clarification* — a question the
graph asks the user — never an exception. Required fields absent for the
chosen mode also produce a clarification instead of a doomed API call.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.chains.prompts import RATE_EXTRACTION_SYSTEM_PROMPT
from app.rates.models import SUPPORTED_AIRPORTS, FreightItem, Location, RateMode, RateQuery
from app.utils.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

#: Recent turns fed as context so follow-ups ("make it 300 kg") resolve.
_MAX_HISTORY_MESSAGES = 6

_GENERIC_CLARIFICATION = (
    "I can quote air, LTL (truck) or ocean freight. Could you tell me the origin, "
    "destination and the shipment weight?"
)


@dataclass
class RateExtraction:
    """Result of one extraction: either a query to run or a question to ask."""

    query: Optional[RateQuery] = None
    clarification: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.query is not None


class RateExtractor:
    """Extracts a structured rate request with a small LLM."""

    def __init__(self, llm: BaseChatModel, place_resolver: Optional[Callable] = None) -> None:
        self._llm = llm
        # Completes partial addresses from authoritative postal data so a
        # missing zip is looked up rather than asked for (or invented).
        self._place_resolver = place_resolver

    def extract(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> RateExtraction:
        """Return a :class:`RateExtraction` for ``query`` (never raises)."""
        with Timer() as timer:
            try:
                response = self._llm.invoke(self._messages(query, history))
                data = self._parse_json(str(response.content))
                result = self._build(data)
            except Exception as exc:  # noqa: BLE001 — extraction must never fail hard
                logger.warning("rate_extraction_failed", error=str(exc))
                result = RateExtraction(clarification=_GENERIC_CLARIFICATION)

        logger.info(
            "rate_extracted",
            ok=result.ok,
            mode=result.query.mode.value if result.query else None,
            latency_ms=round(timer.elapsed_ms, 1),
        )
        return result

    # ------------------------------------------------------------------ #
    # Message assembly
    # ------------------------------------------------------------------ #

    def _messages(
        self, query: str, history: Optional[List[Dict[str, str]]]
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [SystemMessage(content=RATE_EXTRACTION_SYSTEM_PROMPT)]
        # Only the USER's own turns carry constraints (lane, weight, mode); the
        # assistant's clarifications/errors are noise that derails extraction.
        user_turns = [t.get("content", "") for t in (history or []) if t.get("role") == "user"]
        for content in user_turns[-_MAX_HISTORY_MESSAGES:]:
            messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=query))
        return messages

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Pull the first JSON object out of the reply, ignoring fences/prose."""
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("no JSON object found in extractor reply")
        parsed = json.loads(cleaned[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("extractor reply was not a JSON object")
        return parsed

    def _build(self, data: Dict[str, Any]) -> RateExtraction:
        mode = self._coerce_mode(data)
        if mode is None:
            return RateExtraction(clarification=_GENERIC_CLARIFICATION)

        query = RateQuery(
            mode=mode,
            origin=self._location(data.get("origin")),
            destination=self._location(data.get("destination")),
            items=self._items(data.get("items")),
            uom=self._uom(data.get("uom")),
            pickup_date=self._str(data.get("pickup_date")),
            hazardous=bool(data.get("hazardous", False)),
        )

        # Standardize before deciding anything is missing: "Union City, CA" with
        # no zip is completable, not incomplete.
        query.origin = self._standardize(query.origin)
        query.destination = self._standardize(query.destination)

        missing = self._missing(query)
        ready = bool(data.get("ready", True)) and not missing
        if not ready:
            # Prefer a specific ask that names what we ALREADY have (the lane) and
            # only the fields still missing — the model's own clarification tends to
            # re-ask for things the user already gave. Fall back to it only when we
            # can't compute specifics.
            if missing:
                return RateExtraction(clarification=self._ask_for(query, missing))
            return RateExtraction(
                clarification=self._str(data.get("clarification")) or _GENERIC_CLARIFICATION
            )
        return RateExtraction(query=query)

    # ------------------------------------------------------------------ #
    # Field builders / validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_mode(data: Dict[str, Any]) -> "Optional[RateMode]":
        """The mode the model gave, or inferred from the lane it parsed.

        A terse follow-up ("10 kgs", "make it LTL") often drops the mode; rather
        than reset to a blank generic ask, infer it from whatever address the
        model carried over from earlier turns.
        """
        try:
            return RateMode(str(data.get("mode", "")).strip().lower())
        except ValueError:
            pass
        origin = data.get("origin") if isinstance(data.get("origin"), dict) else {}
        destination = data.get("destination") if isinstance(data.get("destination"), dict) else {}

        def present(key: str) -> bool:
            return bool(origin.get(key) or destination.get(key))

        if present("airport"):
            return RateMode.AIR
        if present("port"):
            return RateMode.OCEAN
        if present("city") or present("zipcode"):
            return RateMode.LTL
        return None

    def _location(self, raw: Any) -> Location:
        data = raw if isinstance(raw, dict) else {}
        return Location(
            airport=self._str(data.get("airport")),
            port=self._str(data.get("port")),
            city=self._str(data.get("city")),
            state=self._str(data.get("state")),
            zipcode=self._str(data.get("zipcode")),
            country=self._str(data.get("country")) or "US",
            address1=self._str(data.get("address1")),
            address2=self._str(data.get("address2")),
        )

    def _standardize(self, location: Location) -> Location:
        """Complete a partial city/state/zip from postal data; keep what's given.

        Airports and seaports are already canonical codes, so they pass through.
        A resolver miss leaves the location untouched — the caller then asks.
        """
        if self._place_resolver is None or location.airport or location.port:
            return location
        if not (location.city or location.zipcode or location.address1):
            return location
        # A street address is always worth standardizing (spelling, unit, zip);
        # a bare city/state/zip that is already complete is not.
        if not location.address1 and location.city and location.state and location.zipcode:
            return location
        try:
            resolved = self._place_resolver(
                address1=location.address1,
                address2=location.address2,
                city=location.city,
                state=location.state,
                zipcode=location.zipcode,
            )
        except Exception as exc:  # noqa: BLE001 — never let lookup break extraction
            logger.warning("address_standardize_failed", error=str(exc))
            return location
        if not resolved:
            return location
        # Trust the authoritative record for the canonical form, but never drop
        # a field the user supplied that the lookup left blank.
        get = resolved.get if isinstance(resolved, dict) else lambda k: getattr(resolved, k, None)
        return Location(
            address1=get("address1") or location.address1,
            address2=get("address2") or location.address2,
            city=get("city") or location.city,
            state=get("state") or location.state,
            zipcode=get("zipcode") or location.zipcode,
            country=get("country") or location.country,
        )

    def _items(self, raw: Any) -> List[FreightItem]:
        items: List[FreightItem] = []
        for entry in raw or []:
            if not isinstance(entry, dict):
                continue
            weight = self._num(entry.get("weight"))
            if weight is None or weight <= 0:
                continue  # a line without a real weight can't be priced
            qty = self._num(entry.get("qty"))
            items.append(
                FreightItem(
                    weight=weight,
                    qty=int(qty) if qty and qty >= 1 else 1,
                    weight_type=self._str(entry.get("weight_type")) or "each",
                    length=self._num(entry.get("length")),
                    width=self._num(entry.get("width")),
                    height=self._num(entry.get("height")),
                    dim_type=self._str(entry.get("dim_type")) or "PLT",
                    commodity=self._str(entry.get("commodity")) or "General freight",
                    freight_class=self._str(entry.get("freight_class")),
                    hazmat=bool(entry.get("hazmat", False)),
                    stack=bool(entry.get("stack", False)),
                )
            )
        return items

    @staticmethod
    def _missing(query: RateQuery) -> List[str]:
        """Required fields absent for the chosen mode."""
        origin, destination = query.origin, query.destination
        missing: List[str] = []
        if query.mode is RateMode.AIR:
            if not origin.airport:
                missing.append("origin airport")
            if not destination.airport:
                missing.append("destination airport")
            # Only the gateways in Airports.pdf are quotable.
            for location, role in ((origin, "origin"), (destination, "destination")):
                code = (location.airport or "").upper()
                if code and code not in SUPPORTED_AIRPORTS:
                    missing.append(f"a supported {role} gateway airport (we don't ship via {code})")
        elif query.mode is RateMode.OCEAN:
            if not origin.port:
                missing.append("origin port")
            if not destination.port:
                missing.append("destination port")
        else:  # LTL
            if not (origin.city and origin.state and origin.zipcode):
                missing.append("origin city, state and zip")
            if not (destination.city and destination.state and destination.zipcode):
                missing.append("destination city, state and zip")
        # Every 7L rate endpoint (air/LTL/ocean) needs weight AND per-piece
        # dimensions — the API 400s without L×W×H, so we must ask for them.
        if not query.items:
            missing.append("shipment weight")
        elif any(i.length is None or i.width is None or i.height is None for i in query.items):
            missing.append("package dimensions (length × width × height)")
        return missing

    @classmethod
    def _ask_for(cls, query: RateQuery, missing: List[str]) -> str:
        if not missing:
            return _GENERIC_CLARIFICATION
        needs = ", and the ".join(missing)
        lane = cls._known_lane(query)
        prefix = f"I've got {lane}. " if lane else ""
        return f"{prefix}To get you a quote I still need the {needs}. Could you share that?"

    @staticmethod
    def _known_lane(query: RateQuery) -> "Optional[str]":
        """Human description of whatever origin/destination we did parse."""
        origin = query.origin.label()
        destination = query.destination.label()
        origin = None if origin == "unknown" else origin
        destination = None if destination == "unknown" else destination
        if origin and destination:
            return f"{origin} → {destination}"
        if origin:
            return f"origin {origin}"
        if destination:
            return f"destination {destination}"
        return None

    # ------------------------------------------------------------------ #
    # Coercion helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _uom(value: Any) -> str:
        text = str(value or "US").upper().strip()
        if text.startswith("MET"):
            return "METRIC"
        if text == "MIXED":
            return "MIXED"
        return "US"
