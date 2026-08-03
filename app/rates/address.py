"""US street-address standardization via the Census Bureau geocoder.

7LFreight's ``/tools/zipcodes`` only resolves city/state/zip, so a street
address is standardized here instead: free, keyless, and authoritative for US
addresses (``geocoding.geo.census.gov``).

Two things worth knowing about that service:

* It geocodes to the **parcel**, so secondary unit designators are dropped
  ("4835 Decatur Blvd Suite A" → "4835 DECATUR BLVD"). The unit is genuinely
  needed for delivery, so it is carried through as ``address2`` instead of
  being lost.
* An unmatched address returns no candidates rather than a wrong guess, so a
  miss is safe to treat as "ask the user".

Note a standardized street does **not** change a freight rate — LTL/air rating
is priced on city/state/zip — but it makes the quoted address correct, pins the
zip precisely, and is what a booking will need.
"""

import re
from typing import Any, Callable, Dict, Optional

import httpx

from app.utils.logging import get_logger

logger = get_logger(__name__)

_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
#: Current address ranges, as opposed to a frozen vintage.
_BENCHMARK = "Public_AR_Current"
#: Street data is static; cache aggressively to stay light on a public service.
_CACHE_SECONDS = 30 * 24 * 3600
_TIMEOUT_SECONDS = 20

#: Secondary unit designators the geocoder discards but delivery needs. "#" is
#: kept out of the \b-delimited group: a word boundary can never precede it
#: after a space, so "Ave #140" would never match.
_UNIT_RE = re.compile(
    r"((?:\b(?:suite|ste|apt|apartment|unit|bldg|building|floor|fl|room|rm)\b\.?\s*|#\s*)[\w-]+)",
    re.IGNORECASE,
)


#: Directionals must stay uppercase — "152 Selig Dr SW", never "Dr Sw".
_DIRECTIONALS = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
#: str.title() breaks ordinals too: "47TH" -> "47Th". Put the suffix back down.
_ORDINAL_RE = re.compile(r"\b(\d+)(ST|ND|RD|TH)\b", re.IGNORECASE)


def _pretty(text: Optional[str]) -> Optional[str]:
    """Title-case a street line the way postal addresses are actually written.

    The geocoder returns all-caps ("152 SELIG DR SW"); plain ``str.title()``
    would corrupt both directionals and ordinals, which are exactly the parts a
    driver needs to read correctly.
    """
    if not text:
        return text
    words = [w.upper() if w.upper() in _DIRECTIONALS else w.title() for w in text.split()]
    return _ORDINAL_RE.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}", " ".join(words))


def _unit(text: str) -> Optional[str]:
    """Pull "Suite A" / "#140" out of a raw address line."""
    match = _UNIT_RE.search(text or "")
    return _pretty(" ".join(match.group(1).split())) if match else None


def standardize_street(
    address1: str,
    city: Optional[str] = None,
    state: Optional[str] = None,
    zipcode: Optional[str] = None,
    cache: Any = None,
    client: Optional[httpx.Client] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """Standardize a US street address, or ``None`` if it can't be matched.

    Returns ``{address1, address2, city, state, zipcode}`` — ``address2`` holds
    any secondary unit found in the input.
    """
    if not (address1 or "").strip():
        return None
    query = ", ".join(part for part in (address1, city, state) if part)
    if zipcode:
        query = f"{query} {zipcode}"

    cache_key = f"census:{query.lower()}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached or None  # {} caches a known miss

    try:
        request = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
        response = request.get(
            _GEOCODER_URL,
            params={"address": query, "benchmark": _BENCHMARK, "format": "json"},
            timeout=_TIMEOUT_SECONDS,
        )
        matches = ((response.json().get("result") or {}).get("addressMatches")) or []
    except Exception as exc:  # noqa: BLE001 — standardization must never break a quote
        logger.warning("street_standardize_failed", error=str(exc))
        return None

    if not matches:
        logger.info("street_no_match", query=query[:80])
        if cache is not None:
            cache.set(cache_key, {}, ttl_seconds=_CACHE_SECONDS)
        return None

    best = matches[0]
    components = best.get("addressComponents") or {}
    # matchedAddress is "1500 ATLANTIC ST, UNION CITY, CA, 94587"; its first
    # segment is the cleaned street line.
    matched = str(best.get("matchedAddress") or "")
    street = matched.split(",")[0].strip() or address1
    result = {
        "address1": _pretty(street),
        # The geocoder drops the unit, so keep whatever the user gave.
        "address2": _unit(address1),
        "city": str(components.get("city") or city or "").title() or None,
        "state": str(components.get("state") or state or "").upper() or None,
        "zipcode": str(components.get("zip") or zipcode or "").strip() or None,
    }
    if cache is not None:
        cache.set(cache_key, result, ttl_seconds=_CACHE_SECONDS)
    return result


def make_address_resolver(zip_resolver: Optional[Callable] = None, cache: Any = None) -> Callable:
    """One hook the extractor can call for any completeness of address.

    A street address goes to the Census geocoder; without one (or when it can't
    be matched) it falls back to ``zip_resolver`` — 7LFreight's city/zip lookup.
    """

    def resolve(
        address1: Optional[str] = None,
        address2: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zipcode: Optional[str] = None,
    ) -> Optional[Dict[str, Optional[str]]]:
        if address1:
            street = standardize_street(address1, city, state, zipcode, cache=cache)
            if street:
                # Never drop a unit the user typed into address2 explicitly.
                street["address2"] = street.get("address2") or address2
                return street
        if zip_resolver is None:
            return None
        place = zip_resolver(city=city, state=state, zipcode=zipcode)
        if place is None:
            return None
        return {
            "address1": address1,
            "address2": address2,
            "city": place.city,
            "state": place.state,
            "zipcode": place.zipcode,
        }

    return resolve
