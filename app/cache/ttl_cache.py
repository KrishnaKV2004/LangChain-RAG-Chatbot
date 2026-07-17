"""Disk-backed TTL cache.

Used for transient web content: search results are cached for a configurable
TTL (default 24 h) and *never* permanently stored — expired entries are
deleted on read and by :meth:`purge_expired`.

Design notes:

* One JSON file per key (filename = SHA-256 of the key) — no extra
  dependencies, safe across process restarts, trivially inspectable.
* Values must be JSON-serializable; callers own (de)serialization of richer
  types.
* A process-wide lock guards multi-threaded access (FastAPI worker threads);
  cross-process races are tolerable because entries are immutable snapshots.
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.utils.exceptions import CacheError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TTLCache:
    """A small, dependency-free, disk-persisted TTL key/value store."""

    def __init__(self, path: Path, default_ttl_seconds: float) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl_seconds
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or ``None`` if absent or expired.

        Expired entries are deleted eagerly so stale web content never
        outlives its TTL on disk.
        """
        entry_path = self._entry_path(key)
        with self._lock:
            if not entry_path.exists():
                return None
            try:
                entry = json.loads(entry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entry_path.unlink(missing_ok=True)  # corrupt entry → drop it
                return None

            if entry["expires_at"] <= time.time():
                entry_path.unlink(missing_ok=True)
                logger.debug("cache_expired", key=key[:64])
                return None
            return entry["value"]

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Store ``value`` under ``key`` for ``ttl_seconds`` (default TTL if None)."""
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        entry = {
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "value": value,
        }
        try:
            payload = json.dumps(entry, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise CacheError(f"Value for '{key[:64]}' is not JSON-serializable") from exc

        with self._lock:
            self._entry_path(key).write_text(payload, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Maintenance (powers the /cache endpoint)
    # ------------------------------------------------------------------ #

    def purge_expired(self) -> int:
        """Delete every expired entry; returns how many were removed."""
        removed = 0
        now = time.time()
        with self._lock:
            for entry_path in self._path.glob("*.json"):
                try:
                    entry = json.loads(entry_path.read_text(encoding="utf-8"))
                    expired = entry["expires_at"] <= now
                except (json.JSONDecodeError, OSError, KeyError):
                    expired = True  # unreadable = treat as garbage
                if expired:
                    entry_path.unlink(missing_ok=True)
                    removed += 1
        if removed:
            logger.info("cache_purged", removed=removed)
        return removed

    def clear(self) -> int:
        """Delete every entry regardless of TTL; returns how many were removed."""
        with self._lock:
            entries = list(self._path.glob("*.json"))
            for entry_path in entries:
                entry_path.unlink(missing_ok=True)
        logger.info("cache_cleared", removed=len(entries))
        return len(entries)

    def stats(self) -> Dict[str, Any]:
        """Entry counts and disk footprint for monitoring."""
        now = time.time()
        total = expired = 0
        size_bytes = 0
        with self._lock:
            for entry_path in self._path.glob("*.json"):
                total += 1
                size_bytes += entry_path.stat().st_size
                try:
                    entry = json.loads(entry_path.read_text(encoding="utf-8"))
                    if entry["expires_at"] <= now:
                        expired += 1
                except (json.JSONDecodeError, OSError, KeyError):
                    expired += 1
        return {
            "path": str(self._path),
            "entries": total,
            "expired_entries": expired,
            "size_bytes": size_bytes,
            "default_ttl_seconds": self._default_ttl,
        }

    # ------------------------------------------------------------------ #

    def _entry_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._path / f"{digest}.json"
