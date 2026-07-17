"""Unit tests for the disk-backed TTL cache."""

import time
from pathlib import Path

import pytest

from app.cache.ttl_cache import TTLCache
from app.utils.exceptions import CacheError


@pytest.fixture()
def cache(tmp_path: Path) -> TTLCache:
    return TTLCache(tmp_path / "cache", default_ttl_seconds=60)


class TestTTLCache:
    def test_set_get_roundtrip(self, cache: TTLCache) -> None:
        cache.set("key", {"results": [1, 2, 3]})
        assert cache.get("key") == {"results": [1, 2, 3]}

    def test_missing_key_returns_none(self, cache: TTLCache) -> None:
        assert cache.get("ghost") is None

    def test_expired_entry_is_deleted_on_read(self, cache: TTLCache, tmp_path: Path) -> None:
        cache.set("short", "value", ttl_seconds=0.05)
        time.sleep(0.1)
        assert cache.get("short") is None
        # The file must be physically gone — "never permanently stored".
        assert not list((tmp_path / "cache").glob("*.json"))

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        TTLCache(tmp_path / "c", default_ttl_seconds=60).set("k", "v")
        assert TTLCache(tmp_path / "c", default_ttl_seconds=60).get("k") == "v"

    def test_purge_expired_removes_only_stale(self, cache: TTLCache) -> None:
        cache.set("stale", "x", ttl_seconds=0.05)
        cache.set("fresh", "y", ttl_seconds=60)
        time.sleep(0.1)
        assert cache.purge_expired() == 1
        assert cache.get("fresh") == "y"

    def test_clear_removes_everything(self, cache: TTLCache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.clear() == 2
        assert cache.get("a") is None

    def test_stats(self, cache: TTLCache) -> None:
        cache.set("a", "x")
        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["size_bytes"] > 0
        assert stats["default_ttl_seconds"] == 60

    def test_non_serializable_value_raises(self, cache: TTLCache) -> None:
        with pytest.raises(CacheError):
            cache.set("bad", object())

    def test_corrupt_entry_treated_as_miss(self, cache: TTLCache, tmp_path: Path) -> None:
        cache.set("k", "v")
        for entry in (tmp_path / "cache").glob("*.json"):
            entry.write_text("{corrupt")
        assert cache.get("k") is None
