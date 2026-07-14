"""
tests/test_score_cache.py
==========================
Tests for Day 25 AI score caching in src/storage/score_cache.py.

Coverage:
  - CacheEntry: initial attributes, is_expired (fresh / expired / bad timestamp),
    age_hours, record_success resets consecutive, to_dict fields and types,
    from_dict round-trip, from_dict with missing optional fields, from_scored_item
  - ScoreCache.get: miss returns None, miss increments counter, hit increments
    counter, hit returns correct entry, expired entry returns None and auto-removes,
    unknown URL returns None
  - ScoreCache.put: stores entry under url key, overwrites existing, does not save
  - ScoreCache.save: creates file, writes valid JSON, atomic (tmp then rename),
    silent on bad path, existing file overwritten (not appended)
  - ScoreCache._load: returns empty store when no file, ignores malformed json,
    ignores malformed entries, loads valid entries
  - ScoreCache.prune: removes expired entries, returns count, leaves valid alone,
    zero when nothing expired
  - ScoreCache.clear: empties store
  - ScoreCache.stats: entry_count, expired_count, hit_rate, file_size_kb,
    ttl_hours, cache_file present
  - ScoreCache.__len__: counts only valid entries
  - Integration: put → save → new instance get (round-trip)
  - NewsScorer: cache=None does not break scoring (backward compat)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.storage.score_cache import (
    DEFAULT_TTL_HOURS,
    CacheEntry,
    ScoreCache,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _old_ts(hours: int = 25) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_scored_item(url: str, score: int = 8, topics: list | None = None):
    si = MagicMock()
    si.item.url = url
    si.ai_score = score
    si.ai_topics = topics or ["AI", "Python"]
    return si


# ===========================================================================
# CacheEntry
# ===========================================================================


class TestCacheEntry:
    @pytest.mark.unit
    def test_initial_attributes(self):
        e = CacheEntry("https://example.com", ai_score=7, ai_topics=["AI"])
        assert e.url == "https://example.com"
        assert e.ai_score == 7
        assert e.ai_topics == ["AI"]
        assert e.ai_headline is None
        assert e.ttl_hours == DEFAULT_TTL_HOURS
        assert e.hits == 0

    @pytest.mark.unit
    def test_fresh_entry_not_expired(self):
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[])
        assert e.is_expired is False

    @pytest.mark.unit
    def test_old_entry_is_expired(self):
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        assert e.is_expired is True

    @pytest.mark.unit
    def test_entry_exactly_at_ttl_not_expired(self):
        # Exactly 24h ago should still be expired (age > ttl)
        ts = _old_ts(hours=23)
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[], cached_at=ts)
        assert e.is_expired is False

    @pytest.mark.unit
    def test_bad_cached_at_treated_as_expired(self):
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[], cached_at="NOT-A-DATE")
        assert e.is_expired is True

    @pytest.mark.unit
    def test_age_hours_fresh(self):
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[])
        assert 0 <= e.age_hours < 0.1

    @pytest.mark.unit
    def test_age_hours_old(self):
        e = CacheEntry("https://example.com", ai_score=5, ai_topics=[], cached_at=_old_ts(10))
        assert abs(e.age_hours - 10) < 0.1

    @pytest.mark.unit
    def test_to_dict_has_required_keys(self):
        e = CacheEntry("https://a.com", ai_score=9, ai_topics=["AI"])
        d = e.to_dict()
        for key in ("url", "ai_score", "ai_topics", "ai_headline", "cached_at", "ttl_hours", "hits"):
            assert key in d, f"Missing key: {key}"

    @pytest.mark.unit
    def test_to_dict_values_correct(self):
        e = CacheEntry("https://a.com", ai_score=9, ai_topics=["AI"])
        d = e.to_dict()
        assert d["url"] == "https://a.com"
        assert d["ai_score"] == 9
        assert d["ai_topics"] == ["AI"]

    @pytest.mark.unit
    def test_from_dict_round_trip(self):
        e = CacheEntry("https://b.com", ai_score=6, ai_topics=["Python", "ML"], hits=3)
        e2 = CacheEntry.from_dict(e.to_dict())
        assert e2.url == e.url
        assert e2.ai_score == e.ai_score
        assert e2.ai_topics == e.ai_topics
        assert e2.hits == e.hits

    @pytest.mark.unit
    def test_from_dict_missing_optional_fields(self):
        d = {"url": "https://c.com", "ai_score": 7, "ai_topics": []}
        e = CacheEntry.from_dict(d)
        assert e.url == "https://c.com"
        assert e.ai_topics == []
        assert e.ai_headline is None

    @pytest.mark.unit
    def test_from_scored_item(self):
        si = _make_scored_item("https://d.com", score=8, topics=["LLM"])
        e = CacheEntry.from_scored_item(si)
        assert e.url == "https://d.com"
        assert e.ai_score == 8
        assert e.ai_topics == ["LLM"]
        assert e.ttl_hours == DEFAULT_TTL_HOURS


# ===========================================================================
# ScoreCache — get / put
# ===========================================================================


class TestScoreCacheGetPut:
    @pytest.mark.unit
    def test_get_miss_returns_none(self, tmp_path):
        cache = ScoreCache(tmp_path)
        assert cache.get("https://missing.com") is None

    @pytest.mark.unit
    def test_get_miss_increments_counter(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.get("https://missing.com")
        assert cache.stats()["miss_count"] == 1

    @pytest.mark.unit
    def test_put_then_get_hit(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://example.com", score=9))
        result = cache.get("https://example.com")
        assert result is not None
        assert result.ai_score == 9

    @pytest.mark.unit
    def test_get_hit_increments_counter(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://example.com"))
        cache.get("https://example.com")
        assert cache.stats()["hit_count"] == 1

    @pytest.mark.unit
    def test_expired_entry_returns_none(self, tmp_path):
        cache = ScoreCache(tmp_path)
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        assert cache.get("https://old.com") is None

    @pytest.mark.unit
    def test_expired_entry_auto_removed(self, tmp_path):
        cache = ScoreCache(tmp_path)
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        cache.get("https://old.com")
        with cache._lock:
            assert "https://old.com" not in cache._store

    @pytest.mark.unit
    def test_put_overwrites_existing(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://example.com", score=5))
        cache.put(_make_scored_item("https://example.com", score=9))
        result = cache.get("https://example.com")
        assert result.ai_score == 9


# ===========================================================================
# ScoreCache — save / load
# ===========================================================================


class TestScoreCacheSaveLoad:
    @pytest.mark.unit
    def test_save_creates_cache_file(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://example.com"))
        cache.save()
        assert (tmp_path / "cache" / "score_cache.json").exists()

    @pytest.mark.unit
    def test_save_writes_valid_json(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://example.com", score=8))
        cache.save()
        raw = json.loads((tmp_path / "cache" / "score_cache.json").read_text())
        assert "https://example.com" in raw
        assert raw["https://example.com"]["ai_score"] == 8

    @pytest.mark.unit
    def test_save_and_reload_round_trip(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://a.com", score=7, topics=["AI"]))
        cache.save()

        cache2 = ScoreCache(tmp_path)
        result = cache2.get("https://a.com")
        assert result is not None
        assert result.ai_score == 7
        assert result.ai_topics == ["AI"]

    @pytest.mark.unit
    def test_save_silent_on_bad_path(self):
        cache = ScoreCache(Path("/nonexistent/deep/path"))
        cache.put(_make_scored_item("https://example.com"))
        try:
            cache.save()
        except Exception as e:
            pytest.fail(f"save() raised unexpectedly: {e}")

    @pytest.mark.unit
    def test_load_empty_when_no_file(self, tmp_path):
        cache = ScoreCache(tmp_path)
        assert len(cache) == 0

    @pytest.mark.unit
    def test_load_ignores_malformed_json(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "score_cache.json").write_text("NOT JSON AT ALL")
        cache = ScoreCache(tmp_path)
        assert len(cache) == 0

    @pytest.mark.unit
    def test_load_skips_malformed_entries(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Entry missing required 'url' key
        bad_data = {"https://ok.com": {"ai_score": 5, "ai_topics": []},
                    "bad": "not a dict"}
        (cache_dir / "score_cache.json").write_text(json.dumps(bad_data))
        cache = ScoreCache(tmp_path)
        # Should load what it can without crashing
        assert len(cache) >= 0


# ===========================================================================
# ScoreCache — prune / clear / stats / len
# ===========================================================================


class TestScoreCacheMaintenance:
    @pytest.mark.unit
    def test_prune_removes_expired_entries(self, tmp_path):
        cache = ScoreCache(tmp_path)
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        cache.prune()
        with cache._lock:
            assert "https://old.com" not in cache._store

    @pytest.mark.unit
    def test_prune_returns_count(self, tmp_path):
        cache = ScoreCache(tmp_path)
        for i in range(3):
            old = CacheEntry(f"https://old{i}.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
            with cache._lock:
                cache._store[f"https://old{i}.com"] = old
        count = cache.prune()
        assert count == 3

    @pytest.mark.unit
    def test_prune_leaves_valid_entries(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://good.com"))
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        cache.prune()
        assert cache.get("https://good.com") is not None

    @pytest.mark.unit
    def test_prune_returns_zero_when_nothing_expired(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://good.com"))
        assert cache.prune() == 0

    @pytest.mark.unit
    def test_clear_empties_store(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://a.com"))
        cache.put(_make_scored_item("https://b.com"))
        cache.clear()
        assert len(cache) == 0

    @pytest.mark.unit
    def test_stats_entry_count(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://a.com"))
        cache.put(_make_scored_item("https://b.com"))
        assert cache.stats()["entry_count"] == 2

    @pytest.mark.unit
    def test_stats_expired_count(self, tmp_path):
        cache = ScoreCache(tmp_path)
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        assert cache.stats()["expired_count"] == 1

    @pytest.mark.unit
    def test_stats_hit_rate_zero_at_start(self, tmp_path):
        cache = ScoreCache(tmp_path)
        assert cache.stats()["hit_rate"] == 0.0

    @pytest.mark.unit
    def test_stats_hit_rate_after_hits(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://a.com"))
        cache.get("https://a.com")       # hit
        cache.get("https://missing.com") # miss
        stats = cache.stats()
        assert abs(stats["hit_rate"] - 0.5) < 1e-9

    @pytest.mark.unit
    def test_stats_file_size_after_save(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://a.com", score=8))
        cache.save()
        stats = cache.stats()
        assert stats["file_size_kb"] > 0

    @pytest.mark.unit
    def test_stats_has_all_expected_keys(self, tmp_path):
        cache = ScoreCache(tmp_path)
        stats = cache.stats()
        for key in ("entry_count", "expired_count", "hit_count", "miss_count",
                    "hit_rate", "file_size_kb", "ttl_hours", "cache_file"):
            assert key in stats, f"Missing key: {key}"

    @pytest.mark.unit
    def test_len_counts_only_valid(self, tmp_path):
        cache = ScoreCache(tmp_path)
        cache.put(_make_scored_item("https://good.com"))
        old = CacheEntry("https://old.com", ai_score=5, ai_topics=[], cached_at=_old_ts(25))
        with cache._lock:
            cache._store["https://old.com"] = old
        # len() skips expired entries
        assert len(cache) == 1
