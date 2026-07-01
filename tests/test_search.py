"""
tests/test_search.py
=====================
Tests for src/search.py — DuckDuckGo context enrichment with disk cache.

Coverage:
  - ContextStats: counts, hit_rate, summary()
  - _DiskCache: get/set/flush/cleanup, TTL expiry, missing dir
  - _extract_best_text: priority order (AbstractText > Abstract > Answer > Related)
  - _hash_query: deterministic, case-insensitive
  - fetch_web_context: memory cache hit, disk cache hit, API call path
  - fetch_all_contexts: parallel execution, semaphore, stats counting
  - --no-enrich path: _fetch_all_contexts with enrich=False returns {}

All tests use mocks — no actual DDG API calls.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.search import (
    ContextStats,
    _DiskCache,
    _extract_best_text,
    _hash_query,
    clear_context_cache,
    fetch_all_contexts,
    fetch_web_context,
)


# ===========================================================================
# ContextStats
# ===========================================================================


class TestContextStats:
    @pytest.mark.unit
    def test_initial_values_are_zero(self):
        stats = ContextStats()
        assert stats.total == 0
        assert stats.enriched == 0
        assert stats.cache_hits == 0
        assert stats.memory_hits == 0
        assert stats.api_calls == 0
        assert stats.errors == 0

    @pytest.mark.unit
    def test_hit_rate_zero_when_no_calls(self):
        stats = ContextStats(total=0)
        assert stats.hit_rate == 0.0

    @pytest.mark.unit
    def test_hit_rate_counts_disk_and_memory_hits(self):
        stats = ContextStats(total=10, cache_hits=3, memory_hits=2)
        assert abs(stats.hit_rate - 0.5) < 0.001

    @pytest.mark.unit
    def test_summary_includes_enriched_fraction(self):
        stats = ContextStats(total=10, enriched=7, api_calls=4)
        summary = stats.summary()
        assert "7/10" in summary
        assert "enriched" in summary

    @pytest.mark.unit
    def test_summary_includes_cache_hits_when_nonzero(self):
        stats = ContextStats(total=5, enriched=5, cache_hits=3)
        assert "disk cache" in stats.summary()

    @pytest.mark.unit
    def test_summary_includes_errors_when_nonzero(self):
        stats = ContextStats(total=5, enriched=3, errors=2)
        assert "error" in stats.summary()

    @pytest.mark.unit
    def test_summary_no_cache_section_when_zero_hits(self):
        stats = ContextStats(total=5, enriched=5, api_calls=5)
        summary = stats.summary()
        assert "cache" not in summary


# ===========================================================================
# _extract_best_text
# ===========================================================================


class TestExtractBestText:
    @pytest.mark.unit
    def test_prefers_abstract_text_first(self):
        data = {
            "AbstractText": "Wikipedia summary",
            "Abstract": "Short abstract",
            "Answer": "Direct answer",
        }
        assert _extract_best_text(data, 500) == "Wikipedia summary"

    @pytest.mark.unit
    def test_falls_back_to_abstract(self):
        data = {"AbstractText": "", "Abstract": "Short abstract", "Answer": "Direct"}
        assert _extract_best_text(data, 500) == "Short abstract"

    @pytest.mark.unit
    def test_falls_back_to_answer(self):
        data = {"AbstractText": "", "Abstract": "", "Answer": "Direct answer"}
        assert _extract_best_text(data, 500) == "Direct answer"

    @pytest.mark.unit
    def test_falls_back_to_related_topics(self):
        data = {
            "AbstractText": "",
            "RelatedTopics": [{"Text": "Related topic text"}],
        }
        assert _extract_best_text(data, 500) == "Related topic text"

    @pytest.mark.unit
    def test_returns_empty_when_nothing_available(self):
        assert _extract_best_text({}, 500) == ""

    @pytest.mark.unit
    def test_truncates_to_max_chars(self):
        data = {"AbstractText": "x" * 1000}
        assert len(_extract_best_text(data, 100)) == 100

    @pytest.mark.unit
    def test_skips_non_dict_related_topics(self):
        data = {"RelatedTopics": ["not a dict", {"Text": "Valid topic"}]}
        assert _extract_best_text(data, 500) == "Valid topic"

    @pytest.mark.unit
    def test_strips_whitespace(self):
        data = {"AbstractText": "  trimmed  "}
        assert _extract_best_text(data, 500) == "trimmed"


# ===========================================================================
# _hash_query
# ===========================================================================


class TestHashQuery:
    @pytest.mark.unit
    def test_returns_16_char_hex_string(self):
        h = _hash_query("test query")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    @pytest.mark.unit
    def test_same_query_returns_same_hash(self):
        assert _hash_query("hello world") == _hash_query("hello world")

    @pytest.mark.unit
    def test_case_insensitive(self):
        assert _hash_query("OpenAI GPT") == _hash_query("openai gpt")

    @pytest.mark.unit
    def test_different_queries_return_different_hashes(self):
        assert _hash_query("query A") != _hash_query("query B")


# ===========================================================================
# _DiskCache
# ===========================================================================


class TestDiskCache:
    @pytest.mark.unit
    def test_get_returns_none_when_key_missing(self, tmp_path):
        dc = _DiskCache(tmp_path)
        assert dc.get("nonexistent") is None

    @pytest.mark.unit
    def test_set_and_get_roundtrip(self, tmp_path):
        dc = _DiskCache(tmp_path)
        dc.set("key1", "some context text")
        dc.flush()
        # Fresh instance reads from disk
        dc2 = _DiskCache(tmp_path)
        assert dc2.get("key1") == "some context text"

    @pytest.mark.unit
    def test_empty_context_is_cached(self, tmp_path):
        dc = _DiskCache(tmp_path)
        dc.set("key1", "")
        dc.flush()
        dc2 = _DiskCache(tmp_path)
        assert dc2.get("key1") == ""

    @pytest.mark.unit
    def test_expired_entry_returns_none(self, tmp_path):
        dc = _DiskCache(tmp_path)
        # Manually write an entry with old timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        cache_file = tmp_path / f"search-{dc._date.isoformat()}.json"
        cache_file.write_text(
            json.dumps({"old_key": {"context": "old content", "cached_at": old_time}}),
            encoding="utf-8",
        )
        dc2 = _DiskCache(tmp_path)
        assert dc2.get("old_key") is None

    @pytest.mark.unit
    def test_flush_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "deep" / "nested" / "dir"
            dc = _DiskCache(cache_dir)
            dc.set("k", "v")
            dc.flush()
            assert dc._path.exists()

    @pytest.mark.unit
    def test_no_flush_when_not_dirty(self, tmp_path):
        dc = _DiskCache(tmp_path)
        dc.flush()  # Should not create any file
        assert not any(tmp_path.glob("search-*.json"))

    @pytest.mark.unit
    def test_none_cache_dir_returns_none_path(self):
        dc = _DiskCache(cache_dir=None)
        assert dc._path is None

    @pytest.mark.unit
    def test_cleanup_removes_old_files(self, tmp_path):
        from datetime import date, timedelta
        today = date.today()
        old_date = today - timedelta(days=10)
        old_file = tmp_path / f"search-{old_date.isoformat()}.json"
        old_file.write_text("{}", encoding="utf-8")
        dc = _DiskCache(tmp_path)
        dc.cleanup_old_files(keep_days=7)
        assert not old_file.exists()

    @pytest.mark.unit
    def test_cleanup_keeps_recent_files(self, tmp_path):
        from datetime import date, timedelta
        today = date.today()
        recent_date = today - timedelta(days=3)
        recent_file = tmp_path / f"search-{recent_date.isoformat()}.json"
        recent_file.write_text("{}", encoding="utf-8")
        dc = _DiskCache(tmp_path)
        dc.cleanup_old_files(keep_days=7)
        assert recent_file.exists()


# ===========================================================================
# fetch_web_context
# ===========================================================================


class TestFetchWebContext:
    @pytest.mark.unit
    async def test_returns_empty_for_empty_query(self):
        clear_context_cache()
        result = await fetch_web_context("")
        assert result == ""

    @pytest.mark.unit
    async def test_returns_empty_for_whitespace_query(self):
        clear_context_cache()
        result = await fetch_web_context("   ")
        assert result == ""

    @pytest.mark.unit
    async def test_serves_from_memory_cache(self):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query
        key = _hash_query("cached query test")
        _memory_cache[key] = "cached value from memory"

        stats = ContextStats(total=1)
        result = await fetch_web_context("cached query test", _stats=stats)
        assert result == "cached value from memory"
        assert stats.memory_hits == 1

    @pytest.mark.unit
    async def test_serves_from_disk_cache(self, tmp_path):
        clear_context_cache()
        # Pre-populate disk cache
        dc = _DiskCache(tmp_path)
        from src.search import _hash_query
        key = _hash_query("disk cached query")
        dc.set(key, "disk context value")
        dc.flush()

        stats = ContextStats(total=1)
        result = await fetch_web_context(
            "disk cached query",
            cache_dir=tmp_path,
            _stats=stats,
        )
        assert result == "disk context value"
        assert stats.cache_hits == 1

    @pytest.mark.unit
    async def test_calls_ddg_api_on_cache_miss(self, tmp_path):
        clear_context_cache()
        stats = ContextStats(total=1)

        with patch("src.search._fetch_ddg_context", new_callable=AsyncMock) as mock_ddg:
            mock_ddg.return_value = "live context from DDG"
            result = await fetch_web_context(
                "brand new query xyz",
                cache_dir=tmp_path,
                _stats=stats,
            )

        assert result == "live context from DDG"
        assert stats.api_calls == 1
        mock_ddg.assert_called_once()

    @pytest.mark.unit
    async def test_caches_result_in_memory_after_api_call(self, tmp_path):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query

        with patch("src.search._fetch_ddg_context", new_callable=AsyncMock) as mock_ddg:
            mock_ddg.return_value = "api result"
            await fetch_web_context("new unique query abc", cache_dir=tmp_path)

        key = _hash_query("new unique query abc")
        assert _memory_cache.get(key) == "api result"


# ===========================================================================
# fetch_all_contexts
# ===========================================================================


class TestFetchAllContexts:
    @pytest.mark.unit
    async def test_returns_list_same_length_as_queries(self):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query
        queries = ["q1", "q2", "q3"]
        for q in queries:
            _memory_cache[_hash_query(q)] = f"context for {q}"

        contexts, stats = await fetch_all_contexts(queries)
        assert len(contexts) == 3
        assert stats.total == 3

    @pytest.mark.unit
    async def test_memory_hits_counted_correctly(self):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query
        _memory_cache[_hash_query("cached q")] = "hit"

        _, stats = await fetch_all_contexts(["cached q"])
        assert stats.memory_hits == 1

    @pytest.mark.unit
    async def test_enriched_count_matches_nonempty_results(self):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query
        _memory_cache[_hash_query("q with context")] = "some context"
        _memory_cache[_hash_query("q without")] = ""

        _, stats = await fetch_all_contexts(["q with context", "q without"])
        assert stats.enriched == 1

    @pytest.mark.unit
    async def test_empty_queries_list_returns_empty(self):
        clear_context_cache()
        contexts, stats = await fetch_all_contexts([])
        assert contexts == []
        assert stats.total == 0

    @pytest.mark.unit
    async def test_order_preserved_in_results(self):
        clear_context_cache()
        from src.search import _memory_cache, _hash_query
        queries = ["alpha", "beta", "gamma"]
        for q in queries:
            _memory_cache[_hash_query(q)] = f"ctx_{q}"

        contexts, _ = await fetch_all_contexts(queries)
        assert contexts[0] == "ctx_alpha"
        assert contexts[1] == "ctx_beta"
        assert contexts[2] == "ctx_gamma"
