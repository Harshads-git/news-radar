"""
src/search.py
=============
DuckDuckGo web search for fetching background context before AI summarization.

Why web context matters:
  A story title like "OpenAI announces Project Strawberry" means nothing
  without context. Before summarizing, we query DuckDuckGo for background
  information, then inject it into the AI prompt. This lets the AI write
  summaries like:
    "Project Strawberry is OpenAI's internal codename for their reasoning
    model, later revealed as o1. This announcement introduces..."

  Rather than just:
    "OpenAI announces a new project."

Day 17 improvements over the original:
  1. DISK CACHE — persists between pipeline runs. Stories that were seen
     yesterday don't need a fresh DDG call today. Cache is stored at
     data/cache/search-YYYY-MM-DD.json with 24h TTL by default.

  2. PARALLEL BATCH — fetch_all_contexts() runs all queries concurrently
     but with a semaphore-controlled concurrency limit (_MAX_CONCURRENT=5)
     to avoid overwhelming DDG and triggering rate limits.

  3. DDGS TEXT FALLBACK — when the Instant Answer API returns empty, we
     fall back to a `ddgs.text()` search and grab the first result's body.
     This significantly improves context coverage for niche tech stories.

  4. CONTEXT STATS — returns a ContextStats dataclass so the orchestrator
     can log "12/15 stories enriched, 3 cache hits".

Usage:
    from src.search import fetch_web_context, fetch_all_contexts

    # Single story
    context = await fetch_web_context("OpenAI Project Strawberry reasoning model")

    # Batch — all stories in parallel with caching
    from pathlib import Path
    contexts = await fetch_all_contexts(
        ["query 1", "query 2", "query 3"],
        cache_dir=Path("data/cache"),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DDG_URL = "https://api.duckduckgo.com/"
_USER_AGENT = "NewsRadar/0.1 (+https://github.com/Harshads-git/news-radar)"
_REQUEST_TIMEOUT = 8.0
_MAX_CONTEXT_CHARS = 600
_MAX_CONCURRENT = 5          # Max parallel DDG requests
_CACHE_TTL_HOURS = 24        # Disk cache TTL in hours


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


@dataclass
class ContextStats:
    """Stats for one batch fetch_all_contexts() call."""
    total: int = 0
    enriched: int = 0          # non-empty context returned
    cache_hits: int = 0        # served from disk cache
    memory_hits: int = 0       # served from in-memory cache
    api_calls: int = 0         # actual DDG API calls made
    ddgs_fallback_used: int = 0  # times DDGS text() fallback was used
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queries served from cache (disk or memory)."""
        total_hits = self.cache_hits + self.memory_hits
        return total_hits / self.total if self.total else 0.0

    def summary(self) -> str:
        parts = [f"{self.enriched}/{self.total} enriched"]
        if self.cache_hits:
            parts.append(f"{self.cache_hits} disk cache hits")
        if self.memory_hits:
            parts.append(f"{self.memory_hits} memory hits")
        if self.api_calls:
            parts.append(f"{self.api_calls} API calls")
        if self.errors:
            parts.append(f"{self.errors} errors")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# In-memory cache (per-process, lives for one pipeline run)
# ---------------------------------------------------------------------------

_memory_cache: dict[str, str] = {}


def clear_context_cache() -> None:
    """Clear the in-memory context cache. Call between test runs."""
    _memory_cache.clear()


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class _DiskCache:
    """
    Simple JSON-based disk cache for DDG context results.

    File format: data/cache/search-YYYY-MM-DD.json
    Content: { "query_hash": { "context": "...", "cached_at": "ISO8601" } }

    Why JSON instead of SQLite?
      - Zero dependencies (no sqlalchemy, no sqlite3 schema migration)
      - Human readable — you can cat the file and see what's cached
      - Safe for single-process writes (one pipeline runs at a time)
      - Small enough — 300 queries * 600 chars ≈ 200KB per day
    """

    def __init__(self, cache_dir: Path | None = None, cache_date: date | None = None) -> None:
        self._dir = cache_dir
        self._date = cache_date or date.today()
        self._data: dict[str, dict] | None = None
        self._dirty = False

    @property
    def _path(self) -> Path | None:
        if self._dir is None:
            return None
        return self._dir / f"search-{self._date.isoformat()}.json"

    def _load(self) -> dict[str, dict]:
        if self._data is not None:
            return self._data
        path = self._path
        if path is None or not path.exists():
            self._data = {}
            return self._data
        try:
            self._data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}
        return self._data

    def get(self, key: str) -> str | None:
        """Return cached context or None if not found / expired."""
        data = self._load()
        entry = data.get(key)
        if entry is None:
            return None
        # Check TTL
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
            if age_hours > _CACHE_TTL_HOURS:
                return None
        except (KeyError, ValueError):
            return None
        return entry.get("context", "")

    def set(self, key: str, context: str) -> None:
        """Store context in the disk cache."""
        data = self._load()
        data[key] = {
            "context": context,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._dirty = True

    def flush(self) -> None:
        """Write dirty cache entries to disk."""
        if not self._dirty or self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("Failed to write search cache: %s", e)
        self._dirty = False

    def cleanup_old_files(self, keep_days: int = 7) -> None:
        """Remove cache files older than keep_days to save disk space."""
        if self._dir is None or not self._dir.exists():
            return
        for f in self._dir.glob("search-*.json"):
            try:
                file_date = date.fromisoformat(f.stem.replace("search-", ""))
                age_days = (self._date - file_date).days
                if age_days > keep_days:
                    f.unlink(missing_ok=True)
                    log.debug("Removed old search cache: %s", f.name)
            except (ValueError, OSError):
                pass


# ---------------------------------------------------------------------------
# Public API — single query
# ---------------------------------------------------------------------------


async def fetch_web_context(
    query: str,
    max_chars: int = _MAX_CONTEXT_CHARS,
    cache_dir: Path | None = None,
    _disk_cache: _DiskCache | None = None,
    _stats: ContextStats | None = None,
) -> str:
    """
    Fetch a short background context snippet from DuckDuckGo for the query.

    Lookup order:
      1. In-memory cache (fastest — same query seen earlier this run)
      2. Disk cache (avoids API call for queries seen in past 24h)
      3. DuckDuckGo Instant Answer API
      4. DDGS text() fallback (if Instant Answer returns empty)

    Parameters
    ----------
    query:
        Search query, typically the story title + key topic words.
    max_chars:
        Maximum length of the returned context string.
    cache_dir:
        Directory for disk cache. None disables disk caching.

    Returns
    -------
    str
        Plain-text background context (may be empty on failure or no result).

    Notes
    -----
    - Never raises exceptions — returns "" on any error.
    """
    if not query or not query.strip():
        return ""

    query = query.strip()
    cache_key = _hash_query(query)

    # 1. Memory cache
    if cache_key in _memory_cache:
        if _stats:
            _stats.memory_hits += 1
        return _memory_cache[cache_key]

    # 2. Disk cache
    disk = _disk_cache or (_DiskCache(cache_dir) if cache_dir else None)
    if disk is not None:
        cached = disk.get(cache_key)
        if cached is not None:
            _memory_cache[cache_key] = cached
            if _stats:
                _stats.cache_hits += 1
            return cached

    # 3. Live DDG call
    if _stats:
        _stats.api_calls += 1
    context = await _fetch_ddg_context(query, max_chars, _stats)

    # Store result (even empty — to avoid re-querying failed lookups)
    _memory_cache[cache_key] = context
    if disk is not None:
        disk.set(cache_key, context)

    return context


# ---------------------------------------------------------------------------
# Public API — batch (parallel with concurrency limit)
# ---------------------------------------------------------------------------


async def fetch_all_contexts(
    queries: Sequence[str],
    cache_dir: Path | None = None,
    max_chars: int = _MAX_CONTEXT_CHARS,
    max_concurrent: int = _MAX_CONCURRENT,
) -> tuple[list[str], ContextStats]:
    """
    Fetch context for multiple queries in parallel with a concurrency limit.

    Why a semaphore?
      Without a concurrency limit, 50 concurrent DDG requests would likely
      trigger rate limiting (HTTP 429). The semaphore ensures at most
      max_concurrent requests are in-flight at any time.

    Parameters
    ----------
    queries:
        List of search queries to fetch context for.
    cache_dir:
        Directory for disk cache. None disables disk caching.
    max_chars:
        Maximum chars per context snippet.
    max_concurrent:
        Maximum parallel DDG requests (default: 5).

    Returns
    -------
    (contexts, stats):
        - contexts: list of context strings in same order as queries
        - stats: ContextStats with hit rates and error counts
    """
    stats = ContextStats(total=len(queries))
    disk = _DiskCache(cache_dir) if cache_dir else None
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(query: str) -> str:
        async with semaphore:
            return await fetch_web_context(
                query,
                max_chars=max_chars,
                _disk_cache=disk,
                _stats=stats,
            )

    results = await asyncio.gather(
        *[_fetch_one(q) for q in queries],
        return_exceptions=True,
    )

    contexts: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            log.debug("Context fetch failed: %s", result)
            stats.errors += 1
            contexts.append("")
        else:
            ctx = str(result)
            if ctx:
                stats.enriched += 1
            contexts.append(ctx)

    # Flush disk cache once at the end (one write instead of N)
    if disk is not None:
        disk.flush()
        disk.cleanup_old_files()

    if stats.total > 0:
        log.info("Context enrichment: %s", stats.summary())

    return contexts, stats


# ---------------------------------------------------------------------------
# DDG Instant Answer API
# ---------------------------------------------------------------------------


async def _fetch_ddg_context(
    query: str,
    max_chars: int,
    _stats: ContextStats | None = None,
) -> str:
    """
    Query DuckDuckGo's Instant Answer API and extract the best snippet.
    Falls back to DDGS text() search if the Instant Answer returns empty.
    """
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "t": "NewsRadar",
    }

    data: dict = {}
    try:
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(_DDG_URL, params=params)

        if response.status_code == 200:
            data = response.json()
        else:
            log.debug("DDG returned HTTP %d for: %s", response.status_code, query[:60])

    except (httpx.TransportError, httpx.TimeoutException) as e:
        log.debug("DDG request failed: %s", e)
    except Exception as e:
        log.debug("DDG unexpected error: %s", e)

    # Try Instant Answer API result first
    context = _extract_best_text(data, max_chars)
    if context:
        log.debug("DDG instant-answer (%d chars): %s", len(context), query[:50])
        return context

    # Fallback: DDGS text() search (richer results for niche tech topics)
    context = await _fetch_ddgs_text(query, max_chars)
    if context:
        if _stats:
            _stats.ddgs_fallback_used += 1
        log.debug("DDG text-search (%d chars): %s", len(context), query[:50])

    return context


async def _fetch_ddgs_text(query: str, max_chars: int) -> str:
    """
    Use the DDGS (DuckDuckGo Search) library as a fallback for richer results.

    Runs in a thread executor to avoid blocking the event loop (ddgs is sync).
    Returns the body of the first result, or "" on failure.
    """
    try:
        from ddgs import DDGS

        def _search() -> str:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=1))
            if results:
                body = results[0].get("body", "")
                return body[:max_chars]
            return ""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except ImportError:
        log.debug("ddgs not installed — skipping text search fallback")
        return ""
    except Exception as e:
        log.debug("DDGS text search failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_best_text(data: dict, max_chars: int) -> str:
    """
    Extract the best available text snippet from a DDG API response dict.

    Priority order:
      1. AbstractText (Wikipedia abstract — richest context)
      2. Abstract (shorter abstract)
      3. Answer (direct answer for factual queries)
      4. First RelatedTopics text (secondary result)
    """
    # Priority 1: AbstractText
    if text := (data.get("AbstractText") or "").strip():
        return text[:max_chars]

    # Priority 2: Abstract
    if text := (data.get("Abstract") or "").strip():
        return text[:max_chars]

    # Priority 3: Answer
    if text := (data.get("Answer") or "").strip():
        return text[:max_chars]

    # Priority 4: First related topic
    topics = data.get("RelatedTopics") or []
    for topic in topics:
        if isinstance(topic, dict) and (text := (topic.get("Text") or "").strip()):
            return text[:max_chars]

    return ""


def _hash_query(query: str) -> str:
    """Return a short hash of the query string for use as a cache key."""
    return hashlib.md5(query.lower().encode(), usedforsecurity=False).hexdigest()[:16]
