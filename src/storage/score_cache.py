"""
src/storage/score_cache.py
===========================
URL-keyed disk cache for AI scoring results.

Avoids re-calling the AI API for news items seen in the last 24 hours.
On a typical run, 30-50% of stories from persistent sources (HN, Reddit)
will appear again within 24 hours, making this cache a meaningful cost
and latency saving.

Cache format: ``data/cache/score_cache.json``

    {
      "https://example.com/story": {
        "url": "https://example.com/story",
        "ai_score": 8,
        "ai_summary": null,
        "ai_topics": ["AI", "OpenAI"],
        "ai_headline": null,
        "cached_at": "2026-07-14T06:00:00Z",
        "ttl_hours": 24
      },
      ...
    }

Design decisions:
  - Single JSON file (not JSONL): scores are keyed by URL for O(1) lookup,
    and the total file is small (scores are tiny). JSONL would require a
    linear scan. If the file grows large (>5000 entries), `prune()` removes
    expired entries.
  - Thread-safe: a threading.Lock guards all reads and writes.
  - Atomic save: always write to a temp file then rename so a crash during
    write doesn't corrupt the cache.

Usage:
    cache = ScoreCache(data_dir)
    result = cache.get("https://example.com/story")
    if result is None:
        result = await provider.score_item(item, ...)
        cache.put(result)
    cache.save()

    # Maintenance:
    pruned = cache.prune()    # remove expired entries
    stats = cache.stats()     # dict with hit_rate, entry_count, etc.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import ScoredItem

# Default TTL for cached scores (in hours)
DEFAULT_TTL_HOURS: int = 24

# Filename inside data/cache/
_CACHE_FILENAME = "score_cache.json"


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class CacheEntry:
    """
    A single cached scoring result keyed by URL.

    Attributes
    ----------
    url: str
        The canonical URL of the news item.
    ai_score: int
        The raw AI relevance score (0–10).
    ai_topics: list[str]
        AI-extracted topics for this item.
    ai_headline: str | None
        AI-generated headline (if available from scorer).
    cached_at: str
        ISO 8601 UTC timestamp of when this entry was written.
    ttl_hours: int
        How long this entry is valid, in hours.
    hits: int
        How many times this cached entry was returned (for analytics).
    """

    __slots__ = ("url", "ai_score", "ai_topics", "ai_headline", "cached_at", "ttl_hours", "hits")

    def __init__(
        self,
        url: str,
        ai_score: int,
        ai_topics: list[str],
        ai_headline: Optional[str] = None,
        cached_at: Optional[str] = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        hits: int = 0,
    ) -> None:
        self.url = url
        self.ai_score = ai_score
        self.ai_topics = ai_topics
        self.ai_headline = ai_headline
        self.cached_at = cached_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.ttl_hours = ttl_hours
        self.hits = hits

    @property
    def is_expired(self) -> bool:
        """True if this entry is older than ttl_hours."""
        try:
            cached_dt = datetime.fromisoformat(self.cached_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - cached_dt).total_seconds() / 3600
            return age_hours > self.ttl_hours
        except (ValueError, AttributeError):
            return True  # unparseable → treat as expired

    @property
    def age_hours(self) -> float:
        """Age of this entry in hours. Returns 0 on parse error."""
        try:
            cached_dt = datetime.fromisoformat(self.cached_at.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - cached_dt).total_seconds() / 3600
        except (ValueError, AttributeError):
            return 0.0

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "ai_score": self.ai_score,
            "ai_topics": self.ai_topics,
            "ai_headline": self.ai_headline,
            "cached_at": self.cached_at,
            "ttl_hours": self.ttl_hours,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            url=d["url"],
            ai_score=int(d.get("ai_score", 0)),
            ai_topics=list(d.get("ai_topics", [])),
            ai_headline=d.get("ai_headline"),
            cached_at=d.get("cached_at"),
            ttl_hours=int(d.get("ttl_hours", DEFAULT_TTL_HOURS)),
            hits=int(d.get("hits", 0)),
        )

    @classmethod
    def from_scored_item(
        cls,
        scored_item: "ScoredItem",
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> "CacheEntry":
        """Create a CacheEntry from a ScoredItem returned by the AI provider."""
        return cls(
            url=scored_item.item.url,
            ai_score=scored_item.ai_score,
            ai_topics=list(scored_item.ai_topics or []),
            ai_headline=None,  # headline comes from summarizer, not scorer
            ttl_hours=ttl_hours,
        )


# ---------------------------------------------------------------------------
# ScoreCache
# ---------------------------------------------------------------------------


class ScoreCache:
    """
    URL-keyed disk cache for AI scoring results with 24h TTL.

    Thread-safe: all mutations are guarded by a threading.Lock.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. Path("data")). The cache file is
        written to ``data_dir / "cache" / "score_cache.json"``.
    ttl_hours:
        How long entries remain valid. Default: 24 hours.
    """

    def __init__(self, data_dir: Path, ttl_hours: int = DEFAULT_TTL_HOURS) -> None:
        self._cache_dir = Path(data_dir) / "cache"
        self._path = self._cache_dir / _CACHE_FILENAME
        self._ttl_hours = ttl_hours
        self._lock = threading.Lock()

        # Runtime counters (not persisted)
        self._hits = 0
        self._misses = 0

        # In-memory store: url → CacheEntry
        self._store: dict[str, CacheEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, url: str) -> Optional[CacheEntry]:
        """
        Look up a URL in the cache.

        Returns the entry if it exists and has not expired.
        Returns None on a miss or if the entry is expired (auto-removed).

        Also increments hit/miss counters for analytics.
        """
        with self._lock:
            entry = self._store.get(url)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[url]
                self._misses += 1
                return None
            entry.hits += 1
            self._hits += 1
            return entry

    def put(self, scored_item: "ScoredItem") -> None:
        """
        Store a ScoredItem result in the cache under its URL key.

        Overwrites any existing entry for the same URL.
        Does NOT flush to disk — call save() when done.
        """
        entry = CacheEntry.from_scored_item(scored_item, ttl_hours=self._ttl_hours)
        with self._lock:
            self._store[entry.url] = entry

    def save(self) -> None:
        """
        Atomically write the in-memory cache to disk.

        Uses a temp-file + rename strategy so a crash during write does
        not corrupt the existing cache file. Silently ignores I/O errors
        so a save failure never aborts a pipeline run.
        """
        with self._lock:
            data = {url: entry.to_dict() for url, entry in self._store.items()}

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)
        except OSError:
            pass  # Never crash the pipeline over a cache write

    def prune(self) -> int:
        """
        Remove all expired entries from the in-memory store.

        Returns the number of entries removed. Does NOT save to disk —
        call save() after pruning if you want to persist the pruned cache.
        """
        with self._lock:
            expired_urls = [url for url, entry in self._store.items() if entry.is_expired]
            for url in expired_urls:
                del self._store[url]
        return len(expired_urls)

    def clear(self) -> None:
        """Remove all entries from the in-memory store (does not save to disk)."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        """
        Return a summary of cache performance and size.

        Returns
        -------
        dict with:
          - entry_count: int   — number of valid (non-expired) entries
          - expired_count: int — number of entries that are expired
          - hit_count: int     — cache hits since this instance was created
          - miss_count: int    — cache misses since this instance was created
          - hit_rate: float    — hit_count / (hit_count + miss_count), or 0.0
          - file_size_kb: float — size of the cache file on disk in KB
          - ttl_hours: int     — configured TTL
          - cache_file: str    — path to the cache file
        """
        with self._lock:
            valid = sum(1 for e in self._store.values() if not e.is_expired)
            expired = len(self._store) - valid
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

        file_size_kb = 0.0
        try:
            if self._path.exists():
                file_size_kb = self._path.stat().st_size / 1024
        except OSError:
            pass

        return {
            "entry_count": valid,
            "expired_count": expired,
            "hit_count": self._hits,
            "miss_count": self._misses,
            "hit_rate": hit_rate,
            "file_size_kb": round(file_size_kb, 2),
            "ttl_hours": self._ttl_hours,
            "cache_file": str(self._path),
        }

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._store.values() if not e.is_expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the cache file from disk into memory. Silently handles errors."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for url, d in raw.items():
                try:
                    self._store[url] = CacheEntry.from_dict(d)
                except (KeyError, TypeError, ValueError):
                    continue  # skip malformed entries
        except (OSError, json.JSONDecodeError):
            pass  # start with an empty cache on parse error
