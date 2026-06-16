"""
src/scrapers/base.py
====================
Abstract base class for all News Radar scrapers.

Every source type (RSS, Hacker News, Reddit, GitHub) implements this
interface. The orchestrator only knows about BaseScraper — it never
imports concrete scraper classes directly. This is the Open/Closed
principle: the system is open for extension (add a new scraper) but
closed for modification (don't touch the orchestrator).

Architecture:
    BaseScraper          ← abstract interface (this file)
        RssScraper       ← feedparser + httpx (Day 5)
        HackerNewsScraper← HN Firebase API (Day 6)
        RedditScraper    ← Reddit JSON API (Day 7)

The ScraperFactory (Day 8) uses SourceConfig.type to instantiate
the correct subclass. Callers always use:

    items = await scraper.fetch(source_config)

The returned list[NewsItem] is guaranteed to be:
  - Non-None (empty list on zero results)
  - Deduplicated within a single source (same URL appears at most once)
  - Sorted by published_at descending where the source provides dates
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import NewsItem, SourceConfig


class BaseScraper(ABC):
    """
    Abstract base class for all news source scrapers.

    Subclasses MUST implement ``fetch()``. They MAY override
    ``validate_source()`` if the source config has special requirements
    beyond what SourceConfig already validates.

    All scrapers are stateless — no instance variables are set after
    ``__init__``. This means a single scraper instance can safely
    process multiple sources concurrently.
    """

    # Override in subclasses to identify the scraper in logs
    SOURCE_TYPE: str = "base"

    def __init__(self) -> None:
        from src.logger import get_logger

        self.log = get_logger(f"src.scrapers.{self.SOURCE_TYPE}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch(self, source: "SourceConfig") -> list["NewsItem"]:
        """
        Fetch news items from the given source configuration.

        Parameters
        ----------
        source:
            A validated SourceConfig from sources.json. The scraper must
            respect ``source.limit`` and ``source.enabled``.

        Returns
        -------
        list[NewsItem]
            A list of fetched news items. Returns an empty list (not None)
            on zero results. Raises ``FetchError`` on unrecoverable errors.

        Raises
        ------
        FetchError
            If the source cannot be reached or the response cannot be parsed.
        RateLimitError
            If the source API returns HTTP 429.
        ParseError
            If the response is malformed or doesn't match the expected schema.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all scrapers
    # ------------------------------------------------------------------

    def validate_source(self, source: "SourceConfig") -> None:
        """
        Validate that the SourceConfig is compatible with this scraper.

        Called automatically at the start of fetch(). Override in subclasses
        to add scraper-specific validation (e.g., check required URL field).

        Raises
        ------
        ConfigError
            If the source config is missing a required field for this scraper.
        """
        if source.type != self.SOURCE_TYPE:
            from src.exceptions import ConfigError

            raise ConfigError(
                f"Scraper '{self.SOURCE_TYPE}' cannot handle source type '{source.type}'",
                field="type",
                expected=self.SOURCE_TYPE,
            )

    def _log_fetch_start(self, source: "SourceConfig") -> None:
        """Log a standard start-of-fetch message."""
        self.log.info(
            "Fetching [source]%s[/source] (limit=%d)",
            source.name,
            source.limit,
        )

    def _log_fetch_done(self, source: "SourceConfig", count: int) -> None:
        """Log a standard end-of-fetch message with item count."""
        self.log.info(
            "[source]%s[/source] → [count]%d[/count] items",
            source.name,
            count,
        )

    def _deduplicate(self, items: list["NewsItem"]) -> list["NewsItem"]:
        """
        Remove duplicate items within a single fetch result.

        Uses URL equality (same logic as NewsItem.__eq__) so that a story
        appearing twice in the same feed is reduced to one entry.

        This is intra-source dedup only. Cross-source dedup happens in
        the dedicated Deduplicator (Day 9).
        """
        seen: set[str] = set()
        unique: list[NewsItem] = []
        for item in items:
            if item.url not in seen:
                seen.add(item.url)
                unique.append(item)
        return unique

    async def fetch_safe(self, source: "SourceConfig") -> list["NewsItem"]:
        """
        Fault-tolerant wrapper around ``fetch()``.

        Returns an empty list instead of propagating exceptions. Logs
        the error at ERROR level. Used by the orchestrator when it wants
        a best-effort result from all sources.
        """
        try:
            return await self.fetch(source)
        except Exception as e:
            self.log.error(
                "Scraper [source]%s[/source] failed: %s",
                source.name,
                e,
            )
            return []
