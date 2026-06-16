"""
src/scrapers/rss.py
===================
RSS/Atom feed scraper using feedparser + httpx.

Architecture decision: why fetch the feed ourselves instead of letting
feedparser do it?
  - httpx gives us async I/O (feedparser.parse() is synchronous + blocking)
  - We can set proper User-Agent, timeouts, and headers
  - We get access to HTTP status codes for rate-limit detection
  - We can run multiple feed fetches concurrently with asyncio.gather()

Flow:
  1. httpx.AsyncClient fetches the raw feed bytes
  2. feedparser.parse() converts bytes → FeedParserDict
  3. _entry_to_news_item() maps each entry → NewsItem
  4. BaseScraper._deduplicate() removes within-feed duplicates
  5. Result trimmed to source.limit
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.exceptions import FetchError, ParseError, RateLimitError
from src.models import NewsItem, SourceConfig
from src.scrapers.base import BaseScraper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mimic a real browser to avoid 403s from protective feeds
_USER_AGENT = (
    "Mozilla/5.0 (compatible; NewsRadar/0.1; "
    "+https://github.com/Harshads-git/news-radar)"
)
_REQUEST_TIMEOUT = 15.0   # seconds
_MAX_RETRIES = 3
_RETRY_MIN_WAIT = 2       # seconds
_RETRY_MAX_WAIT = 30      # seconds


class RssScraper(BaseScraper):
    """
    Scraper for RSS and Atom feeds.

    Compatible with any standard RSS 2.0 / Atom feed URL. feedparser
    handles format detection automatically, so the same scraper works
    for both feed types.

    Retry policy (tenacity):
        - Up to 3 attempts
        - Exponential backoff: 2s → 4s → 8s (max 30s)
        - Retries on: network errors, HTTP 5xx, HTTP 429
        - Does NOT retry on: HTTP 4xx (except 429), parse errors
    """

    SOURCE_TYPE = "rss"

    def __init__(self) -> None:
        super().__init__()
        # Shared async client — created lazily per fetch call
        # (avoids keeping a persistent connection across runs)
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch(self, source: SourceConfig) -> list[NewsItem]:
        """
        Fetch and parse an RSS/Atom feed, returning a list of NewsItems.

        Parameters
        ----------
        source:
            Must have ``type == "rss"`` and a valid ``url``.

        Returns
        -------
        list[NewsItem]
            Up to ``source.limit`` items, deduplicated by URL,
            sorted by published_at descending.
        """
        self.validate_source(source)
        self._log_fetch_start(source)

        try:
            raw_bytes = await self._fetch_with_retry(source.url or "")
        except RetryError as e:
            raise FetchError(
                f"Failed to fetch RSS feed after {_MAX_RETRIES} attempts",
                source_id=source.id,
                url=source.url or "",
            ) from e

        items = self._parse_feed(raw_bytes, source)
        items = self._deduplicate(items)
        items = items[: source.limit]

        # Sort newest-first where dates are available
        items.sort(
            key=lambda x: x.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        self._log_fetch_done(source, len(items))
        return items

    # ------------------------------------------------------------------
    # HTTP layer (with tenacity retry)
    # ------------------------------------------------------------------

    async def _fetch_with_retry(self, url: str) -> bytes:
        """
        Fetch the feed URL with exponential backoff retry.

        Wrapped by tenacity so transient network issues are retried
        automatically without any retry logic leaking into fetch().
        """
        # Define the retry-decorated inner function here so we can pass
        # `url` as a regular argument (tenacity doesn't support instance
        # methods with retry cleanly on older versions).
        @retry(
            retry=retry_if_exception_type((httpx.TransportError, FetchError)),
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
            reraise=False,
        )
        async def _do_fetch() -> bytes:
            return await self._get(url)

        return await _do_fetch()

    async def _get(self, url: str) -> bytes:
        """Execute a single HTTP GET for the feed URL."""
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            try:
                response = await client.get(url)
            except httpx.TransportError as e:
                raise FetchError(
                    f"Network error fetching feed: {e}",
                    url=url,
                ) from e

        # Handle HTTP errors
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(
                "RSS feed returned HTTP 429",
                url=url,
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            raise FetchError(
                f"RSS feed returned HTTP {response.status_code}",
                url=url,
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            raise FetchError(
                f"RSS feed returned HTTP {response.status_code} (not retrying)",
                url=url,
                status_code=response.status_code,
            )

        return response.content

    # ------------------------------------------------------------------
    # Parsing layer
    # ------------------------------------------------------------------

    def _parse_feed(self, raw_bytes: bytes, source: SourceConfig) -> list[NewsItem]:
        """
        Parse raw feed bytes into a list of NewsItems using feedparser.

        feedparser is synchronous — wrap in asyncio.to_thread() if you
        ever need to parse massive feeds without blocking the event loop.
        For typical feeds (<1000 entries) this is negligible.
        """
        feed = feedparser.parse(raw_bytes)

        if feed.bozo and not feed.entries:
            # bozo=True means feedparser found errors, but it often still
            # parses successfully. Only raise if we got zero entries.
            raise ParseError(
                f"Feed parse error: {feed.bozo_exception}",
                source_id=source.id,
                url=source.url or "",
                raw_data=str(raw_bytes[:200]),
            )

        items: list[NewsItem] = []
        for entry in feed.entries:
            item = self._entry_to_news_item(entry, source)
            if item is not None:
                items.append(item)

        return items

    def _entry_to_news_item(
        self, entry: Any, source: SourceConfig
    ) -> NewsItem | None:
        """
        Convert a single feedparser entry dict into a NewsItem.

        Returns None if the entry is missing required fields (url / title),
        so callers can filter gracefully without raising exceptions.

        feedparser normalises field names across RSS 2.0 and Atom:
          - entry.link     → canonical URL
          - entry.title    → story headline
          - entry.summary  → description/summary
          - entry.author   → author name
          - entry.published→ publication date string
        """
        # Required fields
        url = entry.get("link") or entry.get("id") or ""
        title = entry.get("title") or ""

        if not url or not title:
            self.log.debug("Skipping entry with missing url or title")
            return None

        # Optional fields
        summary = self._clean_html(entry.get("summary") or "")
        author = entry.get("author") or entry.get("author_detail", {}).get("name") or ""
        published_at = self._parse_date(entry)

        return NewsItem(
            url=url,
            title=title,
            summary=summary or None,
            author=author or None,
            source_id=source.id,
            source_name=source.name,
            source_type="rss",
            published_at=published_at,
            tags=source.tags,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(entry: Any) -> datetime | None:
        """
        Extract and normalise the publication datetime from a feed entry.

        feedparser provides a ``published_parsed`` time.struct_time, but
        it drops timezone info. We prefer ``published`` (raw string) and
        parse it with email.utils for proper tz handling.
        """
        # Try raw published string first (preserves timezone)
        raw = entry.get("published") or entry.get("updated") or ""
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

        # Fall back to feedparser's parsed struct (always UTC-ish)
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pass

        return None

    @staticmethod
    def _clean_html(text: str) -> str:
        """
        Strip HTML tags from a summary string using BeautifulSoup.

        RSS summaries often contain raw HTML (<p>, <a>, <strong>, etc.)
        which we don't want in the AI prompt. We convert to plain text.
        """
        if not text:
            return ""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            # If bs4 fails, just return the raw text (may contain HTML tags)
            return text
