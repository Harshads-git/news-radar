"""
src/scrapers/hackernews.py
==========================
Hacker News scraper using the official Firebase REST API.

API Docs: https://github.com/HackerNews/API

Why the HN API instead of the RSS feed?
  - The Firebase API gives us vote counts, comment counts, and story IDs
    which the RSS feed omits. These engagement signals are valuable for
    AI scoring (a story with 500 points is likely more worth reading).
  - The API lets us choose between topstories, newstories, beststories,
    and askstories — the RSS only has "frontpage".

Architecture:
  Step 1: GET /v0/topstories.json → list of up to 500 story IDs
  Step 2: GET /v0/item/{id}.json for each ID (concurrently, up to limit)
  Step 3: Map each item JSON → NewsItem

Concurrency: We fetch individual items concurrently using asyncio.gather()
with a semaphore to cap simultaneous connections at MAX_CONCURRENT requests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.exceptions import FetchError, ParseError, RateLimitError
from src.models import NewsItem, SourceConfig
from src.scrapers.base import BaseScraper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://hacker-news.firebaseio.com/v0"
_USER_AGENT = "NewsRadar/0.1 (+https://github.com/Harshads-git/news-radar)"
_REQUEST_TIMEOUT = 15.0
_MAX_CONCURRENT = 10   # max simultaneous item fetches
_MAX_RETRIES = 3
_RETRY_MIN_WAIT = 1
_RETRY_MAX_WAIT = 15

# HN story types we care about (exclude jobs, polls, comments)
_STORY_TYPES = {"story", "ask", "show"}


class HackerNewsScraper(BaseScraper):
    """
    Scraper for Hacker News using the Firebase REST API.

    Fetches the top N story IDs, then resolves each ID to a full
    story object concurrently (bounded by _MAX_CONCURRENT semaphore).

    Supports these HN story feeds (set via source config name/tags):
      - topstories  (default — the HN frontpage)
      - newstories  (newest submissions)
      - beststories (all-time highest scoring)
      - askstories  (Ask HN posts only)
      - showstories (Show HN posts only)
    """

    SOURCE_TYPE = "hackernews"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch(self, source: SourceConfig) -> list[NewsItem]:
        """Fetch top HN stories and return as NewsItems."""
        self.validate_source(source)
        self._log_fetch_start(source)

        feed_type = self._get_feed_type(source)

        try:
            story_ids = await self._fetch_story_ids(feed_type)
        except RetryError as e:
            raise FetchError(
                f"Failed to fetch HN story list after {_MAX_RETRIES} attempts",
                source_id=source.id,
            ) from e

        # Limit which IDs we resolve (avoid fetching 500 items)
        story_ids = story_ids[: source.limit]

        # Fetch all stories concurrently
        items = await self._fetch_stories_concurrent(story_ids, source)
        items = self._deduplicate(items)

        self._log_fetch_done(source, len(items))
        return items

    # ------------------------------------------------------------------
    # Feed type resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _get_feed_type(source: SourceConfig) -> str:
        """
        Determine which HN feed to use based on the source config.

        Checks source.tags for known feed type keywords.
        Falls back to 'topstories' (frontpage) if not specified.
        """
        tag_to_feed = {
            "new": "newstories",
            "newest": "newstories",
            "best": "beststories",
            "ask": "askstories",
            "show": "showstories",
            "top": "topstories",
        }
        for tag in (source.tags or []):
            if tag.lower() in tag_to_feed:
                return tag_to_feed[tag.lower()]
        return "topstories"

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    async def _fetch_story_ids(self, feed_type: str) -> list[int]:
        """Fetch the ordered list of story IDs for the given feed type."""
        url = f"{_BASE_URL}/{feed_type}.json"
        data = await self._get_json(url)
        if not isinstance(data, list):
            raise ParseError(
                f"HN API returned unexpected type for story IDs: {type(data)}",
                source_id="hackernews",
                url=url,
            )
        return [int(i) for i in data if isinstance(i, int)]

    async def _fetch_story_item(self, story_id: int) -> dict[str, Any] | None:
        """Fetch a single HN item by ID. Returns None if not found or deleted."""
        url = f"{_BASE_URL}/item/{story_id}.json"
        try:
            data = await self._get_json(url)
        except FetchError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    async def _fetch_stories_concurrent(
        self, story_ids: list[int], source: SourceConfig
    ) -> list[NewsItem]:
        """Fetch multiple story items concurrently using a semaphore."""
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        async def fetch_one(story_id: int) -> NewsItem | None:
            async with semaphore:
                data = await self._fetch_story_item(story_id)
                if data is None:
                    return None
                return self._item_to_news_item(data, source)

        results = await asyncio.gather(*[fetch_one(sid) for sid in story_ids])
        return [item for item in results if item is not None]

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
        reraise=True,
    )
    async def _get_json(self, url: str) -> Any:
        """GET a URL and parse response as JSON, with retry on transport errors."""
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            try:
                response = await client.get(url)
            except httpx.TransportError as e:
                raise FetchError(f"Network error: {e}", url=url) from e

        if response.status_code == 429:
            raise RateLimitError("HN API rate limited", url=url, retry_after=60)
        if response.status_code >= 400:
            raise FetchError(
                f"HN API returned HTTP {response.status_code}",
                url=url,
                status_code=response.status_code,
            )

        return response.json()

    # ------------------------------------------------------------------
    # Data mapping
    # ------------------------------------------------------------------

    def _item_to_news_item(
        self, data: dict[str, Any], source: SourceConfig
    ) -> NewsItem | None:
        """
        Convert a raw HN Firebase item dict into a NewsItem.

        HN item fields:
          id        → unique integer ID
          type      → "story", "comment", "ask", "job", "poll"
          title     → story headline
          url       → external URL (absent for Ask HN — use HN thread URL)
          score     → upvote count
          by        → author username
          time      → Unix timestamp
          descendants → total comment count
          text      → body text for Ask HN / Show HN posts
        """
        # Skip non-story types and deleted/dead items
        item_type = data.get("type", "")
        if item_type not in _STORY_TYPES:
            return None
        if data.get("deleted") or data.get("dead"):
            return None

        title = data.get("title") or ""
        story_id = data.get("id")

        if not title or not story_id:
            return None

        # External URL (may be absent for Ask HN — use the discussion page)
        hn_url = f"https://news.ycombinator.com/item?id={story_id}"
        url = data.get("url") or hn_url

        # Comments URL always points to the HN thread
        comments_url = hn_url

        # Convert Unix timestamp → UTC datetime
        published_at: datetime | None = None
        if ts := data.get("time"):
            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc)

        # Body text for Ask/Show HN (used as summary)
        summary: str | None = None
        if text := data.get("text"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            summary = soup.get_text(separator=" ", strip=True)[:500] or None

        return NewsItem(
            url=url,
            title=title,
            summary=summary,
            author=data.get("by") or None,
            source_id=source.id,
            source_name=source.name,
            source_type="hackernews",
            score=data.get("score"),
            comment_count=data.get("descendants"),
            comments_url=comments_url,
            published_at=published_at,
            tags=source.tags,
            raw=data,
        )
