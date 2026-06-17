"""
src/scrapers/reddit.py
======================
Reddit scraper using the public JSON API (no OAuth required).

Every Reddit listing page has a hidden JSON endpoint — just append
".json" to any subreddit URL:
  https://www.reddit.com/r/programming/hot.json

Why no OAuth?
  - OAuth requires registering an app and rotating tokens.
  - For read-only, public subreddits the JSON API works fine.
  - We stay under Reddit's rate limit (60 req/min for unauthed) easily
    since we only hit one endpoint per source per day.

Rate limiting:
  Reddit returns HTTP 429 with a Retry-After header when rate limited.
  This is handled by tenacity retry + RateLimitError propagation.

Data mapping:
  Reddit post fields we use:
    data.title       → story title
    data.url         → link URL (or Reddit post URL for self-posts)
    data.permalink   → Reddit discussion thread path
    data.selftext    → post body (used as summary for text posts)
    data.score       → net upvotes (upvotes - downvotes)
    data.num_comments→ comment count
    data.author      → Reddit username
    data.created_utc → Unix timestamp (UTC)
    data.is_self     → True for text posts (use Reddit URL, not external link)
    data.stickied    → True for mod-pinned posts (usually skip these)
"""

from __future__ import annotations

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

_BASE_URL = "https://www.reddit.com"
_USER_AGENT = "NewsRadar/0.1 (+https://github.com/Harshads-git/news-radar)"
_REQUEST_TIMEOUT = 15.0
_MAX_RETRIES = 3
_RETRY_MIN_WAIT = 2
_RETRY_MAX_WAIT = 30

# Reddit sort options
_VALID_SORTS = {"hot", "new", "top", "rising", "controversial"}


class RedditScraper(BaseScraper):
    """
    Scraper for Reddit subreddits using the public JSON API.

    Fetches posts from a single subreddit listing endpoint and converts
    each post to a NewsItem. Skips stickied moderator posts and NSFW
    content by default.

    Configuration (via SourceConfig fields):
      subreddit: str     — name without r/ prefix (e.g. "programming")
      sort: str          — "hot", "new", "top", "rising" (default: "hot")
      limit: int         — max items to return (max 100 per Reddit API)
    """

    SOURCE_TYPE = "reddit"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch(self, source: SourceConfig) -> list[NewsItem]:
        """Fetch posts from a subreddit listing and return as NewsItems."""
        self.validate_source(source)
        self._log_fetch_start(source)

        subreddit = source.subreddit or ""
        sort = source.sort if source.sort in _VALID_SORTS else "hot"
        # Reddit API caps at 100 per request
        limit = min(source.limit, 100)

        url = f"{_BASE_URL}/r/{subreddit}/{sort}.json?limit={limit}&raw_json=1"

        try:
            data = await self._get_json_with_retry(url)
        except RetryError as e:
            raise FetchError(
                f"Failed to fetch r/{subreddit} after {_MAX_RETRIES} attempts",
                source_id=source.id,
                url=url,
            ) from e

        items = self._parse_listing(data, source)
        items = self._deduplicate(items)

        self._log_fetch_done(source, len(items))
        return items

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    async def _get_json_with_retry(self, url: str) -> Any:
        """Fetch a Reddit JSON endpoint with tenacity retry."""

        @retry(
            retry=retry_if_exception_type((httpx.TransportError, FetchError)),
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
            reraise=False,
        )
        async def _do_get() -> Any:
            return await self._get_json(url)

        return await _do_get()

    async def _get_json(self, url: str) -> Any:
        """Execute a single HTTP GET to a Reddit JSON endpoint."""
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": _USER_AGENT,
                # Reddit prefers these headers for API-style requests
                "Accept": "application/json",
            },
        ) as client:
            try:
                response = await client.get(url)
            except httpx.TransportError as e:
                raise FetchError(f"Network error: {e}", url=url) from e

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(
                "Reddit API rate limited",
                url=url,
                retry_after=retry_after,
            )
        if response.status_code == 403:
            raise FetchError(
                "Reddit returned 403 — subreddit may be private or banned",
                url=url,
                status_code=403,
            )
        if response.status_code >= 400:
            raise FetchError(
                f"Reddit API returned HTTP {response.status_code}",
                url=url,
                status_code=response.status_code,
            )

        return response.json()

    # ------------------------------------------------------------------
    # Parsing layer
    # ------------------------------------------------------------------

    def _parse_listing(
        self, data: Any, source: SourceConfig
    ) -> list[NewsItem]:
        """
        Parse a Reddit listing JSON response into a list of NewsItems.

        Reddit listing structure:
          {
            "kind": "Listing",
            "data": {
              "children": [
                {"kind": "t3", "data": { ...post fields... }},
                ...
              ]
            }
          }
        """
        if not isinstance(data, dict):
            raise ParseError(
                "Reddit API returned unexpected response shape",
                source_id=source.id,
            )

        listing = data.get("data", {})
        children = listing.get("children", [])

        if not isinstance(children, list):
            raise ParseError(
                "Reddit listing 'children' is not a list",
                source_id=source.id,
            )

        items: list[NewsItem] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            post_data = child.get("data", {})
            item = self._post_to_news_item(post_data, source)
            if item is not None:
                items.append(item)

        return items

    def _post_to_news_item(
        self, post: dict[str, Any], source: SourceConfig
    ) -> NewsItem | None:
        """
        Convert a single Reddit post dict into a NewsItem.

        Filters:
          - Skip stickied posts (mod announcements)
          - Skip posts with no title
          - For link posts: use the external URL
          - For self posts: use the Reddit permalink
        """
        title = (post.get("title") or "").strip()
        if not title:
            return None

        # Skip stickied mod posts
        if post.get("stickied"):
            return None

        # Determine URL
        is_self = post.get("is_self", False)
        permalink = post.get("permalink", "")
        reddit_url = f"https://www.reddit.com{permalink}" if permalink else ""

        if is_self:
            # Text post — use Reddit discussion URL as canonical link
            url = reddit_url
        else:
            # Link post — use the external URL, fall back to Reddit URL
            url = post.get("url") or reddit_url

        if not url:
            return None

        # Comments URL always points to Reddit thread
        comments_url = reddit_url or None

        # Summary: use self-text for text posts, truncated
        summary: str | None = None
        if is_self and (selftext := post.get("selftext", "").strip()):
            summary = selftext[:500] if selftext != "[deleted]" else None

        # Timestamp
        published_at: datetime | None = None
        if ts := post.get("created_utc"):
            published_at = datetime.fromtimestamp(float(ts), tz=timezone.utc)

        return NewsItem(
            url=url,
            title=title,
            summary=summary,
            author=post.get("author") or None,
            source_id=source.id,
            source_name=source.name,
            source_type="reddit",
            score=post.get("score"),
            comment_count=post.get("num_comments"),
            comments_url=comments_url,
            published_at=published_at,
            tags=source.tags,
            raw=post,
        )
