"""
tests/test_scrapers/test_reddit.py
====================================
Unit tests for the Reddit JSON API scraper (src/scrapers/reddit.py).

All tests use mocked HTTP — no real network calls.
We patch _get_json() to return pre-built Reddit listing dicts.
"""

from __future__ import annotations

from datetime import timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.exceptions import ConfigError, FetchError
from src.models import NewsItem, SourceConfig
from src.scrapers.reddit import RedditScraper


# ---------------------------------------------------------------------------
# Sample data (mimics Reddit JSON API structure)
# ---------------------------------------------------------------------------


def make_listing(*posts) -> dict:
    """Build a minimal Reddit listing API response."""
    return {
        "kind": "Listing",
        "data": {
            "children": [
                {"kind": "t3", "data": post}
                for post in posts
            ]
        }
    }


LINK_POST = {
    "id": "abc123",
    "title": "New study finds Rust 40% faster than C++ in web servers",
    "url": "https://techcrunch.com/rust-study",
    "permalink": "/r/programming/comments/abc123/",
    "score": 2847,
    "num_comments": 312,
    "author": "rust_dev",
    "created_utc": 1750000000.0,
    "is_self": False,
    "stickied": False,
    "selftext": "",
}

SELF_POST = {
    "id": "def456",
    "title": "I built a news aggregator in Python — here's what I learned",
    "url": "https://www.reddit.com/r/Python/comments/def456/",
    "permalink": "/r/Python/comments/def456/",
    "score": 934,
    "num_comments": 87,
    "author": "py_builder",
    "created_utc": 1749990000.0,
    "is_self": True,
    "stickied": False,
    "selftext": "I spent 30 days building this project. Here is what I learned about async Python...",
}

STICKIED_POST = {
    "id": "zzz000",
    "title": "Community Rules — Please Read Before Posting",
    "url": "https://www.reddit.com/r/programming/comments/zzz000/",
    "permalink": "/r/programming/comments/zzz000/",
    "score": 1,
    "num_comments": 0,
    "author": "moderator",
    "created_utc": 1740000000.0,
    "is_self": True,
    "stickied": True,
    "selftext": "Please read the rules...",
}

DELETED_SELF_POST = {
    "id": "del789",
    "title": "This post was deleted",
    "url": "https://www.reddit.com/r/Python/comments/del789/",
    "permalink": "/r/Python/comments/del789/",
    "score": 5,
    "num_comments": 3,
    "author": "[deleted]",
    "created_utc": 1749980000.0,
    "is_self": True,
    "stickied": False,
    "selftext": "[deleted]",
}

DUPLICATE_POST_A = {**LINK_POST, "id": "dup001", "title": "Duplicate story A"}
DUPLICATE_POST_B = {**LINK_POST, "id": "dup001b", "title": "Duplicate story B",
                    "url": "https://techcrunch.com/rust-study"}  # same URL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reddit_scraper() -> RedditScraper:
    return RedditScraper()


@pytest.fixture
def reddit_source() -> SourceConfig:
    return SourceConfig(
        id="reddit-prog-test",
        type="reddit",
        name="r/programming",
        subreddit="programming",
        sort="hot",
        limit=25,
        tags=["programming"],
    )


# ---------------------------------------------------------------------------
# Core fetch tests
# ---------------------------------------------------------------------------


class TestRedditScraperFetch:
    @pytest.mark.unit
    async def test_fetch_returns_news_items(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST, SELF_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert len(items) == 2
        assert all(isinstance(i, NewsItem) for i in items)

    @pytest.mark.unit
    async def test_fetch_link_post_uses_external_url(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].url == "https://techcrunch.com/rust-study"

    @pytest.mark.unit
    async def test_fetch_self_post_uses_reddit_url(self, reddit_scraper, reddit_source):
        listing = make_listing(SELF_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert "reddit.com" in items[0].url

    @pytest.mark.unit
    async def test_fetch_self_post_summary_from_selftext(self, reddit_scraper, reddit_source):
        listing = make_listing(SELF_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].summary is not None
        assert "30 days" in (items[0].summary or "")

    @pytest.mark.unit
    async def test_fetch_deleted_selftext_has_no_summary(self, reddit_scraper, reddit_source):
        listing = make_listing(DELETED_SELF_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].summary is None

    @pytest.mark.unit
    async def test_fetch_skips_stickied_posts(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST, STICKIED_POST, SELF_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        titles = [i.title for i in items]
        assert "Community Rules" not in titles
        assert len(items) == 2

    @pytest.mark.unit
    async def test_fetch_score_and_comments(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].score == 2847
        assert items[0].comment_count == 312

    @pytest.mark.unit
    async def test_fetch_deduplicates_same_url(self, reddit_scraper, reddit_source):
        listing = make_listing(DUPLICATE_POST_A, DUPLICATE_POST_B)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        urls = [i.url for i in items]
        assert len(urls) == len(set(urls))

    @pytest.mark.unit
    async def test_fetch_source_type_is_reddit(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].source_type == "reddit"
        assert items[0].source_id == "reddit-prog-test"

    @pytest.mark.unit
    async def test_fetch_published_at_is_utc(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items[0].published_at is not None
        assert items[0].published_at.tzinfo == timezone.utc

    @pytest.mark.unit
    async def test_fetch_comments_url_is_reddit_thread(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert "reddit.com" in (items[0].comments_url or "")

    @pytest.mark.unit
    async def test_fetch_tags_inherited_from_source(self, reddit_scraper, reddit_source):
        listing = make_listing(LINK_POST)
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert "programming" in items[0].tags

    @pytest.mark.unit
    async def test_empty_listing_returns_empty_list(self, reddit_scraper, reddit_source):
        listing = make_listing()
        with patch.object(reddit_scraper, "_get_json", new_callable=AsyncMock,
                          return_value=listing):
            items = await reddit_scraper.fetch(reddit_source)
        assert items == []

    @pytest.mark.unit
    async def test_wrong_source_type_raises(self, reddit_scraper):
        wrong = SourceConfig(id="hn-1", type="hackernews", name="HN")
        with pytest.raises(ConfigError):
            await reddit_scraper.fetch(wrong)

    @pytest.mark.unit
    async def test_fetch_safe_returns_empty_on_error(self, reddit_scraper, reddit_source):
        async def bad_get(url):
            raise FetchError("503", url=url)
        with patch.object(reddit_scraper, "_get_json", side_effect=bad_get):
            items = await reddit_scraper.fetch_safe(reddit_source)
        assert items == []

    @pytest.mark.unit
    async def test_sort_passed_in_url(self, reddit_scraper, reddit_source):
        """The sort type (hot/new/top) must appear in the request URL."""
        captured_urls = []

        async def capture_get(url):
            captured_urls.append(url)
            return make_listing()

        with patch.object(reddit_scraper, "_get_json", side_effect=capture_get):
            await reddit_scraper.fetch(reddit_source)

        assert any("hot" in u for u in captured_urls)
