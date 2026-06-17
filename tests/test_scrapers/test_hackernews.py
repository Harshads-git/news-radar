"""
tests/test_scrapers/test_hackernews.py
======================================
Unit tests for the Hacker News API scraper (src/scrapers/hackernews.py).

All tests use mocked HTTP — no real network calls.
We patch _get_json() to return pre-built response dicts.
"""

from __future__ import annotations

from datetime import timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.exceptions import ConfigError, FetchError
from src.models import NewsItem, SourceConfig
from src.scrapers.hackernews import HackerNewsScraper


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

STORY_IDS = [1001, 1002, 1003, 1004, 1005]

STORY_1001 = {
    "id": 1001,
    "type": "story",
    "title": "Python 4.0 Performance Benchmarks Are Stunning",
    "url": "https://example.com/python-4-benchmarks",
    "score": 842,
    "by": "guido",
    "time": 1750000000,
    "descendants": 234,
}

STORY_1002 = {
    "id": 1002,
    "type": "story",
    "title": "Show HN: I built a CLI tool in Rust",
    "url": "https://github.com/user/cli-tool",
    "score": 312,
    "by": "rustacean",
    "time": 1749990000,
    "descendants": 87,
}

STORY_1003 = {
    "id": 1003,
    "type": "ask",
    "title": "Ask HN: What tools do you use for daily news?",
    "score": 156,
    "by": "curious_dev",
    "time": 1749980000,
    "descendants": 203,
    "text": "<p>I want to build a personal news radar. Any suggestions?</p>",
}

STORY_1004_DEAD = {
    "id": 1004,
    "type": "story",
    "title": "This story was flagged",
    "url": "https://example.com/flagged",
    "score": 1,
    "by": "spammer",
    "time": 1749970000,
    "dead": True,
}

STORY_1005_JOB = {
    "id": 1005,
    "type": "job",
    "title": "We are hiring Python engineers",
    "url": "https://company.com/jobs",
    "score": 0,
    "by": "company_hr",
    "time": 1749960000,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hn_scraper() -> HackerNewsScraper:
    return HackerNewsScraper()


@pytest.fixture
def hn_source() -> SourceConfig:
    return SourceConfig(
        id="hn-api-test",
        type="hackernews",
        name="HN Top Stories",
        limit=5,
        tags=["tech"],
    )


def make_json_side_effect(story_ids: list, stories: dict):
    """
    Build an async side_effect for _get_json that returns:
      - story_ids list for the /topstories.json URL
      - individual story dicts for /item/{id}.json URLs
    """
    async def _side_effect(url: str):
        if "topstories.json" in url or "stories.json" in url:
            return story_ids
        for story_id, data in stories.items():
            if f"/item/{story_id}.json" in url:
                return data
        return None
    return _side_effect


# ---------------------------------------------------------------------------
# Core fetch tests
# ---------------------------------------------------------------------------


class TestHNScraperFetch:
    @pytest.mark.unit
    async def test_fetch_returns_news_items(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001, 1002: STORY_1002, 1003: STORY_1003,
                   1004: STORY_1004_DEAD, 1005: STORY_1005_JOB}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect(STORY_IDS, stories)):
            items = await hn_scraper.fetch(hn_source)

        assert isinstance(items, list)
        # Dead story (1004) and job post (1005) must be filtered
        assert all(isinstance(i, NewsItem) for i in items)
        urls = [i.url for i in items]
        assert not any("flagged" in u for u in urls), "Dead story must be excluded"

    @pytest.mark.unit
    async def test_fetch_excludes_dead_stories(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001, 1004: STORY_1004_DEAD}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1001, 1004], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert all(not i.url.endswith("flagged") for i in items)
        assert len(items) == 1

    @pytest.mark.unit
    async def test_fetch_excludes_job_posts(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001, 1005: STORY_1005_JOB}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1001, 1005], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert len(items) == 1
        assert items[0].title == STORY_1001["title"]

    @pytest.mark.unit
    async def test_fetch_score_and_comment_count(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1001], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert items[0].score == 842
        assert items[0].comment_count == 234

    @pytest.mark.unit
    async def test_fetch_comments_url_is_hn_thread(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1001], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert "news.ycombinator.com/item?id=1001" in (items[0].comments_url or "")

    @pytest.mark.unit
    async def test_fetch_ask_hn_has_summary(self, hn_scraper, hn_source):
        """Ask HN text body must be stripped of HTML and used as summary."""
        stories = {1003: STORY_1003}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1003], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert len(items) == 1
        assert items[0].summary is not None
        assert "<p>" not in (items[0].summary or "")

    @pytest.mark.unit
    async def test_fetch_ask_hn_url_is_hn_thread(self, hn_scraper, hn_source):
        """Ask HN posts have no external URL — canonical URL must be HN thread."""
        stories = {1003: STORY_1003}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1003], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert "news.ycombinator.com" in items[0].url

    @pytest.mark.unit
    async def test_fetch_published_at_is_utc(self, hn_scraper, hn_source):
        stories = {1001: STORY_1001}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect([1001], stories)):
            items = await hn_scraper.fetch(hn_source)
        assert items[0].published_at is not None
        assert items[0].published_at.tzinfo == timezone.utc

    @pytest.mark.unit
    async def test_fetch_respects_limit(self, hn_scraper, hn_source):
        hn_source.limit = 2
        all_ids = [1001, 1002, 1003]
        stories = {1001: STORY_1001, 1002: STORY_1002, 1003: STORY_1003}
        with patch.object(hn_scraper, "_get_json",
                          side_effect=make_json_side_effect(all_ids, stories)):
            items = await hn_scraper.fetch(hn_source)
        assert len(items) <= 2

    @pytest.mark.unit
    async def test_wrong_source_type_raises(self, hn_scraper):
        from src.exceptions import ConfigError
        wrong = SourceConfig(id="rss-1", type="rss", name="RSS", url="https://ex.com")
        with pytest.raises(ConfigError):
            await hn_scraper.fetch(wrong)

    @pytest.mark.unit
    async def test_fetch_safe_returns_empty_on_error(self, hn_scraper, hn_source):
        async def bad_get(url):
            raise FetchError("Down", source_id="hn")
        with patch.object(hn_scraper, "_get_json", side_effect=bad_get):
            items = await hn_scraper.fetch_safe(hn_source)
        assert items == []


class TestHNScraperFeedType:
    @pytest.mark.unit
    def test_default_feed_type_is_topstories(self):
        source = SourceConfig(id="hn", type="hackernews", name="HN", tags=[])
        assert HackerNewsScraper._get_feed_type(source) == "topstories"

    @pytest.mark.unit
    def test_new_tag_maps_to_newstories(self):
        source = SourceConfig(id="hn", type="hackernews", name="HN", tags=["new"])
        assert HackerNewsScraper._get_feed_type(source) == "newstories"

    @pytest.mark.unit
    def test_best_tag_maps_to_beststories(self):
        source = SourceConfig(id="hn", type="hackernews", name="HN", tags=["best"])
        assert HackerNewsScraper._get_feed_type(source) == "beststories"

    @pytest.mark.unit
    def test_ask_tag_maps_to_askstories(self):
        source = SourceConfig(id="hn", type="hackernews", name="HN", tags=["ask"])
        assert HackerNewsScraper._get_feed_type(source) == "askstories"
