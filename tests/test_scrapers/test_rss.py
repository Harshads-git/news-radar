"""
tests/test_scrapers/test_rss.py
================================
Unit tests for the RSS feed scraper (src/scrapers/rss.py).

All tests use mocked HTTP — no real network calls.
We mock httpx.AsyncClient.get() to return pre-built responses,
so tests run fast and reliably in CI.

Integration tests (real feeds) are kept in:
  tests/test_scrapers/test_rss_integration.py
and are marked @pytest.mark.integration so they're skipped in CI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.exceptions import FetchError, ParseError, RateLimitError
from src.models import NewsItem, SourceConfig
from src.scrapers.rss import RssScraper


# ---------------------------------------------------------------------------
# Sample RSS feed content (minimal valid RSS 2.0 and Atom)
# ---------------------------------------------------------------------------

RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample Tech Feed</title>
    <link>https://example.com</link>
    <description>A sample tech RSS feed for testing</description>
    <item>
      <title>Python 4.0 Released with Major Performance Gains</title>
      <link>https://example.com/python-4-released</link>
      <description>The Python team announces version 4.0 with &lt;b&gt;3x speed improvements&lt;/b&gt; over 3.12.</description>
      <author>Guido van Rossum</author>
      <pubDate>Sun, 15 Jun 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>AI Model Beats Human Doctors at Diagnosis</title>
      <link>https://example.com/ai-medical-diagnosis</link>
      <description>A new large language model achieves 94% accuracy on medical image diagnosis.</description>
      <pubDate>Sat, 14 Jun 2026 08:00:00 +0000</pubDate>
    </item>
    <item>
      <title>  Rust Overtakes C++ in Systems Programming Survey  </title>
      <link>https://example.com/rust-cpp-survey/</link>
      <description>Annual survey shows Rust gaining ground in embedded systems.</description>
      <pubDate>Fri, 13 Jun 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Sample Atom Feed</title>
  <link href="https://atom.example.com" rel="alternate"/>
  <updated>2026-06-15T10:00:00Z</updated>
  <entry>
    <title>Atom Entry One</title>
    <link href="https://atom.example.com/entry-1"/>
    <id>https://atom.example.com/entry-1</id>
    <updated>2026-06-15T09:00:00Z</updated>
    <summary>First test Atom entry.</summary>
    <author><name>Jane Doe</name></author>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <link href="https://atom.example.com/entry-2"/>
    <id>https://atom.example.com/entry-2</id>
    <updated>2026-06-14T09:00:00Z</updated>
    <summary>Second test Atom entry.</summary>
  </entry>
</feed>"""

EMPTY_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://empty.example.com</link>
    <description>Feed with no items</description>
  </channel>
</rss>"""

DUPLICATE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Duplicate Feed</title>
    <link>https://dup.example.com</link>
    <item>
      <title>Same Story</title>
      <link>https://dup.example.com/same-story</link>
      <pubDate>Sun, 15 Jun 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Same Story (repost)</title>
      <link>https://dup.example.com/same-story</link>
      <pubDate>Sun, 15 Jun 2026 11:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Unique Story</title>
      <link>https://dup.example.com/unique-story</link>
      <pubDate>Sat, 14 Jun 2026 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rss_scraper() -> RssScraper:
    return RssScraper()


@pytest.fixture
def rss_source() -> SourceConfig:
    return SourceConfig(
        id="rss-test",
        type="rss",
        name="Test RSS Feed",
        url="https://example.com/feed.xml",
        limit=10,
        tags=["tech", "programming"],
    )


def make_httpx_response(content: bytes, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response with given content and status code."""
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"Content-Type": "application/rss+xml"},
    )


# ---------------------------------------------------------------------------
# RssScraper — Core fetch tests
# ---------------------------------------------------------------------------


class TestRssScraperFetch:
    @pytest.mark.unit
    async def test_fetch_returns_list_of_news_items(self, rss_scraper, rss_source):
        """Fetching a valid RSS feed returns a list of NewsItem objects."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        assert isinstance(items, list)
        assert len(items) == 3
        assert all(isinstance(i, NewsItem) for i in items)

    @pytest.mark.unit
    async def test_fetch_respects_source_limit(self, rss_scraper, rss_source):
        """Fetch must not return more items than source.limit."""
        rss_source.limit = 2
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        assert len(items) <= 2

    @pytest.mark.unit
    async def test_fetch_correct_item_fields(self, rss_scraper, rss_source):
        """NewsItems must have the correct title, url, and source metadata."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        # Find the Python item (order may vary after sort)
        python_item = next(i for i in items if "Python" in i.title)
        assert python_item.url == "https://example.com/python-4-released"
        assert python_item.source_id == "rss-test"
        assert python_item.source_name == "Test RSS Feed"
        assert python_item.source_type == "rss"
        assert "tech" in python_item.tags

    @pytest.mark.unit
    async def test_fetch_title_whitespace_cleaned(self, rss_scraper, rss_source):
        """Leading/trailing whitespace in titles must be stripped."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        rust_item = next(i for i in items if "Rust" in i.title)
        assert not rust_item.title.startswith(" ")
        assert not rust_item.title.endswith(" ")

    @pytest.mark.unit
    async def test_fetch_url_trailing_slash_stripped(self, rss_scraper, rss_source):
        """Trailing slashes in URLs must be removed for consistent dedup."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        rust_item = next(i for i in items if "Rust" in i.title)
        assert not rust_item.url.endswith("/")

    @pytest.mark.unit
    async def test_fetch_html_stripped_from_summary(self, rss_scraper, rss_source):
        """HTML tags in RSS summaries must be stripped to plain text."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        python_item = next(i for i in items if "Python" in i.title)
        assert "<b>" not in (python_item.summary or "")
        assert "3x speed improvements" in (python_item.summary or "")

    @pytest.mark.unit
    async def test_fetch_atom_feed_works(self, rss_scraper):
        """The scraper must handle Atom feeds, not just RSS 2.0."""
        atom_source = SourceConfig(
            id="atom-test",
            type="rss",
            name="Atom Feed",
            url="https://atom.example.com/feed.atom",
            limit=10,
        )
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ATOM_SAMPLE
            items = await rss_scraper.fetch(atom_source)

        assert len(items) == 2
        assert items[0].title == "Atom Entry One"

    @pytest.mark.unit
    async def test_fetch_empty_feed_returns_empty_list(self, rss_scraper, rss_source):
        """An RSS feed with no items returns an empty list (not an error)."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = EMPTY_RSS
            items = await rss_scraper.fetch(rss_source)

        assert items == []

    @pytest.mark.unit
    async def test_fetch_deduplicates_same_url(self, rss_scraper, rss_source):
        """If the same URL appears twice in a feed, only one item is kept."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = DUPLICATE_RSS
            items = await rss_scraper.fetch(rss_source)

        urls = [i.url for i in items]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in result"
        assert len(items) == 2  # 3 entries, 1 deduped → 2

    @pytest.mark.unit
    async def test_fetch_sorted_newest_first(self, rss_scraper, rss_source):
        """Items must be sorted by published_at descending (newest first)."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        dates = [i.published_at for i in items if i.published_at is not None]
        assert dates == sorted(dates, reverse=True)

    @pytest.mark.unit
    async def test_fetch_author_extracted(self, rss_scraper, rss_source):
        """Author field should be populated when the feed provides it."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        python_item = next(i for i in items if "Python" in i.title)
        assert python_item.author == "Guido van Rossum"

    @pytest.mark.unit
    async def test_fetch_published_at_is_utc(self, rss_scraper, rss_source):
        """All parsed datetimes must have UTC timezone."""
        with patch.object(rss_scraper, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = RSS_SAMPLE
            items = await rss_scraper.fetch(rss_source)

        for item in items:
            if item.published_at:
                assert item.published_at.tzinfo is not None
                assert item.published_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# RssScraper — Error handling tests
# ---------------------------------------------------------------------------


class TestRssScraperErrors:
    @pytest.mark.unit
    async def test_wrong_source_type_raises_config_error(self, rss_scraper):
        """Passing a non-RSS source to RssScraper must raise ConfigError."""
        from src.exceptions import ConfigError

        wrong_source = SourceConfig(
            id="hn-1", type="hackernews", name="HN"
        )
        with pytest.raises(ConfigError, match="cannot handle source type"):
            await rss_scraper.fetch(wrong_source)

    @pytest.mark.unit
    async def test_http_429_raises_rate_limit_error(self, rss_scraper, rss_source):
        """HTTP 429 from the feed must raise RateLimitError."""
        async def mock_get(url: str) -> bytes:
            raise RateLimitError("429", source_id="rss-test", retry_after=30)

        with patch.object(rss_scraper, "_get", side_effect=mock_get):
            with pytest.raises((RateLimitError, FetchError)):
                await rss_scraper.fetch(rss_source)

    @pytest.mark.unit
    async def test_network_error_raises_fetch_error(self, rss_scraper, rss_source):
        """A network transport error must eventually raise FetchError."""
        async def mock_get(url: str) -> bytes:
            raise FetchError("Connection refused", source_id="rss-test", url=url)

        with patch.object(rss_scraper, "_get", side_effect=mock_get):
            with pytest.raises(FetchError):
                await rss_scraper.fetch(rss_source)

    @pytest.mark.unit
    def test_source_type_is_rss(self, rss_scraper):
        assert rss_scraper.SOURCE_TYPE == "rss"

    @pytest.mark.unit
    async def test_fetch_safe_returns_empty_on_error(self, rss_scraper, rss_source):
        """fetch_safe() must return [] instead of raising on errors."""
        async def mock_get(url: str) -> bytes:
            raise FetchError("Network down", source_id="rss-test", url=url)

        with patch.object(rss_scraper, "_get", side_effect=mock_get):
            items = await rss_scraper.fetch_safe(rss_source)

        assert items == []


# ---------------------------------------------------------------------------
# RssScraper — Utility method tests
# ---------------------------------------------------------------------------


class TestRssScraperUtils:
    @pytest.mark.unit
    def test_clean_html_removes_tags(self):
        html = "<p>Hello <b>world</b> from <a href='#'>here</a>.</p>"
        result = RssScraper._clean_html(html)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    @pytest.mark.unit
    def test_clean_html_empty_string(self):
        assert RssScraper._clean_html("") == ""

    @pytest.mark.unit
    def test_clean_html_plain_text_unchanged(self):
        text = "No HTML tags here at all."
        assert RssScraper._clean_html(text) == text

    @pytest.mark.unit
    def test_parse_date_valid_rfc2822(self):
        mock_entry = {"published": "Sun, 15 Jun 2026 10:00:00 +0000"}
        result = RssScraper._parse_date(mock_entry)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2026
        assert result.month == 6
        assert result.day == 15

    @pytest.mark.unit
    def test_parse_date_returns_none_for_missing(self):
        result = RssScraper._parse_date({})
        assert result is None

    @pytest.mark.unit
    def test_parse_date_handles_bad_format_gracefully(self):
        mock_entry = {"published": "not-a-date"}
        # Should not raise — returns None
        result = RssScraper._parse_date(mock_entry)
        assert result is None
