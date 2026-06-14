"""
tests/test_models.py
====================
Unit tests for all Pydantic data models in src/models.py.

Tests verify:
  - Valid construction of all model types
  - Field validators (URL cleaning, title whitespace)
  - Cross-field model validators (required fields per source type)
  - Invalid input → ValidationError behaviour
  - Briefing auto-sort by score
  - SourcesConfig.enabled_sources filtering
  - Computed properties (has_email, briefings_dir, etc.)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
import pytest

from src.models import (
    Briefing,
    NewsItem,
    ScoredItem,
    SourceConfig,
    SourcesConfig,
    SummarizedItem,
)

# ===========================================================================
# SourceConfig Tests
# ===========================================================================


class TestSourceConfig:
    def test_valid_rss_source(self):
        source = SourceConfig(
            id="rss-1",
            type="rss",
            name="HN RSS",
            url="https://hnrss.org/frontpage",
        )
        assert source.id == "rss-1"
        assert source.type == "rss"
        assert source.enabled is True  # default

    def test_valid_hackernews_source(self):
        source = SourceConfig(id="hn-1", type="hackernews", name="HN API")
        assert source.type == "hackernews"
        assert source.limit == 30  # default

    def test_valid_reddit_source(self):
        source = SourceConfig(
            id="reddit-1",
            type="reddit",
            name="r/Python",
            subreddit="Python",
            sort="hot",
        )
        assert source.subreddit == "Python"

    def test_rss_source_missing_url_raises(self):
        """RSS source without a URL must raise ValidationError."""
        with pytest.raises(ValidationError, match="requires a 'url' field"):
            SourceConfig(id="rss-bad", type="rss", name="Bad RSS")

    def test_reddit_source_missing_subreddit_raises(self):
        """Reddit source without a subreddit must raise ValidationError."""
        with pytest.raises(ValidationError, match="requires a 'subreddit' field"):
            SourceConfig(id="r-bad", type="reddit", name="Bad Reddit")

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValidationError, match="not supported"):
            SourceConfig(id="bad", type="twitter", name="Twitter")

    def test_disabled_source(self):
        source = SourceConfig(
            id="rss-off",
            type="rss",
            name="Disabled Feed",
            url="https://example.com/feed",
            enabled=False,
        )
        assert source.enabled is False

    def test_limit_clamped_to_max(self):
        """Limit above 200 should raise ValidationError."""
        with pytest.raises(ValidationError):
            SourceConfig(id="x", type="hackernews", name="X", limit=999)

    def test_tags_default_empty(self):
        source = SourceConfig(id="hn", type="hackernews", name="HN")
        assert source.tags == []


class TestSourcesConfig:
    def test_enabled_sources_filter(self):
        config = SourcesConfig(
            sources=[
                SourceConfig(id="a", type="hackernews", name="HN", enabled=True),
                SourceConfig(id="b", type="rss", name="RSS", url="https://ex.com", enabled=False),
                SourceConfig(id="c", type="hackernews", name="HN2", enabled=True),
            ]
        )
        enabled = config.enabled_sources
        assert len(enabled) == 2
        assert all(s.enabled for s in enabled)

    def test_empty_sources(self):
        config = SourcesConfig()
        assert config.enabled_sources == []


# ===========================================================================
# NewsItem Tests
# ===========================================================================


class TestNewsItem:
    def test_minimal_valid_item(self):
        item = NewsItem(
            url="https://example.com/article",
            title="Test Article",
            source_id="src-1",
            source_name="Test Source",
            source_type="rss",
        )
        assert item.title == "Test Article"
        assert item.url == "https://example.com/article"
        assert item.id is not None  # auto-generated UUID
        assert isinstance(item.fetched_at, datetime)

    def test_url_trailing_slash_stripped(self):
        item = NewsItem(
            url="https://example.com/article/",
            title="Title",
            source_id="s",
            source_name="S",
            source_type="rss",
        )
        assert not item.url.endswith("/")

    def test_url_whitespace_stripped(self):
        item = NewsItem(
            url="  https://example.com/article  ",
            title="Title",
            source_id="s",
            source_name="S",
            source_type="rss",
        )
        assert item.url == "https://example.com/article"

    def test_title_excess_whitespace_collapsed(self):
        item = NewsItem(
            url="https://example.com",
            title="  Hello   World  \n",
            source_id="s",
            source_name="S",
            source_type="rss",
        )
        assert item.title == "Hello World"

    def test_equality_based_on_url(self):
        kwargs = {"source_id": "s", "source_name": "S", "source_type": "rss", "title": "T"}
        a = NewsItem(url="https://example.com/a", **kwargs)
        b = NewsItem(url="https://example.com/a", **kwargs)
        c = NewsItem(url="https://example.com/b", **kwargs)
        assert a == b
        assert a != c

    def test_hash_based_on_url(self):
        kwargs = {"source_id": "s", "source_name": "S", "source_type": "rss", "title": "T"}
        a = NewsItem(url="https://example.com/a", **kwargs)
        b = NewsItem(url="https://example.com/a", **kwargs)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1  # deduplication works in a set

    def test_optional_fields_default_none(self):
        item = NewsItem(
            url="https://example.com",
            title="T",
            source_id="s",
            source_name="S",
            source_type="rss",
        )
        assert item.summary is None
        assert item.author is None
        assert item.score is None
        assert item.comment_count is None
        assert item.published_at is None

    def test_raw_field_excluded_from_serialization(self):
        item = NewsItem(
            url="https://example.com",
            title="T",
            source_id="s",
            source_name="S",
            source_type="rss",
            raw={"secret": "data"},
        )
        serialized = item.model_dump()
        assert "raw" not in serialized

    def test_tags_inherited_from_source(self):
        item = NewsItem(
            url="https://example.com",
            title="T",
            source_id="s",
            source_name="S",
            source_type="rss",
            tags=["tech", "AI"],
        )
        assert "tech" in item.tags


# ===========================================================================
# ScoredItem Tests
# ===========================================================================


class TestScoredItem:
    @pytest.fixture
    def base_item(self) -> NewsItem:
        return NewsItem(
            url="https://example.com/story",
            title="AI Takes Over the World",
            source_id="hn",
            source_name="Hacker News",
            source_type="hackernews",
        )

    def test_valid_scored_item(self, base_item: NewsItem):
        scored = ScoredItem(
            item=base_item,
            ai_score=9,
            ai_score_reason="Highly relevant to AI/ML audience.",
            model_used="gpt-4o-mini",
        )
        assert scored.ai_score == 9
        assert scored.title == "AI Takes Over the World"
        assert scored.url == "https://example.com/story"
        assert scored.source_name == "Hacker News"

    def test_score_below_zero_raises(self, base_item: NewsItem):
        with pytest.raises(ValidationError):
            ScoredItem(item=base_item, ai_score=-1)

    def test_score_above_ten_raises(self, base_item: NewsItem):
        with pytest.raises(ValidationError):
            ScoredItem(item=base_item, ai_score=11)

    def test_score_boundary_values(self, base_item: NewsItem):
        """Scores 0 and 10 must be valid."""
        low = ScoredItem(item=base_item, ai_score=0)
        high = ScoredItem(item=base_item, ai_score=10)
        assert low.ai_score == 0
        assert high.ai_score == 10

    def test_from_cache_default_false(self, base_item: NewsItem):
        scored = ScoredItem(item=base_item, ai_score=7)
        assert scored.from_cache is False

    def test_scored_at_is_utc(self, base_item: NewsItem):
        scored = ScoredItem(item=base_item, ai_score=5)
        assert scored.scored_at.tzinfo is not None


# ===========================================================================
# SummarizedItem Tests
# ===========================================================================


class TestSummarizedItem:
    @pytest.fixture
    def scored_item(self) -> ScoredItem:
        item = NewsItem(
            url="https://example.com/story",
            title="Python 4.0 Released",
            source_id="rss-1",
            source_name="Tech News",
            source_type="rss",
        )
        return ScoredItem(item=item, ai_score=8, ai_score_reason="Big Python news")

    def test_valid_summarized_item(self, scored_item: ScoredItem):
        summ = SummarizedItem(
            scored=scored_item,
            ai_summary="Python 4.0 introduces major performance improvements.",
            web_context="Python is the most popular programming language for AI.",
        )
        assert summ.title == "Python 4.0 Released"
        assert summ.ai_score == 8
        assert summ.url == "https://example.com/story"

    def test_empty_summary_allowed(self, scored_item: ScoredItem):
        """Summary can be empty (e.g. AI failed, we store the item anyway)."""
        summ = SummarizedItem(scored=scored_item)
        assert summ.ai_summary == ""


# ===========================================================================
# Briefing Tests
# ===========================================================================


class TestBriefing:
    @staticmethod
    def make_item(url: str, score: int) -> SummarizedItem:
        item = NewsItem(
            url=url,
            title=f"Story at {url}",
            source_id="test",
            source_name="Test",
            source_type="rss",
        )
        scored = ScoredItem(item=item, ai_score=score)
        return SummarizedItem(scored=scored, ai_summary=f"Summary for score {score}")

    def test_briefing_items_sorted_by_score(self):
        """Briefing must auto-sort items highest score first."""
        briefing = Briefing(
            date="2026-06-13",
            items=[
                self.make_item("https://a.com", score=3),
                self.make_item("https://b.com", score=9),
                self.make_item("https://c.com", score=6),
            ],
        )
        scores = [item.ai_score for item in briefing.items]
        assert scores == sorted(scores, reverse=True), "Items must be sorted highest first"

    def test_briefing_top_items_limit(self):
        """top_items should return at most 5 items."""
        briefing = Briefing(
            date="2026-06-13",
            items=[self.make_item(f"https://example.com/{i}", score=i) for i in range(10)],
        )
        assert len(briefing.top_items) == 5

    def test_briefing_item_count(self):
        briefing = Briefing(
            date="2026-06-13",
            items=[self.make_item(f"https://example.com/{i}", score=i) for i in range(7)],
        )
        assert briefing.item_count == 7

    def test_empty_briefing(self):
        briefing = Briefing(date="2026-06-13")
        assert briefing.item_count == 0
        assert briefing.top_items == []

    def test_briefing_generated_at_is_set(self):
        briefing = Briefing(date="2026-06-13")
        assert isinstance(briefing.generated_at, datetime)

    def test_briefing_metadata_defaults(self):
        briefing = Briefing(date="2026-06-13")
        assert briefing.metadata.total_fetched == 0
        assert briefing.metadata.sources_used == []


# ===========================================================================
# Config Integration Tests
# ===========================================================================


class TestSettings:
    def test_settings_load_with_defaults(self):
        """Settings should load without any .env file using all defaults."""
        from src.config import Settings

        s = Settings()
        assert s.ai_model == "gpt-4o-mini"
        assert s.score_threshold == 6
        assert s.output_language == "English"
        assert s.log_level == "INFO"

    def test_settings_paths(self):
        from pathlib import Path

        from src.config import Settings

        s = Settings()
        assert isinstance(s.briefings_dir, Path)
        assert isinstance(s.cache_dir, Path)
        assert str(s.briefings_dir).endswith("briefings")
        assert str(s.cache_dir).endswith("cache")

    def test_settings_has_flags_default_false(self):
        from src.config import Settings

        s = Settings()
        # No API keys in default environment
        assert s.has_openai is False
        assert s.has_gemini is False
        assert s.has_email is False
        assert s.has_discord is False
