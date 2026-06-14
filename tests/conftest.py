"""
tests/conftest.py
=================
Shared pytest fixtures available to all test modules.

Fixtures here are automatically discovered by pytest — no imports needed.
They cover:
  - Common NewsItem / ScoredItem / SummarizedItem factory functions
  - Sample SourceConfig objects
  - Temporary directory fixtures for storage tests
  - Async event loop configuration

Naming convention:
  - `sample_*`  → minimal valid objects for simple tests
  - `full_*`    → objects with all optional fields populated
  - `make_*`    → factory fixtures that return a callable
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import (
    Briefing,
    BriefingMetadata,
    NewsItem,
    ScoredItem,
    SourceConfig,
    SourcesConfig,
    SummarizedItem,
)


# ---------------------------------------------------------------------------
# SourceConfig Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rss_source() -> SourceConfig:
    """Minimal valid RSS source config."""
    return SourceConfig(
        id="rss-test",
        type="rss",
        name="Test RSS Feed",
        url="https://example.com/feed.xml",
        enabled=True,
        limit=10,
        tags=["tech"],
    )


@pytest.fixture
def hn_source() -> SourceConfig:
    """Minimal valid Hacker News source config."""
    return SourceConfig(
        id="hn-test",
        type="hackernews",
        name="Hacker News Test",
        limit=10,
    )


@pytest.fixture
def reddit_source() -> SourceConfig:
    """Minimal valid Reddit source config."""
    return SourceConfig(
        id="reddit-test",
        type="reddit",
        name="r/Python Test",
        subreddit="Python",
        sort="hot",
        limit=10,
    )


@pytest.fixture
def sources_config(rss_source: SourceConfig, hn_source: SourceConfig) -> SourcesConfig:
    """SourcesConfig with two enabled sources."""
    return SourcesConfig(sources=[rss_source, hn_source])


# ---------------------------------------------------------------------------
# NewsItem Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_news_item() -> NewsItem:
    """Minimal valid NewsItem — used as a baseline in most tests."""
    return NewsItem(
        url="https://example.com/sample-article",
        title="Sample Article About AI",
        source_id="rss-test",
        source_name="Test RSS Feed",
        source_type="rss",
    )


@pytest.fixture
def full_news_item() -> NewsItem:
    """NewsItem with all optional fields populated."""
    return NewsItem(
        url="https://news.ycombinator.com/item?id=12345",
        title="Show HN: I built an AI-powered news radar in 30 days",
        summary="A personal news aggregation tool built with Python and Pydantic.",
        author="harshads-git",
        source_id="hn-api",
        source_name="Hacker News",
        source_type="hackernews",
        score=342,
        comment_count=87,
        comments_url="https://news.ycombinator.com/item?id=12345",
        published_at=datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc),
        tags=["tech", "programming", "AI"],
    )


@pytest.fixture
def make_news_item():
    """
    Factory fixture: returns a callable that creates NewsItems with overrides.

    Usage in tests:
        def test_something(make_news_item):
            item = make_news_item(url="https://custom.com", score=100)
    """

    def _factory(**overrides) -> NewsItem:
        defaults = {
            "url": "https://example.com/default-article",
            "title": "Default Test Article",
            "source_id": "test-source",
            "source_name": "Test Source",
            "source_type": "rss",
        }
        defaults.update(overrides)
        return NewsItem(**defaults)

    return _factory


# ---------------------------------------------------------------------------
# ScoredItem Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scored_item(sample_news_item: NewsItem) -> ScoredItem:
    """ScoredItem with score=7 (above default threshold of 6)."""
    return ScoredItem(
        item=sample_news_item,
        ai_score=7,
        ai_score_reason="Relevant to tech audience with practical applications.",
        model_used="gpt-4o-mini",
    )


@pytest.fixture
def make_scored_item(make_news_item):
    """Factory fixture for ScoredItems with score and url overrides."""

    def _factory(score: int = 7, url: str = "https://example.com/story", **item_overrides):
        item = make_news_item(url=url, **item_overrides)
        return ScoredItem(
            item=item,
            ai_score=score,
            ai_score_reason=f"Score {score}/10 assigned.",
            model_used="gpt-4o-mini",
        )

    return _factory


# ---------------------------------------------------------------------------
# SummarizedItem Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_summarized_item(sample_scored_item: ScoredItem) -> SummarizedItem:
    """SummarizedItem with AI summary and web context populated."""
    return SummarizedItem(
        scored=sample_scored_item,
        ai_summary=(
            "This article explores recent advances in AI-powered news curation. "
            "The author demonstrates how LLMs can score and summarize news items "
            "to create a personalized daily briefing pipeline."
        ),
        web_context=(
            "Large Language Models (LLMs) are AI systems trained on vast text corpora. "
            "They can perform tasks like summarization, classification, and scoring."
        ),
        model_used="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Briefing Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_briefing(make_scored_item) -> Briefing:
    """A Briefing with 5 items at varying scores."""
    items = []
    for i, score in enumerate([9, 7, 5, 8, 6]):
        scored = make_scored_item(score=score, url=f"https://example.com/story-{i}")
        items.append(
            SummarizedItem(
                scored=scored,
                ai_summary=f"Summary for story {i} with score {score}.",
            )
        )
    return Briefing(
        date="2026-06-14",
        language="English",
        items=items,
        metadata=BriefingMetadata(
            total_fetched=50,
            total_after_dedup=40,
            total_scored=40,
            total_in_briefing=5,
            sources_used=["rss-test", "hn-test"],
            run_duration_seconds=12.5,
        ),
    )


# ---------------------------------------------------------------------------
# Path / Filesystem Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """
    Temporary data directory with the expected subdirectory layout.
    Used by storage-layer tests to avoid touching the real data/ folder.
    """
    briefings = tmp_path / "briefings"
    cache = tmp_path / "cache"
    briefings.mkdir()
    cache.mkdir()
    return tmp_path
