"""
tests/test_summarizer_briefing.py
==================================
Unit tests for:
  - NewsSummarizer (parse, fallback, full flow)
  - BriefingBuilder (_extract_top_topics, exec summary, build)
  - BriefingStore (save, load, delete, list, range, cleanup)

All AI calls use MockProvider — zero real API calls.
Storage tests use tmp_path (pytest's built-in temp dir fixture).
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.ai.base import BaseAIProvider
from src.exceptions import StorageError
from src.models import (
    Briefing,
    NewsItem,
    ScoredItem,
    SummarizedItem,
)
from src.storage import BriefingStore


# ---------------------------------------------------------------------------
# MockProvider for AI calls
# ---------------------------------------------------------------------------


class MockProvider(BaseAIProvider):
    PROVIDER_NAME = "mock"

    def __init__(self, response: str = ""):
        super().__init__("mock-model")
        self._response = response

    async def complete(self, prompt, *, max_tokens=512, temperature=0.3, system=None) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_news_item(
    url: str = "https://example.com/article",
    title: str = "OpenAI Releases GPT-5 Reasoning Model",
    score: int = 500,
) -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        summary="OpenAI announces its most capable model.",
        author="test_author",
        source_id="hn-test",
        source_name="Hacker News",
        source_type="hackernews",
        score=score,
        comment_count=200,
        published_at=datetime(2026, 6, 20, 10, 0, 0, tzinfo=timezone.utc),
    )


def make_scored_item(
    url: str = "https://example.com/article",
    title: str = "OpenAI Releases GPT-5 Reasoning Model",
    ai_score: int = 8,
    topics: list[str] | None = None,
) -> ScoredItem:
    item = make_news_item(url=url, title=title)
    return ScoredItem(
        item=item,
        ai_score=ai_score,
        ai_score_reason="Very relevant to AI topics",
        ai_reason="Very relevant to AI topics",
        ai_topics=topics or ["AI", "LLM"],
        model_used="mock-model",
    )


def make_summarized_item(
    url: str = "https://example.com/article",
    headline: str = "GPT-5 Sets New AI Benchmarks",
    topics: list[str] | None = None,
) -> SummarizedItem:
    scored = make_scored_item(url=url, topics=topics)
    return SummarizedItem(
        scored=scored,
        ai_headline=headline,
        ai_summary="Para 1: GPT-5 launched.\n\nPara 2: It changes AI.\n\nPara 3: Watch for responses.",
        key_points=["Point 1", "Point 2", "Point 3"],
        model_used="mock-model",
    )


def make_briefing(date_str: str = "2026-06-20", items: int = 3) -> Briefing:
    summarized = [
        make_summarized_item(url=f"https://example.com/{i}", headline=f"Headline {i}")
        for i in range(items)
    ]
    return Briefing(
        date=date_str,
        items=summarized,
        executive_summary="Today in tech: major AI announcements.",
        top_topics=["AI", "Python", "open source"],
        total_fetched=50,
        total_scored=10,
        generated_at=datetime(2026, 6, 20, 15, 0, 0, tzinfo=timezone.utc),
    )


def make_mock_settings(interests: str = "AI, Python, open source"):
    from unittest.mock import MagicMock
    s = MagicMock()
    s.user_interests = interests
    return s


# ===========================================================================
# NewsSummarizer Tests
# ===========================================================================


class TestNewsSummarizerParseResponse:
    @pytest.mark.unit
    def test_clean_json_parsed_correctly(self):
        from src.ai.summarizer import NewsSummarizer
        raw = json.dumps({
            "headline": "GPT-5 Launches with Record Performance",
            "summary": "Para 1.\n\nPara 2.\n\nPara 3.",
            "key_points": ["Point 1", "Point 2", "Point 3"],
        })
        headline, summary, key_points = NewsSummarizer._parse_response(raw)
        assert headline == "GPT-5 Launches with Record Performance"
        assert "Para 1" in summary
        assert len(key_points) == 3

    @pytest.mark.unit
    def test_markdown_fences_stripped(self):
        from src.ai.summarizer import NewsSummarizer
        raw = '```json\n{"headline": "Test", "summary": "Body", "key_points": []}\n```'
        headline, summary, key_points = NewsSummarizer._parse_response(raw)
        assert headline == "Test"

    @pytest.mark.unit
    def test_missing_headline_raises(self):
        from src.ai.summarizer import NewsSummarizer
        raw = '{"headline": "", "summary": "Body", "key_points": []}'
        with pytest.raises(ValueError, match="headline"):
            NewsSummarizer._parse_response(raw)

    @pytest.mark.unit
    def test_missing_summary_raises(self):
        from src.ai.summarizer import NewsSummarizer
        raw = '{"headline": "Title", "summary": "", "key_points": []}'
        with pytest.raises(ValueError, match="summary"):
            NewsSummarizer._parse_response(raw)

    @pytest.mark.unit
    def test_no_json_raises_value_error(self):
        from src.ai.summarizer import NewsSummarizer
        with pytest.raises(ValueError, match="No JSON"):
            NewsSummarizer._parse_response("Not JSON at all")

    @pytest.mark.unit
    def test_empty_key_points_allowed(self):
        from src.ai.summarizer import NewsSummarizer
        raw = '{"headline": "H", "summary": "S", "key_points": []}'
        headline, summary, key_points = NewsSummarizer._parse_response(raw)
        assert key_points == []


class TestNewsSummarizerFallback:
    @pytest.mark.unit
    def test_fallback_uses_original_title(self):
        from src.ai.summarizer import NewsSummarizer
        scored = make_scored_item(title="My Original Title")
        headline, summary, key_points = NewsSummarizer._make_fallback(scored)
        assert headline == "My Original Title"

    @pytest.mark.unit
    def test_fallback_uses_original_summary(self):
        from src.ai.summarizer import NewsSummarizer
        scored = make_scored_item()
        headline, summary, key_points = NewsSummarizer._make_fallback(scored)
        assert summary  # must not be empty

    @pytest.mark.unit
    def test_fallback_returns_key_points_list(self):
        from src.ai.summarizer import NewsSummarizer
        scored = make_scored_item()
        _, _, key_points = NewsSummarizer._make_fallback(scored)
        assert isinstance(key_points, list)
        assert len(key_points) > 0


class TestNewsSummarizerSummarizeAll:
    @pytest.mark.unit
    async def test_summarize_all_returns_summarized_items(self):
        from src.ai.summarizer import NewsSummarizer
        mock_resp = json.dumps({
            "headline": "AI Changes Everything",
            "summary": "P1.\n\nP2.\n\nP3.",
            "key_points": ["K1", "K2"],
        })
        provider = MockProvider(mock_resp)
        scorer = NewsSummarizer(provider, make_mock_settings())
        scored_items = [make_scored_item(url=f"https://ex.com/{i}") for i in range(3)]
        results = await scorer.summarize_all(scored_items)
        assert len(results) == 3
        assert all(isinstance(r, SummarizedItem) for r in results)

    @pytest.mark.unit
    async def test_summarize_all_empty_returns_empty(self):
        from src.ai.summarizer import NewsSummarizer
        provider = MockProvider()
        scorer = NewsSummarizer(provider, make_mock_settings())
        results = await scorer.summarize_all([])
        assert results == []

    @pytest.mark.unit
    async def test_summarize_all_fallback_on_ai_failure(self):
        """When AI returns garbage, fallback summary is used (no crash)."""
        from src.ai.summarizer import NewsSummarizer
        provider = MockProvider("NOT JSON")
        scorer = NewsSummarizer(provider, make_mock_settings())
        scored_items = [make_scored_item()]
        results = await scorer.summarize_all(scored_items)
        assert len(results) == 1
        # Fallback: headline is the original title
        assert results[0].ai_headline == scored_items[0].item.title

    @pytest.mark.unit
    async def test_summarize_single_returns_summarized_item(self):
        from src.ai.summarizer import NewsSummarizer
        mock_resp = json.dumps({
            "headline": "Single Test Headline",
            "summary": "P1.\n\nP2.\n\nP3.",
            "key_points": ["K1"],
        })
        provider = MockProvider(mock_resp)
        scorer = NewsSummarizer(provider, make_mock_settings())
        result = await scorer.summarize_single(make_scored_item(), web_context="Some context")
        assert isinstance(result, SummarizedItem)
        assert result.ai_headline == "Single Test Headline"


# ===========================================================================
# BriefingBuilder Tests
# ===========================================================================


class TestBriefingBuilderTopics:
    @pytest.mark.unit
    def test_extract_top_topics_most_frequent(self):
        from src.briefing import BriefingBuilder
        items = [
            make_summarized_item(topics=["AI", "ML"]),
            make_summarized_item(topics=["AI", "Python"]),
            make_summarized_item(topics=["Python", "open source"]),
        ]
        topics = BriefingBuilder._extract_top_topics(items, top_n=3)
        assert "ai" in topics
        assert "python" in topics

    @pytest.mark.unit
    def test_extract_top_topics_empty_items(self):
        from src.briefing import BriefingBuilder
        assert BriefingBuilder._extract_top_topics([]) == []

    @pytest.mark.unit
    def test_extract_top_topics_respects_top_n(self):
        from src.briefing import BriefingBuilder
        items = [make_summarized_item(topics=["A", "B", "C", "D", "E", "F"])]
        topics = BriefingBuilder._extract_top_topics(items, top_n=3)
        assert len(topics) <= 3


class TestBriefingBuilderParseExecResponse:
    @pytest.mark.unit
    def test_clean_json_parsed(self):
        from src.briefing import BriefingBuilder
        raw = json.dumps({
            "executive_summary": "Today was big for AI.",
            "top_themes": ["AI", "open source"],
        })
        summary, themes = BriefingBuilder._parse_exec_response(raw)
        assert "AI" in summary
        assert "AI" in themes

    @pytest.mark.unit
    def test_no_json_raises_value_error(self):
        from src.briefing import BriefingBuilder
        with pytest.raises(ValueError, match="No JSON"):
            BriefingBuilder._parse_exec_response("plain text")

    @pytest.mark.unit
    def test_missing_summary_raises(self):
        from src.briefing import BriefingBuilder
        with pytest.raises(ValueError, match="executive_summary"):
            BriefingBuilder._parse_exec_response('{"executive_summary": "", "top_themes": []}')


class TestBriefingBuilderBuild:
    @pytest.mark.unit
    async def test_build_returns_briefing(self):
        from src.briefing import BriefingBuilder
        exec_resp = json.dumps({
            "executive_summary": "Good day in tech.",
            "top_themes": ["AI"],
        })
        provider = MockProvider(exec_resp)
        builder = BriefingBuilder(provider, make_mock_settings())
        items = [make_summarized_item(url=f"https://ex.com/{i}") for i in range(3)]
        briefing = await builder.build(items, total_fetched=50, total_scored=10)
        assert isinstance(briefing, Briefing)
        assert len(briefing.items) == 3
        assert briefing.total_fetched == 50
        assert briefing.total_scored == 10

    @pytest.mark.unit
    async def test_build_without_exec_summary(self):
        from src.briefing import BriefingBuilder
        provider = MockProvider()
        builder = BriefingBuilder(provider, make_mock_settings())
        items = [make_summarized_item()]
        briefing = await builder.build(items, generate_exec_summary=False)
        assert briefing.executive_summary  # fallback text

    @pytest.mark.unit
    async def test_build_empty_items(self):
        from src.briefing import BriefingBuilder
        provider = MockProvider()
        builder = BriefingBuilder(provider, make_mock_settings())
        briefing = await builder.build([], generate_exec_summary=False)
        assert isinstance(briefing, Briefing)
        assert briefing.items == []

    @pytest.mark.unit
    async def test_build_date_defaults_to_today(self):
        from src.briefing import BriefingBuilder
        provider = MockProvider()
        builder = BriefingBuilder(provider, make_mock_settings())
        briefing = await builder.build([], generate_exec_summary=False)
        assert briefing.date == date.today().isoformat()

    @pytest.mark.unit
    async def test_build_fallback_on_ai_failure(self):
        from src.briefing import BriefingBuilder
        provider = MockProvider("NOT JSON")  # will cause parse error
        builder = BriefingBuilder(provider, make_mock_settings())
        items = [make_summarized_item()]
        briefing = await builder.build(items, generate_exec_summary=True)
        assert briefing.executive_summary  # fallback applied, no crash


# ===========================================================================
# BriefingStore Tests
# ===========================================================================


class TestBriefingStore:
    @pytest.mark.unit
    def test_save_and_load_roundtrip(self, tmp_path):
        store = BriefingStore(tmp_path)
        briefing = make_briefing("2026-06-20")
        store.save(briefing)
        loaded = store.load("2026-06-20")
        assert loaded is not None
        assert loaded.date == "2026-06-20"
        assert len(loaded.items) == 3

    @pytest.mark.unit
    def test_load_nonexistent_returns_none(self, tmp_path):
        store = BriefingStore(tmp_path)
        assert store.load("2099-01-01") is None

    @pytest.mark.unit
    def test_exists_returns_true_after_save(self, tmp_path):
        store = BriefingStore(tmp_path)
        briefing = make_briefing("2026-06-20")
        store.save(briefing)
        assert store.exists("2026-06-20")
        assert not store.exists("2026-06-19")

    @pytest.mark.unit
    def test_list_dates_returns_sorted(self, tmp_path):
        store = BriefingStore(tmp_path)
        for d in ["2026-06-22", "2026-06-20", "2026-06-21"]:
            store.save(make_briefing(d))
        dates = store.list_dates()
        assert dates == ["2026-06-20", "2026-06-21", "2026-06-22"]

    @pytest.mark.unit
    def test_load_latest_returns_most_recent(self, tmp_path):
        store = BriefingStore(tmp_path)
        store.save(make_briefing("2026-06-19"))
        store.save(make_briefing("2026-06-20"))
        store.save(make_briefing("2026-06-18"))
        latest = store.load_latest()
        assert latest is not None
        assert latest.date == "2026-06-20"

    @pytest.mark.unit
    def test_load_latest_empty_store_returns_none(self, tmp_path):
        store = BriefingStore(tmp_path)
        assert store.load_latest() is None

    @pytest.mark.unit
    def test_count_returns_correct_number(self, tmp_path):
        store = BriefingStore(tmp_path)
        assert store.count() == 0
        store.save(make_briefing("2026-06-20"))
        assert store.count() == 1
        store.save(make_briefing("2026-06-21"))
        assert store.count() == 2

    @pytest.mark.unit
    def test_delete_removes_file(self, tmp_path):
        store = BriefingStore(tmp_path)
        store.save(make_briefing("2026-06-20"))
        assert store.exists("2026-06-20")
        deleted = store.delete("2026-06-20")
        assert deleted is True
        assert not store.exists("2026-06-20")

    @pytest.mark.unit
    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = BriefingStore(tmp_path)
        assert store.delete("2099-01-01") is False

    @pytest.mark.unit
    def test_load_range_returns_briefings_in_range(self, tmp_path):
        store = BriefingStore(tmp_path)
        for d in ["2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21"]:
            store.save(make_briefing(d))
        results = store.load_range("2026-06-19", "2026-06-20")
        dates = [r.date for r in results]
        assert dates == ["2026-06-19", "2026-06-20"]

    @pytest.mark.unit
    def test_invalid_date_format_raises_storage_error(self, tmp_path):
        store = BriefingStore(tmp_path)
        with pytest.raises(StorageError):
            store.load("not-a-date")

    @pytest.mark.unit
    def test_save_creates_json_file(self, tmp_path):
        store = BriefingStore(tmp_path)
        store.save(make_briefing("2026-06-20"))
        json_file = tmp_path / "briefings" / "2026-06-20.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["date"] == "2026-06-20"

    @pytest.mark.unit
    def test_save_overwrites_existing(self, tmp_path):
        store = BriefingStore(tmp_path)
        store.save(make_briefing("2026-06-20", items=3))
        store.save(make_briefing("2026-06-20", items=5))
        loaded = store.load("2026-06-20")
        assert loaded is not None
        assert len(loaded.items) == 5

    @pytest.mark.unit
    def test_cleanup_old_removes_stale_briefings(self, tmp_path):
        store = BriefingStore(tmp_path)
        # Save some very old briefings
        store.save(make_briefing("2020-01-01"))
        store.save(make_briefing("2020-06-15"))
        # Save a recent one
        store.save(make_briefing(date.today().isoformat()))
        deleted = store.cleanup_old(keep_days=30)
        assert deleted == 2
        assert store.count() == 1  # only today's remains
