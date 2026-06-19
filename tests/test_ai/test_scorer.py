"""
tests/test_ai/test_scorer.py
=============================
Unit tests for the AI layer: BaseAIProvider, AIProviderFactory, and NewsScorer.

All tests use a MockProvider (implements BaseAIProvider with a fake complete())
so no real API calls are made.

Test coverage:
  - _parse_score_response: clean JSON, markdown-wrapped, missing fields, clamping
  - _build_score_prompt: contains required fields from the item
  - score_item: end-to-end scoring with mock provider
  - complete_safe: returns fallback on errors
  - AIProviderFactory: model routing
  - NewsScorer.score_all: filtering, sorting, concurrency
  - NewsScorer.score_single: single item scoring
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.base import AIProviderFactory, BaseAIProvider
from src.exceptions import AIError, AIProviderError
from src.models import NewsItem, ScoredItem


# ---------------------------------------------------------------------------
# MockProvider — concrete subclass for testing
# ---------------------------------------------------------------------------


class MockProvider(BaseAIProvider):
    """A concrete BaseAIProvider that returns configurable mock responses."""

    PROVIDER_NAME = "mock"

    def __init__(self, response: str = '{"score": 8, "reason": "Very relevant", "topics": ["AI"]}'):
        super().__init__("mock-model")
        self._response = response
        self.call_count = 0

    async def complete(self, prompt: str, *, max_tokens=512, temperature=0.3, system=None) -> str:
        self.call_count += 1
        return self._response


class FailingProvider(BaseAIProvider):
    """A provider that always raises AIError."""

    PROVIDER_NAME = "failing"

    def __init__(self):
        super().__init__("fail-model")

    async def complete(self, prompt: str, **kwargs) -> str:
        raise AIError("API is down", model="fail-model")


# ---------------------------------------------------------------------------
# Helper: create a NewsItem for testing
# ---------------------------------------------------------------------------


def make_news_item(
    url: str = "https://techcrunch.com/ai-article",
    title: str = "OpenAI Releases GPT-5 with Groundbreaking Reasoning",
    score: int = 500,
    source_type: str = "hackernews",
) -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        summary="OpenAI announces its most capable model yet.",
        author="sam_altman",
        source_id="hn-api",
        source_name="Hacker News",
        source_type=source_type,
        score=score,
        comment_count=312,
        published_at=datetime(2026, 6, 19, 10, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# _parse_score_response Tests
# ---------------------------------------------------------------------------


class TestParseScoreResponse:
    @pytest.mark.unit
    def test_clean_json_parsed_correctly(self):
        raw = '{"score": 8, "reason": "Very relevant to AI topic", "topics": ["AI", "ML"]}'
        item = make_news_item()
        score, reason, topics = MockProvider._parse_score_response(raw, item)
        assert score == 8
        assert reason == "Very relevant to AI topic"
        assert "AI" in topics

    @pytest.mark.unit
    def test_markdown_fences_stripped(self):
        raw = '```json\n{"score": 7, "reason": "Interesting", "topics": ["tech"]}\n```'
        item = make_news_item()
        score, reason, topics = MockProvider._parse_score_response(raw, item)
        assert score == 7

    @pytest.mark.unit
    def test_score_clamped_to_10(self):
        raw = '{"score": 15, "reason": "Extreme", "topics": []}'
        item = make_news_item()
        score, _, _ = MockProvider._parse_score_response(raw, item)
        assert score == 10

    @pytest.mark.unit
    def test_score_clamped_to_1(self):
        raw = '{"score": -5, "reason": "Negative", "topics": []}'
        item = make_news_item()
        score, _, _ = MockProvider._parse_score_response(raw, item)
        assert score == 1

    @pytest.mark.unit
    def test_missing_reason_uses_default(self):
        raw = '{"score": 5, "topics": []}'
        item = make_news_item()
        _, reason, _ = MockProvider._parse_score_response(raw, item)
        assert reason  # should not be empty

    @pytest.mark.unit
    def test_missing_topics_returns_empty_list(self):
        raw = '{"score": 6, "reason": "OK"}'
        item = make_news_item()
        _, _, topics = MockProvider._parse_score_response(raw, item)
        assert topics == []

    @pytest.mark.unit
    def test_no_json_raises_value_error(self):
        raw = "I cannot score this item."
        item = make_news_item()
        with pytest.raises(ValueError, match="No JSON"):
            MockProvider._parse_score_response(raw, item)

    @pytest.mark.unit
    def test_json_embedded_in_text(self):
        """JSON embedded in surrounding text must still be parsed."""
        raw = 'Here is my answer: {"score": 9, "reason": "Top story", "topics": ["news"]} Done.'
        item = make_news_item()
        score, _, _ = MockProvider._parse_score_response(raw, item)
        assert score == 9


# ---------------------------------------------------------------------------
# _build_score_prompt Tests
# ---------------------------------------------------------------------------


class TestBuildScorePrompt:
    @pytest.mark.unit
    def test_prompt_contains_title(self):
        provider = MockProvider()
        item = make_news_item(title="Unique Test Title XYZ123")
        prompt = provider._build_score_prompt(item, "AI and Python", "")
        assert "Unique Test Title XYZ123" in prompt

    @pytest.mark.unit
    def test_prompt_contains_user_interests(self):
        provider = MockProvider()
        item = make_news_item()
        prompt = provider._build_score_prompt(item, "AI and Python", "")
        assert "AI and Python" in prompt

    @pytest.mark.unit
    def test_prompt_contains_source_name(self):
        provider = MockProvider()
        item = make_news_item()
        prompt = provider._build_score_prompt(item, "tech", "")
        assert "Hacker News" in prompt

    @pytest.mark.unit
    def test_prompt_contains_web_context(self):
        provider = MockProvider()
        item = make_news_item()
        prompt = provider._build_score_prompt(item, "tech", "GPT-5 is the latest reasoning model")
        assert "GPT-5" in prompt

    @pytest.mark.unit
    def test_prompt_contains_score_guide(self):
        provider = MockProvider()
        item = make_news_item()
        prompt = provider._build_score_prompt(item, "tech", "")
        assert "9-10" in prompt or "Must-read" in prompt


# ---------------------------------------------------------------------------
# score_item Tests
# ---------------------------------------------------------------------------


class TestScoreItem:
    @pytest.mark.unit
    async def test_score_item_returns_scored_item(self):
        provider = MockProvider('{"score": 8, "reason": "Very relevant", "topics": ["AI"]}')
        item = make_news_item()
        result = await provider.score_item(item, "AI and ML")
        assert isinstance(result, ScoredItem)

    @pytest.mark.unit
    async def test_score_item_has_correct_score(self):
        provider = MockProvider('{"score": 9, "reason": "Must read", "topics": ["AI"]}')
        item = make_news_item()
        result = await provider.score_item(item, "AI")
        assert result.ai_score == 9

    @pytest.mark.unit
    async def test_score_item_preserves_original_item(self):
        provider = MockProvider('{"score": 7, "reason": "Good", "topics": []}')
        item = make_news_item(title="My Test Article")
        result = await provider.score_item(item, "tech")
        assert result.item.title == "My Test Article"
        assert result.item is item

    @pytest.mark.unit
    async def test_score_item_fallback_on_parse_failure(self):
        """When AI returns garbage, score_item returns score=5 (neutral fallback)."""
        provider = MockProvider("This is not JSON at all!!!")
        item = make_news_item()
        result = await provider.score_item(item, "AI")
        assert result.ai_score == 5  # fallback neutral score

    @pytest.mark.unit
    async def test_score_item_records_model_used(self):
        provider = MockProvider('{"score": 8, "reason": "OK", "topics": []}')
        item = make_news_item()
        result = await provider.score_item(item, "tech")
        assert result.model_used == "mock-model"

    @pytest.mark.unit
    async def test_complete_safe_returns_fallback_on_error(self):
        provider = FailingProvider()
        result = await provider.complete_safe("test prompt", fallback="FALLBACK")
        assert result == "FALLBACK"

    @pytest.mark.unit
    async def test_complete_safe_returns_response_on_success(self):
        provider = MockProvider("actual response")
        result = await provider.complete_safe("test prompt")
        assert result == "actual response"


# ---------------------------------------------------------------------------
# AIProviderFactory Tests
# ---------------------------------------------------------------------------


class TestAIProviderFactory:
    @pytest.mark.unit
    def test_gpt_model_returns_openai_provider(self):
        from src.ai.openai_adapter import OpenAIProvider
        p = AIProviderFactory.from_model("gpt-4o-mini")
        assert isinstance(p, OpenAIProvider)

    @pytest.mark.unit
    def test_gemini_model_returns_gemini_provider(self):
        from src.ai.gemini_adapter import GeminiProvider
        p = AIProviderFactory.from_model("gemini-1.5-flash")
        assert isinstance(p, GeminiProvider)

    @pytest.mark.unit
    def test_claude_model_returns_anthropic_provider(self):
        from src.ai.anthropic_adapter import AnthropicProvider
        p = AIProviderFactory.from_model("claude-3-5-haiku-20241022")
        assert isinstance(p, AnthropicProvider)

    @pytest.mark.unit
    def test_o1_model_routes_to_openai(self):
        from src.ai.openai_adapter import OpenAIProvider
        p = AIProviderFactory.from_model("o1-mini")
        assert isinstance(p, OpenAIProvider)

    @pytest.mark.unit
    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            AIProviderFactory.from_model("unknown-model-xyz")

    @pytest.mark.unit
    def test_model_name_case_insensitive(self):
        from src.ai.openai_adapter import OpenAIProvider
        p = AIProviderFactory.from_model("GPT-4O-MINI")
        assert isinstance(p, OpenAIProvider)


# ---------------------------------------------------------------------------
# NewsScorer Tests
# ---------------------------------------------------------------------------


class TestNewsScorer:
    """Tests for the batch NewsScorer."""

    def _make_settings(self, threshold: int = 6, max_items: int = 20):
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.score_threshold = threshold
        settings.max_briefing_items = max_items
        settings.user_interests = "AI, Python, open source"
        return settings

    @pytest.mark.unit
    async def test_score_all_returns_scored_items(self):
        from src.ai.scorer import NewsScorer
        provider = MockProvider('{"score": 8, "reason": "Good", "topics": ["AI"]}')
        scorer = NewsScorer(provider, self._make_settings())
        items = [make_news_item(url=f"https://example.com/{i}") for i in range(3)]
        results = await scorer.score_all(items, fetch_context=False)
        assert len(results) == 3
        assert all(isinstance(r, ScoredItem) for r in results)

    @pytest.mark.unit
    async def test_score_all_filters_below_threshold(self):
        from src.ai.scorer import NewsScorer
        provider = MockProvider('{"score": 3, "reason": "Low relevance", "topics": []}')
        scorer = NewsScorer(provider, self._make_settings(threshold=6))
        items = [make_news_item(url=f"https://example.com/{i}") for i in range(5)]
        results = await scorer.score_all(items, fetch_context=False)
        assert results == []  # all items scored 3 < threshold 6

    @pytest.mark.unit
    async def test_score_all_respects_max_items(self):
        from src.ai.scorer import NewsScorer
        provider = MockProvider('{"score": 8, "reason": "Good", "topics": []}')
        scorer = NewsScorer(provider, self._make_settings(threshold=1, max_items=3))
        items = [make_news_item(url=f"https://example.com/{i}") for i in range(10)]
        results = await scorer.score_all(items, fetch_context=False)
        assert len(results) <= 3

    @pytest.mark.unit
    async def test_score_all_empty_input_returns_empty(self):
        from src.ai.scorer import NewsScorer
        provider = MockProvider()
        scorer = NewsScorer(provider, self._make_settings())
        results = await scorer.score_all([], fetch_context=False)
        assert results == []

    @pytest.mark.unit
    async def test_score_all_sorted_highest_first(self):
        from src.ai.scorer import NewsScorer
        responses = [
            '{"score": 3, "reason": "Low", "topics": []}',
            '{"score": 9, "reason": "High", "topics": []}',
            '{"score": 6, "reason": "Mid", "topics": []}',
        ]
        call_idx = 0

        class SequentialProvider(BaseAIProvider):
            PROVIDER_NAME = "sequential"
            async def complete(self, prompt, **kwargs):
                nonlocal call_idx
                r = responses[call_idx % len(responses)]
                call_idx += 1
                return r

        provider = SequentialProvider("seq-model")
        scorer = NewsScorer(provider, self._make_settings(threshold=1))
        items = [make_news_item(url=f"https://example.com/{i}") for i in range(3)]
        results = await scorer.score_all(items, fetch_context=False)
        scores = [r.ai_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    async def test_score_single_returns_scored_item(self):
        from src.ai.scorer import NewsScorer
        provider = MockProvider('{"score": 7, "reason": "Good", "topics": ["Python"]}')
        scorer = NewsScorer(provider, self._make_settings())
        item = make_news_item()
        result = await scorer.score_single(item, web_context="Python is great")
        assert isinstance(result, ScoredItem)
        assert result.ai_score == 7
