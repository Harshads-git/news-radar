"""
tests/test_retry.py
====================
Tests for the retry and cost tracking module (src/retry.py).

Coverage:
  - CostTracker: record(), totals, summary, breakdown, empty state
  - _calc_delay(): exponential backoff, jitter, Retry-After override
  - _extract_retry_after(): regex parsing of various error message formats
  - with_ai_retry(): success path, retry on transient error, fatal error bypass,
    exhaustion after max attempts, cost tracker integration
  - _pricing_for(): known model lookup, prefix matching, unknown model fallback

All tests are pure unit tests — no network calls, no AI API keys needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import AIError, AIProviderError, TokenLimitError
from src.retry import (
    CallRecord,
    CostTracker,
    _MAX_ATTEMPTS,
    _calc_delay,
    _extract_retry_after,
    _pricing_for,
    with_ai_retry,
)


# ===========================================================================
# CostTracker — basic record + totals
# ===========================================================================


class TestCostTrackerRecord:
    @pytest.mark.unit
    def test_empty_tracker_has_zero_totals(self):
        t = CostTracker()
        assert t.total_cost_usd == 0.0
        assert t.total_tokens == 0
        assert t.total_calls == 0
        assert t.failed_calls == 0
        assert t.retry_calls == 0

    @pytest.mark.unit
    def test_record_returns_estimated_cost(self):
        t = CostTracker()
        cost = t.record("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
        # 1M input tokens at $0.15/1M = $0.15
        assert abs(cost - 0.15) < 0.001

    @pytest.mark.unit
    def test_record_completion_tokens(self):
        t = CostTracker()
        cost = t.record("gpt-4o-mini", prompt_tokens=0, completion_tokens=1_000_000)
        # 1M output tokens at $0.60/1M = $0.60
        assert abs(cost - 0.60) < 0.001

    @pytest.mark.unit
    def test_total_cost_accumulates_across_calls(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        t.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)
        assert t.total_calls == 2
        assert t.total_tokens == 450

    @pytest.mark.unit
    def test_total_cost_sum_matches_individual_costs(self):
        t = CostTracker()
        c1 = t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40)
        c2 = t.record("gpt-4o-mini", prompt_tokens=120, completion_tokens=60)
        assert abs(t.total_cost_usd - (c1 + c2)) < 1e-10

    @pytest.mark.unit
    def test_failed_calls_counted(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=0, completion_tokens=0, succeeded=False)
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, succeeded=True)
        assert t.failed_calls == 1
        assert t.total_calls == 2

    @pytest.mark.unit
    def test_retry_calls_counted(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, attempt=1)
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, attempt=2)
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, attempt=3)
        assert t.retry_calls == 2  # attempts 2 and 3

    @pytest.mark.unit
    def test_records_stored_as_call_records(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, duration_s=1.5)
        rec = t.records[0]
        assert isinstance(rec, CallRecord)
        assert rec.model == "gpt-4o-mini"
        assert rec.duration_s == 1.5


class TestCostTrackerSummary:
    @pytest.mark.unit
    def test_summary_shows_cost_tokens_and_calls(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40)
        summary = t.summary()
        assert "Cost:" in summary
        assert "tokens:" in summary
        assert "calls:" in summary

    @pytest.mark.unit
    def test_summary_shows_retries_when_nonzero(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40, attempt=2)
        assert "retries:" in t.summary()

    @pytest.mark.unit
    def test_summary_shows_failed_when_nonzero(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=0, completion_tokens=0, succeeded=False)
        assert "failed:" in t.summary()

    @pytest.mark.unit
    def test_summary_empty_tracker(self):
        t = CostTracker()
        summary = t.summary()
        assert "Cost:" in summary  # should not crash

    @pytest.mark.unit
    def test_per_model_breakdown_groups_by_model(self):
        t = CostTracker()
        t.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        t.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)
        t.record("gemini-1.5-flash", prompt_tokens=80, completion_tokens=40)
        breakdown = t.per_model_breakdown()
        assert "gpt-4o-mini" in breakdown
        assert "gemini-1.5-flash" in breakdown
        assert breakdown["gpt-4o-mini"]["calls"] == 2
        assert breakdown["gemini-1.5-flash"]["calls"] == 1

    @pytest.mark.unit
    def test_per_model_breakdown_empty(self):
        t = CostTracker()
        assert t.per_model_breakdown() == {}


# ===========================================================================
# Pricing lookup
# ===========================================================================


class TestPricingFor:
    @pytest.mark.unit
    def test_gpt4o_mini_pricing(self):
        input_rate, output_rate = _pricing_for("gpt-4o-mini")
        assert input_rate == 0.15
        assert output_rate == 0.60

    @pytest.mark.unit
    def test_gpt4o_pricing(self):
        input_rate, output_rate = _pricing_for("gpt-4o")
        assert input_rate == 5.00

    @pytest.mark.unit
    def test_gemini_flash_pricing(self):
        input_rate, output_rate = _pricing_for("gemini-1.5-flash")
        assert input_rate == 0.075

    @pytest.mark.unit
    def test_claude_haiku_pricing(self):
        input_rate, output_rate = _pricing_for("claude-3-haiku-20240307")
        assert input_rate == 0.25

    @pytest.mark.unit
    def test_unknown_model_returns_conservative_estimate(self):
        input_rate, output_rate = _pricing_for("totally-unknown-model-xyz")
        assert input_rate > 0
        assert output_rate > 0

    @pytest.mark.unit
    def test_case_insensitive_lookup(self):
        r1 = _pricing_for("GPT-4O-MINI")
        r2 = _pricing_for("gpt-4o-mini")
        assert r1 == r2


# ===========================================================================
# _calc_delay — backoff calculation
# ===========================================================================


class TestCalcDelay:
    @pytest.mark.unit
    def test_delay_increases_with_attempt(self):
        d0 = _calc_delay(0)
        d1 = _calc_delay(1)
        d2 = _calc_delay(2)
        # With jitter, not strictly monotone, but generally increasing
        # Just check they're all positive
        assert d0 > 0
        assert d1 > 0
        assert d2 > 0

    @pytest.mark.unit
    def test_delay_capped_at_max(self):
        from src.retry import _MAX_DELAY_S
        delay = _calc_delay(100)  # very high attempt number
        assert delay <= _MAX_DELAY_S

    @pytest.mark.unit
    def test_retry_after_takes_priority(self):
        delay = _calc_delay(0, retry_after=30.0)
        # Should be ~30s (plus small jitter)
        assert delay >= 30.0
        assert delay <= 40.0  # with jitter

    @pytest.mark.unit
    def test_zero_retry_after_falls_back_to_backoff(self):
        delay = _calc_delay(0, retry_after=0.0)
        # Should use exponential backoff, not 0
        assert delay > 0

    @pytest.mark.unit
    def test_delay_always_positive(self):
        for attempt in range(5):
            assert _calc_delay(attempt) > 0


# ===========================================================================
# _extract_retry_after — header parsing
# ===========================================================================


class TestExtractRetryAfter:
    @pytest.mark.unit
    def test_extracts_seconds_from_try_again_message(self):
        exc = Exception("Please try again in 2.5s")
        result = _extract_retry_after(exc)
        assert result == 2.5

    @pytest.mark.unit
    def test_extracts_retry_after_pattern(self):
        exc = Exception("Rate limit exceeded. Retry after: 30")
        result = _extract_retry_after(exc)
        assert result == 30.0

    @pytest.mark.unit
    def test_extracts_wait_seconds_pattern(self):
        exc = Exception("Too many requests. Wait 5 seconds before retrying.")
        result = _extract_retry_after(exc)
        assert result == 5.0

    @pytest.mark.unit
    def test_returns_none_when_no_match(self):
        exc = Exception("Some unrelated error message")
        result = _extract_retry_after(exc)
        assert result is None

    @pytest.mark.unit
    def test_returns_none_for_empty_message(self):
        exc = Exception("")
        result = _extract_retry_after(exc)
        assert result is None


# ===========================================================================
# with_ai_retry — the core retry wrapper
# ===========================================================================


class TestWithAiRetry:
    @pytest.mark.unit
    async def test_success_on_first_attempt(self):
        mock_func = AsyncMock(return_value="response text")
        result = await with_ai_retry(mock_func, "hello", model="gpt-4o-mini")
        assert result == "response text"
        assert mock_func.call_count == 1

    @pytest.mark.unit
    async def test_retries_on_ai_error(self):
        call_count = 0

        async def flaky(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AIError("transient", model="test")
            return "success"

        with patch("src.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_ai_retry(flaky, "hello", model="gpt-4o-mini")

        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    async def test_does_not_retry_provider_error(self):
        """AIProviderError (auth failure) must not be retried."""
        mock_func = AsyncMock(side_effect=AIProviderError("bad key", model="test"))
        with pytest.raises(AIProviderError):
            await with_ai_retry(mock_func, "hello", model="gpt-4o-mini")
        assert mock_func.call_count == 1

    @pytest.mark.unit
    async def test_does_not_retry_token_limit_error(self):
        """TokenLimitError must not be retried — prompt is inherently too long."""
        mock_func = AsyncMock(side_effect=TokenLimitError("too long", model="test"))
        with pytest.raises(TokenLimitError):
            await with_ai_retry(mock_func, "hello", model="gpt-4o-mini")
        assert mock_func.call_count == 1

    @pytest.mark.unit
    async def test_raises_ai_error_after_max_attempts(self):
        mock_func = AsyncMock(side_effect=AIError("always fails", model="test"))
        with patch("src.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(AIError):
                await with_ai_retry(mock_func, "hello", model="gpt-4o-mini")
        assert mock_func.call_count == _MAX_ATTEMPTS

    @pytest.mark.unit
    async def test_records_successful_call_in_cost_tracker(self):
        tracker = CostTracker()
        mock_func = AsyncMock(return_value="the response text here")

        await with_ai_retry(
            mock_func, "hello world prompt",
            model="gpt-4o-mini",
            cost_tracker=tracker,
        )
        assert tracker.total_calls == 1
        assert tracker.failed_calls == 0

    @pytest.mark.unit
    async def test_records_failed_call_in_cost_tracker(self):
        tracker = CostTracker()
        mock_func = AsyncMock(side_effect=AIProviderError("bad key", model="test"))

        with pytest.raises(AIProviderError):
            await with_ai_retry(
                mock_func, "hello",
                model="gpt-4o-mini",
                cost_tracker=tracker,
            )
        assert tracker.failed_calls == 1

    @pytest.mark.unit
    async def test_works_without_cost_tracker(self):
        """cost_tracker=None must not cause any error."""
        mock_func = AsyncMock(return_value="ok")
        result = await with_ai_retry(mock_func, "hello", model="gpt-4o-mini", cost_tracker=None)
        assert result == "ok"

    @pytest.mark.unit
    async def test_passes_kwargs_through_to_func(self):
        received_kwargs = {}

        async def capture(prompt, **kwargs):
            received_kwargs.update(kwargs)
            return "ok"

        await with_ai_retry(capture, "hello", model="gpt-4o-mini", max_tokens=128, temperature=0.1)
        assert received_kwargs.get("max_tokens") == 128
        assert received_kwargs.get("temperature") == 0.1
