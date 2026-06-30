"""
src/retry.py
=============
Resilient retry logic for AI provider calls.

Why a dedicated retry module?
  AI APIs fail in predictable, recoverable ways:
    - HTTP 429 (rate limit): back off and try again
    - HTTP 503/502 (transient server error): retry with jitter
    - Network timeout: retry a few times before giving up
    - Context window exceeded: no point retrying — fail immediately

  tenacity already handles exponential backoff, but we need:
    1. A CENTRALIZED policy so all three providers (OpenAI, Gemini,
       Anthropic) use identical retry behavior
    2. Rate-limit HEADER parsing to respect the actual Retry-After value
    3. COST TRACKING — log token usage and estimated $ per call
    4. CIRCUIT BREAKER awareness — stop retrying after N consecutive fails

Usage:
    from src.retry import with_ai_retry, CostTracker

    # Wrap any async AI call with retry + backoff
    result = await with_ai_retry(provider.complete, prompt, max_tokens=512)

    # Track token costs across a whole pipeline run
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40)
    print(tracker.summary())   # "Total: 0.0032¢ (120 tokens, 3 calls)"
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.exceptions import AIError, AIProviderError, TokenLimitError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry policy constants
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 4          # 1 initial + 3 retries
_BASE_DELAY_S = 1.0        # starting backoff delay (seconds)
_MAX_DELAY_S = 60.0        # cap on how long we wait between retries
_JITTER_FRACTION = 0.25    # ±25% randomization to avoid thundering herd

# Exceptions that should never be retried (client errors, not server errors)
_FATAL_EXCEPTIONS = (
    AIProviderError,   # auth errors, quota exceeded, invalid key
    TokenLimitError,   # context window exceeded — no point retrying
)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Pricing per 1M tokens as of 2026 (update if OpenAI changes pricing)
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # model_prefix: (input_$/1M, output_$/1M)
    "gpt-4o-mini":          (0.15,  0.60),
    "gpt-4o":               (5.00,  15.00),
    "o1-mini":              (3.00,  12.00),
    "o3-mini":              (1.10,  4.40),
    "gemini-1.5-flash":     (0.075, 0.30),
    "gemini-1.5-pro":       (3.50,  10.50),
    "gemini-2.0-flash":     (0.10,  0.40),
    "claude-3-haiku":       (0.25,  1.25),
    "claude-3-5-haiku":     (0.80,  4.00),
    "claude-3-5-sonnet":    (3.00,  15.00),
}


def _pricing_for(model: str) -> tuple[float, float]:
    """Look up pricing for a model name by prefix matching."""
    model_lower = model.lower()
    for prefix, pricing in _PRICING_PER_1M.items():
        if model_lower.startswith(prefix):
            return pricing
    # Unknown model — return a conservative estimate
    return (5.00, 15.00)


@dataclass
class CallRecord:
    """One AI API call with its token usage."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    duration_s: float
    attempt: int = 1
    succeeded: bool = True


@dataclass
class CostTracker:
    """
    Accumulates token usage and estimated cost across a full pipeline run.

    Usage:
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=80, completion_tokens=40)
        print(tracker.total_cost_usd)      # 0.000036
        print(tracker.summary())           # human-readable summary
    """

    records: list[CallRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        duration_s: float = 0.0,
        attempt: int = 1,
        succeeded: bool = True,
    ) -> float:
        """
        Record one AI API call and return its estimated cost in USD.

        Args:
            model: Model name string (e.g. "gpt-4o-mini")
            prompt_tokens: Number of input tokens used
            completion_tokens: Number of output tokens generated
            duration_s: Wall-clock time for the call in seconds
            attempt: Which retry attempt this was (1 = first try)
            succeeded: Whether the call succeeded

        Returns:
            Estimated cost of this call in USD
        """
        input_rate, output_rate = _pricing_for(model)
        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
        rec = CallRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            duration_s=duration_s,
            attempt=attempt,
            succeeded=succeeded,
        )
        self.records.append(rec)
        return cost

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost across all recorded calls."""
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion) across all calls."""
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    @property
    def total_calls(self) -> int:
        """Total number of API calls recorded."""
        return len(self.records)

    @property
    def failed_calls(self) -> int:
        """Number of calls that did not succeed."""
        return sum(1 for r in self.records if not r.succeeded)

    @property
    def retry_calls(self) -> int:
        """Number of calls that were retries (attempt > 1)."""
        return sum(1 for r in self.records if r.attempt > 1)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        cost_cents = self.total_cost_usd * 100
        if cost_cents < 0.01:
            cost_str = f"< 0.01¢"
        else:
            cost_str = f"~{cost_cents:.2f}¢"

        parts = [
            f"Cost: {cost_str}",
            f"tokens: {self.total_tokens:,}",
            f"calls: {self.total_calls}",
        ]
        if self.retry_calls:
            parts.append(f"retries: {self.retry_calls}")
        if self.failed_calls:
            parts.append(f"failed: {self.failed_calls}")
        return " | ".join(parts)

    def per_model_breakdown(self) -> dict[str, dict]:
        """Group cost and token stats by model."""
        breakdown: dict[str, dict] = {}
        for r in self.records:
            m = breakdown.setdefault(r.model, {
                "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "cost_usd": 0.0,
            })
            m["calls"] += 1
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens
            m["cost_usd"] += r.cost_usd
        return breakdown


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _calc_delay(attempt: int, retry_after: float | None = None) -> float:
    """
    Calculate how long to wait before the next retry.

    Uses exponential backoff with jitter:
      delay = min(base * 2^attempt, max) * (1 ± jitter)

    If the API returned a Retry-After header, that takes priority.
    """
    if retry_after is not None and retry_after > 0:
        # Respect the server's requested wait time, plus a little jitter
        jitter = random.uniform(0, _JITTER_FRACTION * retry_after)
        return min(retry_after + jitter, _MAX_DELAY_S)

    # Exponential backoff: 1s, 2s, 4s, 8s... capped at 60s
    base_delay = min(_BASE_DELAY_S * (2 ** attempt), _MAX_DELAY_S)
    jitter = random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION) * base_delay
    return max(0.1, base_delay + jitter)


def _extract_retry_after(exc: Exception) -> float | None:
    """
    Try to extract a Retry-After value from a rate-limit exception.

    Different SDKs expose headers differently:
      - openai.RateLimitError has no direct retry_after, but sometimes
        the message contains "try again in Xs"
      - We parse the message as a best-effort approach.
    """
    msg = str(exc).lower()

    # "Please try again in 2.5s" / "retry after 30s" patterns
    import re
    patterns = [
        r"try again in ([\d.]+)s",
        r"retry.?after[:\s]+([\d.]+)",
        r"wait ([\d.]+) second",
    ]
    for pat in patterns:
        m = re.search(pat, msg)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


async def with_ai_retry(
    coro_func: Callable[..., Coroutine[Any, Any, str]],
    *args: Any,
    cost_tracker: CostTracker | None = None,
    model: str = "unknown",
    **kwargs: Any,
) -> str:
    """
    Execute an async AI call with exponential backoff retry.

    Retries on transient errors (AIError, network failures).
    Does NOT retry on fatal errors (auth failure, quota exceeded,
    context window exceeded).

    Args:
        coro_func: Async function to call (e.g. provider.complete)
        *args: Positional arguments for coro_func
        cost_tracker: Optional CostTracker to record token usage
        model: Model name for cost tracking (e.g. "gpt-4o-mini")
        **kwargs: Keyword arguments for coro_func

    Returns:
        The string response from the AI provider.

    Raises:
        AIProviderError: On auth/quota failures (not retried)
        TokenLimitError: On context window exceeded (not retried)
        AIError: If all retry attempts are exhausted
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        t0 = time.monotonic()
        try:
            result = await coro_func(*args, **kwargs)
            duration = time.monotonic() - t0

            # Record a successful call (token counts unknown without SDK access)
            if cost_tracker is not None:
                # Estimate tokens from prompt length if no usage object available
                prompt_text = args[0] if args else kwargs.get("prompt", "")
                est_prompt = max(1, len(str(prompt_text)) // 4)
                est_completion = max(1, len(result) // 4)
                cost_tracker.record(
                    model=model,
                    prompt_tokens=est_prompt,
                    completion_tokens=est_completion,
                    duration_s=duration,
                    attempt=attempt + 1,
                    succeeded=True,
                )

            if attempt > 0:
                log.info("AI call succeeded on attempt %d/%d", attempt + 1, _MAX_ATTEMPTS)
            return result

        except _FATAL_EXCEPTIONS as e:
            # Non-retryable — fail immediately
            duration = time.monotonic() - t0
            if cost_tracker is not None:
                cost_tracker.record(
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration_s=duration,
                    attempt=attempt + 1,
                    succeeded=False,
                )
            raise

        except (AIError, Exception) as e:
            last_exc = e
            duration = time.monotonic() - t0

            if cost_tracker is not None:
                cost_tracker.record(
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration_s=duration,
                    attempt=attempt + 1,
                    succeeded=False,
                )

            if attempt == _MAX_ATTEMPTS - 1:
                # Final attempt failed
                log.error(
                    "AI call failed after %d attempts: %s",
                    _MAX_ATTEMPTS, e,
                )
                break

            # Calculate backoff
            retry_after = _extract_retry_after(e)
            delay = _calc_delay(attempt, retry_after)

            log.warning(
                "AI call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, _MAX_ATTEMPTS, type(e).__name__, delay,
            )
            await asyncio.sleep(delay)

    # All attempts exhausted
    raise AIError(
        f"AI call failed after {_MAX_ATTEMPTS} attempts: {last_exc}",
        model=model,
    ) from last_exc


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------

def ai_retry(model_attr: str = "model", tracker_attr: str | None = None):
    """
    Class method decorator that wraps ``complete()`` with retry logic.

    Usage on a provider class::

        class MyProvider(BaseAIProvider):
            @ai_retry(model_attr="model")
            async def complete(self, prompt, *, max_tokens=512, temperature=0.3):
                ...

    ``model_attr``: name of the instance attribute holding the model string.
    ``tracker_attr``: name of the instance attribute holding a CostTracker.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            model = getattr(self, model_attr, "unknown")
            tracker = getattr(self, tracker_attr, None) if tracker_attr else None
            return await with_ai_retry(
                func,
                self,
                *args,
                cost_tracker=tracker,
                model=model,
                **kwargs,
            )
        return wrapper
    return decorator
