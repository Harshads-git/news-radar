"""
src/ai/base.py
==============
Abstract base class for all AI provider adapters.

Design rationale — why adapters?
  The pipeline should be provider-agnostic. Whether the user has an
  OpenAI key, Gemini key, or Anthropic key, the scorer and summarizer
  call the exact same interface. Switching models requires zero code
  changes — just update AI_MODEL in .env.

  This follows the Adapter Pattern: each concrete class wraps a different
  SDK (openai, google-generativeai, anthropic) behind the same interface.

Interface summary:
    complete(prompt, *, max_tokens, temperature) → str
        Single-turn text completion. Returns the model's response as
        a plain string. Raises AIError on failure.

    score_item(item, user_interests) → ScoredItem
        Higher-level method: builds the scoring prompt and calls complete().
        Implemented in BaseAIProvider (not abstract) so subclasses only
        need to implement complete().

The AIProviderFactory (bottom of this file) maps model name prefixes
to provider classes so the orchestrator can call:

    provider = AIProviderFactory.from_model("gpt-4o-mini")
    scored = await provider.score_item(item, interests)
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.retry import CostTracker, with_ai_retry

if TYPE_CHECKING:
    from src.models import NewsItem, ScoredItem

# ---------------------------------------------------------------------------
# Scoring prompt template
# ---------------------------------------------------------------------------

_SCORE_PROMPT_TEMPLATE = """\
You are an expert news curator. Score the relevance and quality of a news story
for a reader interested in: {interests}

Story to evaluate:
  Title: {title}
  Source: {source} ({source_type})
  Published: {published}
  Summary: {summary}
  Platform score: {platform_score}
  Comments: {comments}
  Web context: {web_context}

Respond ONLY with a JSON object in exactly this format (no markdown, no preamble):
{{
  "score": <integer 1-10>,
  "reason": "<one sentence explaining the score>",
  "topics": ["<topic1>", "<topic2>"]
}}

Scoring guide:
  9-10: Must-read. Highly relevant, significant, well-sourced.
  7-8:  Worth reading. Relevant and interesting.
  5-6:  Mildly interesting. Tangentially related.
  3-4:  Low relevance. Off-topic or low quality.
  1-2:  Skip. Spam, job post, or completely irrelevant.
"""


class BaseAIProvider(ABC):
    """
    Abstract base for AI provider adapters.

    Subclasses implement only ``complete()`` — the raw text generation call.
    All higher-level methods (``score_item()``, ``summarize()``) are
    implemented here and work across all providers.
    """

    # Subclasses set this to identify the provider in logs
    PROVIDER_NAME: str = "base"

    def __init__(self, model: str) -> None:
        from src.logger import get_logger

        self.model = model
        self.log = get_logger(f"src.ai.{self.PROVIDER_NAME}")
        # Cost tracker — accumulates token usage for the entire run
        self.cost_tracker = CostTracker()

    # ------------------------------------------------------------------
    # Abstract: must be implemented by each provider subclass
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> str:
        """
        Send a prompt and return the model's text response.

        Parameters
        ----------
        prompt:
            The user prompt text.
        max_tokens:
            Maximum number of tokens in the response.
        temperature:
            Sampling temperature (lower = more deterministic).
        system:
            Optional system message to prepend (not all providers support this).

        Returns
        -------
        str
            The model's response as a plain string (stripped of whitespace).

        Raises
        ------
        AIError
            On any API failure (network error, invalid key, quota exceeded).
        TokenLimitError
            If the prompt is too long for the model's context window.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete: shared across all providers
    # ------------------------------------------------------------------

    async def score_item(
        self,
        item: "NewsItem",
        user_interests: str,
        web_context: str = "",
    ) -> "ScoredItem":
        """
        Score a NewsItem for relevance using the AI provider.

        Builds a structured prompt from the item's metadata, calls
        ``complete()``, and parses the JSON response into a ScoredItem.

        Parameters
        ----------
        item:
            The NewsItem to score.
        user_interests:
            A short description of what the user wants to read about.
            Example: "AI, machine learning, Python, open source tools"
        web_context:
            Optional background context from DuckDuckGo (fetched by search.py).

        Returns
        -------
        ScoredItem
            The original item enriched with ai_score, ai_reason, and ai_topics.
            Falls back to score=5 (neutral) on parsing failures.
        """
        from src.models import ScoredItem

        prompt = self._build_score_prompt(item, user_interests, web_context)

        self.log.debug("Scoring: %s", item.title[:70])

        try:
            raw = await with_ai_retry(
                self.complete,
                prompt,
                max_tokens=256,
                temperature=0.2,
                cost_tracker=self.cost_tracker,
                model=self.model,
            )
            score, reason, topics = self._parse_score_response(raw, item)
        except Exception as e:
            self.log.warning("Score parse failed for '%s': %s", item.title[:50], e)
            score, reason, topics = 5, "Scoring unavailable", []

        return ScoredItem(
            item=item,
            ai_score=score,
            ai_score_reason=reason,
            ai_reason=reason,
            ai_topics=topics,
            model_used=self.model,
        )

    async def complete_safe(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        fallback: str = "",
    ) -> str:
        """
        Fault-tolerant wrapper around ``complete()`` with retry logic.

        Returns ``fallback`` instead of raising on errors.
        Logs the error at WARNING level.
        """
        try:
            return await with_ai_retry(
                self.complete,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                cost_tracker=self.cost_tracker,
                model=self.model,
            )
        except Exception as e:
            self.log.warning("AI call failed after retries (%s): %s", self.model, e)
            return fallback

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_score_prompt(
        self,
        item: "NewsItem",
        user_interests: str,
        web_context: str,
    ) -> str:
        """Build the scoring prompt string from item metadata."""
        from datetime import timezone

        published = (
            item.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if item.published_at
            else "unknown"
        )
        platform_score = str(item.score) if item.score is not None else "n/a"
        comments = str(item.comment_count) if item.comment_count is not None else "n/a"
        summary = (item.summary or "")[:300] or "No summary available"
        context = web_context[:300] if web_context else "No context available"

        return _SCORE_PROMPT_TEMPLATE.format(
            interests=user_interests,
            title=item.title,
            source=item.source_name,
            source_type=item.source_type,
            published=published,
            summary=summary,
            platform_score=platform_score,
            comments=comments,
            web_context=context,
        )

    @staticmethod
    def _parse_score_response(
        raw: str,
        item: "NewsItem",
    ) -> tuple[int, str, list[str]]:
        """
        Parse the AI's JSON score response.

        Returns (score, reason, topics). Falls back to (5, "...", []) on error.

        Handles:
          - Clean JSON: '{"score": 8, "reason": "...", "topics": [...]}'
          - JSON wrapped in markdown: '```json\n{...}\n```'
          - Missing fields (defaults used)
        """
        # Strip markdown code fences if present
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response: {text[:100]!r}")

        data = json.loads(match.group())

        score = int(data.get("score", 5))
        score = max(1, min(10, score))  # clamp to [1, 10]

        reason = str(data.get("reason", "")).strip() or "No reason provided"
        topics = [str(t).strip() for t in data.get("topics", []) if t]

        return score, reason, topics


# ---------------------------------------------------------------------------
# AI Provider Factory
# ---------------------------------------------------------------------------


class AIProviderFactory:
    """
    Factory that instantiates the correct BaseAIProvider based on model name.

    Model name prefixes determine the provider:
      "gpt-*"     → OpenAI
      "gemini-*"  → Google Gemini
      "claude-*"  → Anthropic
      "ollama-*"  → Ollama (local)
    """

    @staticmethod
    def from_model(model: str) -> "BaseAIProvider":
        """
        Create the appropriate AI provider adapter for the given model name.

        Parameters
        ----------
        model:
            Model identifier string (e.g. "gpt-4o-mini", "gemini-1.5-flash").

        Returns
        -------
        BaseAIProvider
            An initialized provider adapter for the model.

        Raises
        ------
        ValueError
            If the model prefix is not recognized.
        """
        model_lower = model.lower().strip()

        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            from src.ai.openai_adapter import OpenAIProvider
            return OpenAIProvider(model)

        if model_lower.startswith("gemini"):
            from src.ai.gemini_adapter import GeminiProvider
            return GeminiProvider(model)

        if model_lower.startswith("claude"):
            from src.ai.anthropic_adapter import AnthropicProvider
            return AnthropicProvider(model)

        raise ValueError(
            f"Unrecognized AI model prefix: '{model}'. "
            "Expected 'gpt-*', 'gemini-*', or 'claude-*'."
        )

    @staticmethod
    def from_settings() -> "BaseAIProvider":
        """
        Create the provider from the app's Settings configuration.

        Reads AI_MODEL from .env via Settings and returns the appropriate
        provider. Raises ConfigError if no matching API key is configured.
        """
        from src.config import Settings
        from src.exceptions import ConfigError

        s = Settings()
        model = s.ai_model

        model_lower = model.lower()
        if model_lower.startswith("gpt") and not s.has_openai:
            raise ConfigError(
                "AI_MODEL is set to a GPT model but OPENAI_API_KEY is not set.",
                field="OPENAI_API_KEY",
            )
        if model_lower.startswith("gemini") and not s.has_gemini:
            raise ConfigError(
                "AI_MODEL is set to a Gemini model but GEMINI_API_KEY is not set.",
                field="GEMINI_API_KEY",
            )
        if model_lower.startswith("claude") and not s.has_anthropic:
            raise ConfigError(
                "AI_MODEL is set to a Claude model but ANTHROPIC_API_KEY is not set.",
                field="ANTHROPIC_API_KEY",
            )

        return AIProviderFactory.from_model(model)
