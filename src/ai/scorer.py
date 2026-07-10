"""
src/ai/scorer.py
================
Batch news scorer — scores all fetched NewsItems in a pipeline run.

The scorer orchestrates:
  1. For each NewsItem, fetch background web context (via search.py)
  2. Call provider.score_item(item, interests, web_context)
  3. Optionally apply multi-factor rubric re-ranking (Day 21)
  4. Persist scores to history (Day 21)
  5. Filter items below the score threshold
  6. Return ScoredItem list sorted by composite score descending

Concurrency: uses asyncio.gather() with a semaphore to limit simultaneous
AI API calls. Too many concurrent calls may hit rate limits or inflate costs.

Design: NewsScorer is stateless — create one per pipeline run with the
provider and settings wired in by the orchestrator.

Usage:
    from src.ai.scorer import NewsScorer
    from src.ai import AIProviderFactory
    from src.scoring import ScoringRubric, RubricScorer, ScoreHistory

    provider = AIProviderFactory.from_model("gpt-4o-mini")
    rubric = ScoringRubric(recency_weight=0.20)
    scorer = NewsScorer(provider, settings, rubric=RubricScorer(rubric))
    scored_items = await scorer.score_all(news_items)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.ai.base import BaseAIProvider
    from src.config import Settings
    from src.models import NewsItem, ScoredItem
    from src.scoring.rubric import RubricScorer
    from src.scoring.history import ScoreHistory

log = get_logger(__name__)

# Maximum concurrent AI calls to avoid rate limits
_DEFAULT_CONCURRENCY = 5


class NewsScorer:
    """
    Batch AI scorer for a list of NewsItems.

    Wraps the AI provider and orchestrates web context fetching,
    concurrent scoring, optional rubric re-ranking, score persistence,
    filtering, and sorting.
    """

    def __init__(
        self,
        provider: "BaseAIProvider",
        settings: "Settings",
        concurrency: int = _DEFAULT_CONCURRENCY,
        rubric: "RubricScorer | None" = None,
        history: "ScoreHistory | None" = None,
    ) -> None:
        """
        Parameters
        ----------
        provider:
            An initialized AI provider adapter (OpenAI, Gemini, etc.)
        settings:
            App settings (used for score_threshold, user_interests).
        concurrency:
            Max simultaneous AI calls. Default: 5.
        rubric:
            Optional RubricScorer for multi-factor composite score adjustment.
            If None, raw AI scores are used for ranking.
        history:
            Optional ScoreHistory for persisting scores to JSONL.
            If None, scores are not persisted.
        """
        self.provider = provider
        self.settings = settings
        self.concurrency = concurrency
        self.rubric = rubric
        self.history = history
        self._semaphore = asyncio.Semaphore(concurrency)

    async def score_all(
        self,
        items: list["NewsItem"],
        *,
        fetch_context: bool = True,
        run_date: str | None = None,
    ) -> list["ScoredItem"]:
        """
        Score all items concurrently and return filtered + sorted results.

        Parameters
        ----------
        items:
            List of NewsItems to score (from all scrapers, already deduped).
        fetch_context:
            If True, fetches DuckDuckGo background context per item before scoring.
            Set False to skip context fetching (e.g. for tests or dry runs).
        run_date:
            ISO date string for score history records. Defaults to today.

        Returns
        -------
        list[ScoredItem]
            Items with score >= settings.score_threshold, sorted
            highest composite score first. At most settings.max_briefing_items returned.
        """
        if not items:
            return []

        log.section("Phase 2: AI Scoring")
        log.info("Scoring %d items (threshold=%d)", len(items), self.settings.score_threshold)

        # Score all items concurrently
        scored = await asyncio.gather(
            *[self._score_one(item, fetch_context=fetch_context) for item in items]
        )

        # Apply rubric re-ranking if configured
        if self.rubric is not None:
            composite_map = {}
            pairs = self.rubric.adjust_batch(list(scored))
            for s, cs in pairs:
                composite_map[id(s)] = cs

            # Persist to history if configured
            if self.history is not None:
                for s, cs in pairs:
                    try:
                        self.history.append(
                            s,
                            composite_score=cs.final_score,
                            run_date=run_date,
                        )
                    except Exception as e:
                        log.debug("Could not write score history: %s", e)

            # Use composite score for filtering and sorting
            threshold = self.settings.score_threshold
            passed = [
                s for s in scored
                if composite_map[id(s)].final_score >= threshold
            ]
            passed.sort(
                key=lambda s: (
                    composite_map[id(s)].final_score,
                    s.item.published_at or "",
                ),
                reverse=True,
            )
            log.info(
                "Rubric re-ranking applied: %d/%d items passed threshold",
                len(passed), len(scored),
            )
        else:
            # Fallback: filter by raw AI score (original behaviour)
            threshold = self.settings.score_threshold
            passed = [s for s in scored if s.ai_score >= threshold]

            # Sort: highest score first, then by published_at (newest first)
            passed.sort(
                key=lambda s: (s.ai_score, s.item.published_at or ""),
                reverse=True,
            )

        # Trim to max items
        result = passed[: self.settings.max_briefing_items]

        log.info(
            "Scoring done: %d/%d items passed threshold (top %d selected)",
            len(passed),
            len(items),
            len(result),
        )

        return result

    async def _score_one(
        self,
        item: "NewsItem",
        *,
        fetch_context: bool = True,
    ) -> "ScoredItem":
        """Score a single item under the concurrency semaphore."""
        async with self._semaphore:
            web_context = ""
            if fetch_context:
                try:
                    from src.search import fetch_web_context
                    web_context = await fetch_web_context(item.title)
                except Exception:
                    pass  # Context is best-effort; never block scoring

            return await self.provider.score_item(
                item,
                self.settings.user_interests,
                web_context=web_context,
            )

    async def score_single(
        self,
        item: "NewsItem",
        web_context: str = "",
    ) -> "ScoredItem":
        """
        Score a single item directly (useful for testing and debugging).

        Does not apply threshold filtering — returns the ScoredItem as-is.
        """
        return await self.provider.score_item(
            item,
            self.settings.user_interests,
            web_context=web_context,
        )
