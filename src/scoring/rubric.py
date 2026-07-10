"""
src/scoring/rubric.py
======================
Multi-factor scoring rubric for News Radar.

Problem: Raw AI scores (0-10) alone don't capture everything that makes a
story valuable. A story might score 7/10 on AI relevance but was published
4 days ago (less urgent), or come from a source that historically pumps
clickbait (lower trust), or have 2000 HN points (very high engagement).

This module provides a configurable rubric that adjusts the raw AI score
into a composite score that factors in:

  1. AI Relevance Score (weight: ai_weight)
     The raw 0-10 AI assessment of how relevant/interesting this story is.

  2. Recency Bonus (weight: recency_weight)
     Stories published in the last 6 hours get the full bonus;
     stories older than 48 hours get 0 bonus. Linear interpolation.

  3. Source Trust Score (weight: source_trust_weight)
     Configurable per-source trust rating (0.0–1.0). High-trust sources
     (HN, ArXiv) get boosted; low-trust sources get penalized.
     Default: 0.7 (slight positive prior for all configured sources).

  4. Platform Engagement Score (weight: platform_weight)
     Normalizes the raw platform score (HN points, Reddit upvotes, stars)
     relative to the typical maximum for that source type.

Why composite scoring?
  - A 6/10 story from ArXiv published 2 hours ago beats a 7/10 clickbait
    article from a low-trust blog published yesterday.
  - Platform engagement is a crowd-sourced quality signal that's cheap
    to compute (no extra AI call needed).

The final composite score is rescaled back to the 0-10 range so it
remains compatible with the existing score_threshold setting.

Usage:
    from src.scoring.rubric import ScoringRubric, RubricScorer
    from src.models import ScoredItem

    rubric = ScoringRubric(recency_weight=0.15)
    scorer = RubricScorer(rubric)
    adjusted = scorer.adjust(scored_item, source_trust=0.9)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import ScoredItem


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Typical maximum platform scores for normalization
# (used when no explicit max is provided)
_DEFAULT_PLATFORM_MAXES: dict[str, float] = {
    "hackernews": 1000.0,
    "reddit": 5000.0,
    "rss": 100.0,   # RSS rarely has engagement scores
    "github": 500.0,
}

# Recency buckets (hours → bonus fraction 0-1)
_RECENCY_FULL_HOURS = 6.0    # ≤ 6h → full recency bonus
_RECENCY_ZERO_HOURS = 48.0   # ≥ 48h → zero recency bonus


# ---------------------------------------------------------------------------
# ScoringRubric — configurable weight settings
# ---------------------------------------------------------------------------


@dataclass
class ScoringRubric:
    """
    Weights and parameters for multi-factor composite scoring.

    All weights should be non-negative and ideally sum to ~1.0, though
    the scorer normalizes the final result to [0, 10] regardless.

    Attributes
    ----------
    ai_weight:
        Weight for the raw AI relevance score. Default: 0.60
        Higher values make the AI assessment dominant.

    recency_weight:
        Weight for the recency bonus. Default: 0.15
        Stories published within the last 6 hours get the full bonus.

    source_trust_weight:
        Weight for per-source trust score. Default: 0.15
        Trust scores are 0.0–1.0 (default: 0.7 for all sources).

    platform_weight:
        Weight for normalized platform engagement. Default: 0.10
        HN points, Reddit upvotes, GitHub stars.

    default_source_trust:
        Trust score assigned to sources not listed in source_trust_map.
        Default: 0.7 (slight positive prior).

    source_trust_map:
        Dict mapping source_id → trust score (0.0–1.0).
        Example: {"hackernews-top": 0.9, "r-programming": 0.8}

    platform_max_map:
        Dict mapping source_type → typical max engagement score.
        Used to normalize raw platform scores into [0, 1].
    """

    ai_weight: float = 0.60
    recency_weight: float = 0.15
    source_trust_weight: float = 0.15
    platform_weight: float = 0.10

    default_source_trust: float = 0.7

    source_trust_map: dict[str, float] = field(default_factory=dict)
    platform_max_map: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PLATFORM_MAXES)
    )

    def trust_for(self, source_id: str) -> float:
        """Return the trust score for a source, falling back to default."""
        return self.source_trust_map.get(source_id, self.default_source_trust)

    def platform_max_for(self, source_type: str) -> float:
        """Return the normalization ceiling for a source type."""
        return self.platform_max_map.get(source_type, 100.0)

    @property
    def total_weight(self) -> float:
        """Sum of all component weights (for diagnostics)."""
        return self.ai_weight + self.recency_weight + self.source_trust_weight + self.platform_weight


# ---------------------------------------------------------------------------
# Recency helpers
# ---------------------------------------------------------------------------


def recency_fraction(published_at: datetime | None) -> float:
    """
    Compute the recency bonus fraction for a story (0.0 to 1.0).

    Returns 1.0 for stories published within _RECENCY_FULL_HOURS (6h),
    0.0 for stories older than _RECENCY_ZERO_HOURS (48h),
    and linear interpolation in between.

    Returns 0.5 (neutral) if published_at is None.
    """
    if published_at is None:
        return 0.5

    # Make timezone-aware if needed
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)

    age_hours = (datetime.now(UTC) - published_at).total_seconds() / 3600.0

    if age_hours <= _RECENCY_FULL_HOURS:
        return 1.0
    if age_hours >= _RECENCY_ZERO_HOURS:
        return 0.0
    # Linear interpolation from full → zero
    span = _RECENCY_ZERO_HOURS - _RECENCY_FULL_HOURS
    return 1.0 - (age_hours - _RECENCY_FULL_HOURS) / span


def normalize_platform_score(score: int | None, max_score: float) -> float:
    """
    Normalize a raw platform score (HN points, Reddit upvotes) to [0, 1].

    Uses a sqrt compression so very high scores don't dominate.
    E.g., 1000 HN points with max=1000 → sqrt(1.0) = 1.0
         100 HN points → sqrt(0.1) ≈ 0.316

    Returns 0.0 if score is None or max_score ≤ 0.
    """
    if score is None or max_score <= 0:
        return 0.0
    ratio = min(score / max_score, 1.0)  # cap at 1.0
    return math.sqrt(ratio)


# ---------------------------------------------------------------------------
# CompositeScore — the breakdown for one item
# ---------------------------------------------------------------------------


@dataclass
class CompositeScore:
    """
    Breakdown of how the composite score was calculated for one item.

    Useful for debugging, logging, and explaining why a story was ranked.
    """

    ai_fraction: float        # raw AI score normalized to [0, 1]
    recency_fraction: float   # recency bonus [0, 1]
    trust_fraction: float     # source trust [0, 1]
    platform_fraction: float  # normalized platform engagement [0, 1]
    composite_0_1: float      # weighted sum, in [0, 1]
    final_score: float        # rescaled to [0, 10] (2 decimal places)

    def explain(self) -> str:
        """Return a human-readable one-line explanation of the score."""
        return (
            f"score={self.final_score:.1f} "
            f"(ai={self.ai_fraction:.2f}, "
            f"recency={self.recency_fraction:.2f}, "
            f"trust={self.trust_fraction:.2f}, "
            f"platform={self.platform_fraction:.2f})"
        )


# ---------------------------------------------------------------------------
# RubricScorer — the main adjustment engine
# ---------------------------------------------------------------------------


class RubricScorer:
    """
    Adjusts raw AI ScoredItems using a multi-factor rubric.

    Usage
    -----
    ::

        rubric = ScoringRubric(recency_weight=0.20, source_trust_weight=0.10)
        scorer = RubricScorer(rubric)
        composite = scorer.compute(scored_item)
        print(composite.final_score)   # e.g. 7.4
        print(composite.explain())
    """

    def __init__(self, rubric: ScoringRubric | None = None) -> None:
        self.rubric = rubric or ScoringRubric()

    def compute(self, scored_item: "ScoredItem") -> CompositeScore:
        """
        Compute the composite score for a ScoredItem.

        Parameters
        ----------
        scored_item:
            A ScoredItem from the AI scoring stage.

        Returns
        -------
        CompositeScore
            Breakdown of each factor and the final composite score.
        """
        r = self.rubric
        item = scored_item.item

        # 1. AI fraction (already 0-10, normalize to 0-1)
        ai_f = scored_item.ai_score / 10.0

        # 2. Recency fraction
        rec_f = recency_fraction(item.published_at)

        # 3. Trust fraction (per source_id)
        trust_f = r.trust_for(item.source_id)

        # 4. Platform engagement fraction
        max_score = r.platform_max_for(item.source_type)
        plat_f = normalize_platform_score(item.score, max_score)

        # 5. Weighted sum → [0, 1]
        total_w = r.total_weight if r.total_weight > 0 else 1.0
        composite = (
            r.ai_weight * ai_f
            + r.recency_weight * rec_f
            + r.source_trust_weight * trust_f
            + r.platform_weight * plat_f
        ) / total_w

        # 6. Rescale to [0, 10]
        final = round(composite * 10.0, 2)

        return CompositeScore(
            ai_fraction=ai_f,
            recency_fraction=rec_f,
            trust_fraction=trust_f,
            platform_fraction=plat_f,
            composite_0_1=composite,
            final_score=final,
        )

    def adjust_batch(
        self,
        scored_items: "list[ScoredItem]",
    ) -> "list[tuple[ScoredItem, CompositeScore]]":
        """
        Compute composite scores for all items and return (item, composite) pairs.

        The caller can then sort by composite.final_score to re-rank items
        beyond what the raw AI score would produce.
        """
        return [(item, self.compute(item)) for item in scored_items]
