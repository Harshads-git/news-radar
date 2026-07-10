"""
tests/test_scoring.py
======================
Tests for Day 21 scoring module (src/scoring/rubric.py + src/scoring/history.py).

Coverage:
  - recency_fraction: None → 0.5, fresh → 1.0, old → 0.0, mid-range interpolation,
    timezone-naive handling, exact boundary values
  - normalize_platform_score: None score → 0, zero max → 0, max → 1, sqrt compression,
    score > max caps at 1
  - ScoringRubric: defaults, custom weights, trust_for fallback/override,
    platform_max_for fallback, total_weight
  - CompositeScore.explain(): format, values in output
  - RubricScorer.compute(): ai_fraction correct, recency signal, trust signal,
    platform signal, composite in [0, 10], high-quality item beats low-quality
  - RubricScorer.adjust_batch(): returns pairs, correct length
  - ScoreHistory.append(): creates file, correct fields, no crash on bad dir
  - ScoreHistory.load(): empty on no file, reads entries, days cutoff filters old
  - ScoreHistory.source_stats(): empty on no history, mean/count correct, multi-source
  - ScoreHistory.topic_trend(): empty on no matches, daily grouping, partial topic match
  - ScoreHistory.top_sources(): returns top N, sorted by mean

All tests use tmp_path — no real file system side effects.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models import NewsItem, ScoredItem
from src.scoring.history import ScoreEntry, ScoreHistory
from src.scoring.rubric import (
    CompositeScore,
    RubricScorer,
    ScoringRubric,
    normalize_platform_score,
    recency_fraction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)


def _item(
    source_id: str = "test",
    source_type: str = "rss",
    score: int | None = None,
    published_at: datetime | None = None,
    title: str = "Test Title",
) -> NewsItem:
    return NewsItem(
        source_id=source_id,
        source_name="Test Source",
        source_type=source_type,
        title=title,
        url=f"https://example.com/{abs(hash(title))}",
        score=score,
        published_at=published_at,
    )


def _scored(ai_score: int = 7, **item_kwargs) -> ScoredItem:
    return ScoredItem(item=_item(**item_kwargs), ai_score=ai_score)


# Expose constants for boundary tests (use module-private names)
try:
    from src.scoring.rubric import _RECENCY_FULL_HOURS as _FULL_HOURS
    from src.scoring.rubric import _RECENCY_ZERO_HOURS as _ZERO_HOURS
except ImportError:
    _FULL_HOURS = 6.0
    _ZERO_HOURS = 48.0


# ===========================================================================
# recency_fraction
# ===========================================================================


class TestRecencyFraction:
    @pytest.mark.unit
    def test_none_returns_neutral(self):
        assert recency_fraction(None) == 0.5

    @pytest.mark.unit
    def test_fresh_story_returns_one(self):
        published = NOW - timedelta(hours=2)
        assert recency_fraction(published) == 1.0

    @pytest.mark.unit
    def test_old_story_returns_zero(self):
        published = NOW - timedelta(hours=60)
        assert recency_fraction(published) == 0.0

    @pytest.mark.unit
    def test_mid_range_between_zero_and_one(self):
        published = NOW - timedelta(hours=27)
        result = recency_fraction(published)
        assert 0.0 < result < 1.0

    @pytest.mark.unit
    def test_exactly_at_full_boundary_returns_one(self):
        published = NOW - timedelta(hours=_FULL_HOURS)
        result = recency_fraction(published)
        # At exactly the boundary, result should be ≥ 1.0 or very close
        # (slight floating-point difference from NOW vs datetime.now() in the function)
        assert result >= 0.999

    @pytest.mark.unit
    def test_exactly_at_zero_boundary_returns_zero(self):
        published = NOW - timedelta(hours=_ZERO_HOURS)
        assert recency_fraction(published) == 0.0

    @pytest.mark.unit
    def test_timezone_naive_handled_gracefully(self):
        # Naive datetime should not raise
        naive = datetime.now() - timedelta(hours=3)
        result = recency_fraction(naive)
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_monotone_decreasing_with_age(self):
        times = [NOW - timedelta(hours=h) for h in [1, 12, 24, 48]]
        fractions = [recency_fraction(t) for t in times]
        assert fractions == sorted(fractions, reverse=True)


# ===========================================================================
# normalize_platform_score
# ===========================================================================


class TestNormalizePlatformScore:
    @pytest.mark.unit
    def test_none_score_returns_zero(self):
        assert normalize_platform_score(None, 1000) == 0.0

    @pytest.mark.unit
    def test_zero_score_returns_zero(self):
        assert normalize_platform_score(0, 1000) == 0.0

    @pytest.mark.unit
    def test_max_score_returns_one(self):
        assert abs(normalize_platform_score(1000, 1000) - 1.0) < 1e-9

    @pytest.mark.unit
    def test_half_max_returns_sqrt_half(self):
        import math
        result = normalize_platform_score(500, 1000)
        assert abs(result - math.sqrt(0.5)) < 1e-9

    @pytest.mark.unit
    def test_score_above_max_capped_at_one(self):
        result = normalize_platform_score(5000, 1000)
        assert abs(result - 1.0) < 1e-9

    @pytest.mark.unit
    def test_zero_max_returns_zero(self):
        assert normalize_platform_score(100, 0) == 0.0

    @pytest.mark.unit
    def test_sqrt_compression_low_score(self):
        # sqrt compression: 100 / 1000 = 0.1, sqrt(0.1) ≈ 0.316
        import math
        result = normalize_platform_score(100, 1000)
        assert abs(result - math.sqrt(0.1)) < 1e-9


# ===========================================================================
# ScoringRubric
# ===========================================================================


class TestScoringRubric:
    @pytest.mark.unit
    def test_default_weights(self):
        r = ScoringRubric()
        assert r.ai_weight == 0.60
        assert r.recency_weight == 0.15
        assert r.source_trust_weight == 0.15
        assert r.platform_weight == 0.10

    @pytest.mark.unit
    def test_custom_weights_accepted(self):
        r = ScoringRubric(ai_weight=0.5, recency_weight=0.3)
        assert r.ai_weight == 0.5
        assert r.recency_weight == 0.3

    @pytest.mark.unit
    def test_trust_for_known_source(self):
        r = ScoringRubric(source_trust_map={"hackernews-top": 0.9})
        assert r.trust_for("hackernews-top") == 0.9

    @pytest.mark.unit
    def test_trust_for_unknown_falls_back_to_default(self):
        r = ScoringRubric(default_source_trust=0.6)
        assert r.trust_for("unknown-source") == 0.6

    @pytest.mark.unit
    def test_platform_max_for_known_type(self):
        r = ScoringRubric()
        assert r.platform_max_for("hackernews") == 1000.0

    @pytest.mark.unit
    def test_platform_max_for_unknown_type_returns_default(self):
        r = ScoringRubric()
        result = r.platform_max_for("unknown_type")
        assert result == 100.0

    @pytest.mark.unit
    def test_total_weight_sums_all_factors(self):
        r = ScoringRubric(ai_weight=0.5, recency_weight=0.2, source_trust_weight=0.2, platform_weight=0.1)
        assert abs(r.total_weight - 1.0) < 1e-9


# ===========================================================================
# CompositeScore.explain
# ===========================================================================


class TestCompositeScoreExplain:
    @pytest.mark.unit
    def test_explain_contains_score(self):
        cs = CompositeScore(
            ai_fraction=0.8, recency_fraction=1.0, trust_fraction=0.9,
            platform_fraction=0.7, composite_0_1=0.85, final_score=8.5,
        )
        assert "8.5" in cs.explain()

    @pytest.mark.unit
    def test_explain_contains_all_factor_names(self):
        cs = CompositeScore(
            ai_fraction=0.8, recency_fraction=1.0, trust_fraction=0.9,
            platform_fraction=0.7, composite_0_1=0.85, final_score=8.5,
        )
        explanation = cs.explain()
        assert "ai=" in explanation
        assert "recency=" in explanation
        assert "trust=" in explanation
        assert "platform=" in explanation

    @pytest.mark.unit
    def test_explain_returns_string(self):
        cs = CompositeScore(
            ai_fraction=0.5, recency_fraction=0.5, trust_fraction=0.5,
            platform_fraction=0.5, composite_0_1=0.5, final_score=5.0,
        )
        assert isinstance(cs.explain(), str)


# ===========================================================================
# RubricScorer.compute
# ===========================================================================


class TestRubricScorerCompute:
    @pytest.mark.unit
    def test_ai_fraction_correctly_normalized(self):
        scorer = RubricScorer()
        scored = _scored(ai_score=8)
        cs = scorer.compute(scored)
        assert cs.ai_fraction == 0.8

    @pytest.mark.unit
    def test_composite_score_in_0_to_10(self):
        scorer = RubricScorer()
        for ai_score in [0, 3, 5, 7, 10]:
            cs = scorer.compute(_scored(ai_score=ai_score))
            assert 0 <= cs.final_score <= 10, f"final_score={cs.final_score} out of range for ai_score={ai_score}"

    @pytest.mark.unit
    def test_fresh_story_gets_higher_score_than_old(self):
        scorer = RubricScorer()
        fresh = _scored(ai_score=7, published_at=NOW - timedelta(hours=2))
        old = _scored(ai_score=7, published_at=NOW - timedelta(hours=60))
        assert scorer.compute(fresh).final_score > scorer.compute(old).final_score

    @pytest.mark.unit
    def test_high_trust_source_scores_higher(self):
        rubric = ScoringRubric(
            source_trust_map={"hn-top": 0.95, "low-trust": 0.1}
        )
        scorer = RubricScorer(rubric)
        high_trust = _scored(ai_score=7, source_id="hn-top")
        low_trust = _scored(ai_score=7, source_id="low-trust")
        assert scorer.compute(high_trust).final_score > scorer.compute(low_trust).final_score

    @pytest.mark.unit
    def test_high_platform_score_boosts_final(self):
        rubric = ScoringRubric(platform_weight=0.3)
        scorer = RubricScorer(rubric)
        viral = _scored(ai_score=7, source_type="hackernews", score=900)
        unpopular = _scored(ai_score=7, source_type="hackernews", score=0)
        assert scorer.compute(viral).final_score > scorer.compute(unpopular).final_score

    @pytest.mark.unit
    def test_ai_score_0_gives_low_composite(self):
        scorer = RubricScorer()
        cs = scorer.compute(_scored(ai_score=0))
        assert cs.final_score < 5

    @pytest.mark.unit
    def test_ai_score_10_gives_high_composite(self):
        rubric = ScoringRubric()
        scorer = RubricScorer(rubric)
        cs = scorer.compute(_scored(ai_score=10, published_at=NOW - timedelta(hours=1)))
        assert cs.final_score > 7

    @pytest.mark.unit
    def test_recency_fraction_is_one_for_very_fresh(self):
        scorer = RubricScorer()
        cs = scorer.compute(_scored(published_at=NOW - timedelta(hours=1)))
        assert cs.recency_fraction == 1.0


# ===========================================================================
# RubricScorer.adjust_batch
# ===========================================================================


class TestRubricScorerBatch:
    @pytest.mark.unit
    def test_returns_correct_number_of_pairs(self):
        scorer = RubricScorer()
        items = [_scored(ai_score=i) for i in [5, 6, 7]]
        pairs = scorer.adjust_batch(items)
        assert len(pairs) == 3

    @pytest.mark.unit
    def test_each_pair_has_scored_item_and_composite(self):
        scorer = RubricScorer()
        pairs = scorer.adjust_batch([_scored(ai_score=8)])
        s, cs = pairs[0]
        assert isinstance(s, ScoredItem)
        assert isinstance(cs, CompositeScore)

    @pytest.mark.unit
    def test_empty_batch_returns_empty(self):
        scorer = RubricScorer()
        assert scorer.adjust_batch([]) == []


# ===========================================================================
# ScoreHistory
# ===========================================================================


class TestScoreHistoryAppend:
    @pytest.mark.unit
    def test_creates_history_file(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(_scored(ai_score=8), composite_score=7.5, run_date="2026-07-10")
        assert h.path.exists()

    @pytest.mark.unit
    def test_written_line_is_valid_json(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(_scored(ai_score=8), composite_score=7.5, run_date="2026-07-10")
        lines = h.path.read_text().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["ai_score"] == 8
        assert d["composite_score"] == 7.5

    @pytest.mark.unit
    def test_multiple_appends_add_multiple_lines(self, tmp_path):
        h = ScoreHistory(tmp_path)
        for i in range(3):
            h.append(_scored(ai_score=i + 5), composite_score=float(i), run_date="2026-07-10")
        lines = h.path.read_text().splitlines()
        assert len(lines) == 3

    @pytest.mark.unit
    def test_does_not_crash_on_unwritable_path(self, tmp_path):
        # Should silently swallow errors
        h = ScoreHistory(tmp_path / "nonexistent" / "deep" / "path")
        # This may or may not create dirs; test that no exception is raised
        try:
            h.append(_scored(ai_score=5), composite_score=5.0)
        except Exception as e:
            pytest.fail(f"append raised an exception: {e}")


class TestScoreHistoryLoad:
    @pytest.mark.unit
    def test_returns_empty_when_no_file(self, tmp_path):
        h = ScoreHistory(tmp_path)
        assert h.load() == []

    @pytest.mark.unit
    def test_loads_written_entries(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(_scored(ai_score=7), composite_score=6.5, run_date="2026-07-10")
        entries = h.load()
        assert len(entries) == 1
        assert entries[0].ai_score == 7
        assert entries[0].composite_score == 6.5

    @pytest.mark.unit
    def test_days_cutoff_filters_old_entries(self, tmp_path):
        h = ScoreHistory(tmp_path)
        # Write one old entry manually
        old_entry = {
            "run_date": "2020-01-01", "ts": "T", "item_id": "x",
            "title": "old", "url": "u", "source_id": "s", "source_type": "rss",
            "ai_score": 5, "composite_score": 5.0, "topics": [], "published_hours_ago": None,
        }
        h.path.parent.mkdir(parents=True, exist_ok=True)
        with h.path.open("a") as f:
            f.write(json.dumps(old_entry) + "\n")
        h.append(_scored(ai_score=8), composite_score=7.0, run_date="2026-07-10")

        # load with days=7 should only return recent entry
        recent = h.load(days=7)
        assert len(recent) == 1
        assert recent[0].ai_score == 8

    @pytest.mark.unit
    def test_malformed_lines_skipped(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.path.parent.mkdir(parents=True, exist_ok=True)
        h.path.write_text("NOT JSON\n")
        entries = h.load()
        assert entries == []


class TestScoreHistorySourceStats:
    @pytest.mark.unit
    def test_empty_history_returns_empty(self, tmp_path):
        h = ScoreHistory(tmp_path)
        assert h.source_stats() == {}

    @pytest.mark.unit
    def test_single_source_mean_correct(self, tmp_path):
        h = ScoreHistory(tmp_path)
        for score in [6.0, 8.0]:
            h.append(
                _scored(ai_score=7, source_id="hn-top"),
                composite_score=score,
                run_date="2026-07-10",
            )
        stats = h.source_stats()
        assert "hn-top" in stats
        assert stats["hn-top"]["mean"] == 7.0
        assert stats["hn-top"]["count"] == 2.0

    @pytest.mark.unit
    def test_multiple_sources_tracked_separately(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(_scored(source_id="hn-top"), composite_score=8.0, run_date="2026-07-10")
        h.append(_scored(source_id="r-prog"), composite_score=5.0, run_date="2026-07-10")
        stats = h.source_stats()
        assert "hn-top" in stats
        assert "r-prog" in stats
        assert stats["hn-top"]["mean"] != stats["r-prog"]["mean"]


class TestScoreHistoryTopicTrend:
    @pytest.mark.unit
    def test_empty_on_no_matching_topic(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(
            ScoredItem(item=_item(), ai_score=7, ai_topics=["Python"]),
            composite_score=7.0,
            run_date="2026-07-10",
        )
        result = h.topic_trend("Rust", days=7)
        assert result == {}

    @pytest.mark.unit
    def test_matching_topic_returns_daily_average(self, tmp_path):
        h = ScoreHistory(tmp_path)
        scored = ScoredItem(item=_item(), ai_score=8, ai_topics=["AI", "OpenAI"])
        h.append(scored, composite_score=8.0, run_date="2026-07-10")
        h.append(scored, composite_score=6.0, run_date="2026-07-10")
        result = h.topic_trend("AI", days=30)
        assert "2026-07-10" in result
        assert result["2026-07-10"] == 7.0

    @pytest.mark.unit
    def test_topic_match_is_case_insensitive(self, tmp_path):
        h = ScoreHistory(tmp_path)
        scored = ScoredItem(item=_item(), ai_score=7, ai_topics=["Python"])
        h.append(scored, composite_score=7.0, run_date="2026-07-10")
        assert "2026-07-10" in h.topic_trend("python", days=30)
        assert "2026-07-10" in h.topic_trend("PYTHON", days=30)


class TestScoreHistoryTopSources:
    @pytest.mark.unit
    def test_top_sources_sorted_by_mean(self, tmp_path):
        h = ScoreHistory(tmp_path)
        h.append(_scored(source_id="a"), composite_score=9.0, run_date="2026-07-10")
        h.append(_scored(source_id="b"), composite_score=6.0, run_date="2026-07-10")
        h.append(_scored(source_id="c"), composite_score=7.5, run_date="2026-07-10")
        top = h.top_sources(n=3)
        assert top[0][0] == "a"
        assert top[-1][0] == "b"

    @pytest.mark.unit
    def test_top_sources_limited_to_n(self, tmp_path):
        h = ScoreHistory(tmp_path)
        for i, src in enumerate(["a", "b", "c", "d", "e"]):
            h.append(_scored(source_id=src), composite_score=float(i), run_date="2026-07-10")
        top = h.top_sources(n=3)
        assert len(top) == 3
