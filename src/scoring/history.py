"""
src/scoring/history.py
========================
Lightweight score history persistence for News Radar.

Problem: Without history, we can't answer:
  - "Has this topic been trending up over the last 7 days?"
  - "Is this story's AI score unusually high compared to similar stories?"
  - "Which sources consistently produce high-scoring stories?"

This module persists per-run scoring results to data/score_history.jsonl
(one JSON line per scored item per run). The format is append-only so:
  - No lock needed for a single pipeline process
  - Old data is automatically preserved
  - The file can be analysed with jq or pandas

Why JSONL instead of SQLite?
  - Zero dependencies: just json + pathlib
  - Human-readable: `tail -n 5 data/score_history.jsonl | jq .`
  - Portable: trivially importable into any analytics tool
  - For a news aggregator running once/day, the file grows ~50 lines/day
    (one per scored item). At 365 days, that's ~18,000 lines — trivial.

Day 21 features:
  - ScoreHistory.append(): persist one ScoredItem + composite score per run
  - ScoreHistory.load(): read all historical entries
  - ScoreHistory.source_stats(): mean/std of scores per source_id
  - ScoreHistory.topic_trend(): average score per topic over N days

Usage:
    from src.scoring.history import ScoreHistory
    from src.config import settings

    history = ScoreHistory(settings.data_dir)
    history.append(scored_item, composite_score=7.4, run_date="2026-07-10")

    stats = history.source_stats()
    print(stats["hackernews-top"])  # {"mean": 7.2, "count": 45}
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ScoreEntry — one persisted record
# ---------------------------------------------------------------------------


class ScoreEntry:
    """
    One persisted scoring record (not a Pydantic model to avoid import cycle).

    Stored as a flat JSON dict:
    {
      "run_date": "2026-07-10",
      "ts": "2026-07-10T11:05:00Z",
      "item_id": "abc123",
      "title": "OpenAI Releases GPT-5",
      "url": "https://...",
      "source_id": "hackernews-top",
      "source_type": "hackernews",
      "ai_score": 8,
      "composite_score": 7.4,
      "topics": ["AI", "OpenAI"],
      "published_hours_ago": 3.2
    }
    """

    __slots__ = (
        "run_date", "ts", "item_id", "title", "url",
        "source_id", "source_type", "ai_score", "composite_score",
        "topics", "published_hours_ago",
    )

    def __init__(
        self,
        *,
        run_date: str,
        ts: str,
        item_id: str,
        title: str,
        url: str,
        source_id: str,
        source_type: str,
        ai_score: int,
        composite_score: float,
        topics: list[str],
        published_hours_ago: float | None,
    ) -> None:
        self.run_date = run_date
        self.ts = ts
        self.item_id = item_id
        self.title = title
        self.url = url
        self.source_id = source_id
        self.source_type = source_type
        self.ai_score = ai_score
        self.composite_score = composite_score
        self.topics = topics
        self.published_hours_ago = published_hours_ago

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "ts": self.ts,
            "item_id": self.item_id,
            "title": self.title,
            "url": self.url,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "ai_score": self.ai_score,
            "composite_score": self.composite_score,
            "topics": self.topics,
            "published_hours_ago": self.published_hours_ago,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoreEntry:
        return cls(
            run_date=d.get("run_date", ""),
            ts=d.get("ts", ""),
            item_id=d.get("item_id", ""),
            title=d.get("title", ""),
            url=d.get("url", ""),
            source_id=d.get("source_id", ""),
            source_type=d.get("source_type", ""),
            ai_score=int(d.get("ai_score", 0)),
            composite_score=float(d.get("composite_score", 0.0)),
            topics=list(d.get("topics", [])),
            published_hours_ago=d.get("published_hours_ago"),
        )


# ---------------------------------------------------------------------------
# ScoreHistory
# ---------------------------------------------------------------------------


class ScoreHistory:
    """
    Append-only JSONL store for per-run scoring history.

    Thread-safety note: Python's file open-append-close is atomic for writes
    smaller than PIPE_BUF (~4 KB) on POSIX. For a single-process pipeline
    running once per day, this is sufficient.
    """

    DEFAULT_FILENAME = "score_history.jsonl"

    def __init__(self, data_dir: Path | str, filename: str = DEFAULT_FILENAME) -> None:
        self.data_dir = Path(data_dir)
        self._path = self.data_dir / filename

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        scored_item: object,  # ScoredItem — typed as object to avoid import cycle
        *,
        composite_score: float,
        run_date: str | None = None,
    ) -> None:
        """
        Persist one ScoredItem's scoring result to the history file.

        Parameters
        ----------
        scored_item:
            A ScoredItem from the pipeline's AI scoring stage.
        composite_score:
            The final composite score (0-10) from RubricScorer.
        run_date:
            ISO date string for the run (e.g. '2026-07-10').
            Defaults to today.
        """
        run_date = run_date or date.today().isoformat()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        item = scored_item.item  # type: ignore[union-attr]

        # Compute published_hours_ago
        pub_hours: float | None = None
        if item.published_at is not None:
            pub_at = item.published_at
            if pub_at.tzinfo is None:
                pub_at = pub_at.replace(tzinfo=timezone.utc)
            pub_hours = round(
                (datetime.now(timezone.utc) - pub_at).total_seconds() / 3600.0, 1
            )

        entry = ScoreEntry(
            run_date=run_date,
            ts=ts,
            item_id=str(item.id),
            title=item.title,
            url=item.url,
            source_id=item.source_id,
            source_type=item.source_type,
            ai_score=int(scored_item.ai_score),  # type: ignore[union-attr]
            composite_score=round(composite_score, 2),
            topics=list(getattr(scored_item, "ai_topics", []) or []),
            published_hours_ago=pub_hours,
        )

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # Never crash the pipeline due to history write failure

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, *, days: int | None = None) -> list[ScoreEntry]:
        """
        Load all (or the most recent N days of) scoring entries.

        Parameters
        ----------
        days:
            If provided, only entries from the last N days are returned.
            None returns all entries.

        Returns
        -------
        list[ScoreEntry]
            Entries in chronological order (oldest first).
        """
        if not self._path.exists():
            return []

        cutoff: str | None = None
        if days is not None:
            from datetime import timedelta
            cutoff = (date.today() - timedelta(days=days)).isoformat()

        entries: list[ScoreEntry] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if cutoff and d.get("run_date", "") < cutoff:
                    continue
                entries.append(ScoreEntry.from_dict(d))
            except (json.JSONDecodeError, KeyError):
                pass  # Skip malformed lines

        return entries

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def source_stats(self, *, days: int | None = 30) -> dict[str, dict[str, float]]:
        """
        Compute mean composite score and count for each source_id.

        Parameters
        ----------
        days:
            Only consider entries from the last N days. Default: 30.

        Returns
        -------
        dict mapping source_id → {"mean": float, "std": float, "count": int}
        """
        entries = self.load(days=days)

        # Group by source_id
        groups: dict[str, list[float]] = {}
        for e in entries:
            groups.setdefault(e.source_id, []).append(e.composite_score)

        result: dict[str, dict[str, float]] = {}
        for source_id, scores in groups.items():
            n = len(scores)
            mean = sum(scores) / n
            variance = sum((s - mean) ** 2 for s in scores) / n if n > 1 else 0.0
            result[source_id] = {
                "mean": round(mean, 2),
                "std": round(math.sqrt(variance), 2),
                "count": float(n),
            }

        return result

    def topic_trend(
        self,
        topic: str,
        *,
        days: int = 7,
    ) -> dict[str, float]:
        """
        Compute the average composite score per day for a given topic.

        Parameters
        ----------
        topic:
            The topic label to filter on (case-insensitive, partial match).
        days:
            Number of days to look back. Default: 7.

        Returns
        -------
        dict mapping ISO date string → average composite score for that day.
        An empty dict if no matching entries are found.

        Why per-day average?
        Trending detection: if a topic's average score increases over
        consecutive days, it's gaining relevance in the corpus.
        """
        entries = self.load(days=days)
        topic_lower = topic.lower()

        day_scores: dict[str, list[float]] = {}
        for e in entries:
            if any(topic_lower in t.lower() for t in e.topics):
                day_scores.setdefault(e.run_date, []).append(e.composite_score)

        return {
            day: round(sum(scores) / len(scores), 2)
            for day, scores in sorted(day_scores.items())
        }

    def top_sources(self, *, days: int = 30, n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top N sources by mean composite score.

        Returns
        -------
        list of (source_id, mean_score) tuples, sorted highest first.
        """
        stats = self.source_stats(days=days)
        ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        return [(src, info["mean"]) for src, info in ranked[:n]]
