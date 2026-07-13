"""
src/pipeline/source_health.py
==============================
Per-source fetch health tracking with JSONL persistence.

Every time a scraper fetches a source we record whether it succeeded or
failed. The tracker maintains per-source counters in memory and persists
them to ``data/source_health.jsonl`` (one summary line per source per
day) so the ``--source-stats`` CLI command can show historical health.

Design goals:
  - Zero dependencies beyond stdlib + existing project modules
  - Thread-safe: all state mutations are guarded by a simple lock so the
    async pipeline can update multiple sources concurrently
  - Auto-disable hint: ``should_disable(source_id)`` returns True when a
    source has ≥ CONSECUTIVE_ERROR_THRESHOLD consecutive fetch failures,
    giving the orchestrator a signal to skip the source and warn the user

Persistence format (JSONL — one JSON object per line per write):
  {
    "date":               "2026-07-13",
    "source_id":          "hn-api-top",
    "attempts":           5,
    "successes":          4,
    "errors":             1,
    "last_item_count":    28,
    "consecutive_errors": 0,
    "last_error":         null,
    "written_at":         "2026-07-13T14:00:00Z"
  }

Usage:
    tracker = SourceHealthTracker(data_dir)
    tracker.record_success("hn-top", item_count=28)
    tracker.record_error("hn-rss", error="Connection timeout")
    tracker.flush()                            # write today's summary row

    # Check before running
    if tracker.should_disable("hn-rss"):
        log.warning("hn-rss disabled: too many consecutive errors")
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

# After this many consecutive errors a source is flagged for auto-disable
CONSECUTIVE_ERROR_THRESHOLD = 5

# Filename inside data_dir
_HEALTH_FILENAME = "source_health.jsonl"


# ---------------------------------------------------------------------------
# In-memory per-source health record
# ---------------------------------------------------------------------------


@dataclass
class SourceHealthRecord:
    """
    In-memory aggregate for one source during a pipeline run.

    All counts start at zero and are updated by record_success /
    record_error. The record is flushed to disk at the end of a run.
    """

    source_id: str
    run_date: str = field(default_factory=lambda: date.today().isoformat())

    # Counts for this run
    attempts: int = 0
    successes: int = 0
    errors: int = 0
    last_item_count: int = 0

    # Rolling consecutive-error counter (persisted across runs)
    consecutive_errors: int = 0

    # Last error message (for display)
    last_error: Optional[str] = None

    def record_success(self, item_count: int) -> None:
        self.attempts += 1
        self.successes += 1
        self.last_item_count = item_count
        self.consecutive_errors = 0
        self.last_error = None

    def record_error(self, error: str) -> None:
        self.attempts += 1
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = str(error)[:200]  # cap length

    @property
    def success_rate(self) -> float:
        """Success rate for this run (0.0 – 1.0)."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def healthy(self) -> bool:
        """True if the source has had no errors in this run."""
        return self.errors == 0

    def to_jsonl_dict(self) -> dict:
        return {
            "date": self.run_date,
            "source_id": self.source_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "errors": self.errors,
            "last_item_count": self.last_item_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


# ---------------------------------------------------------------------------
# SourceHealthTracker
# ---------------------------------------------------------------------------


class SourceHealthTracker:
    """
    Tracks fetch health for all sources during a pipeline run.

    Thread-safe: uses a threading.Lock so the orchestrator can call
    record_* from concurrent asyncio tasks without data races.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. Path("data")). The health file is
        written to ``data_dir / "source_health.jsonl"``.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / _HEALTH_FILENAME
        self._records: dict[str, SourceHealthRecord] = {}
        self._lock = threading.Lock()

        # Load consecutive_errors from last run to persist rolling state
        self._prev_consecutive: dict[str, int] = self._load_prev_consecutive()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_success(self, source_id: str, item_count: int = 0) -> None:
        """Record a successful fetch for a source."""
        with self._lock:
            rec = self._get_or_create(source_id)
            rec.record_success(item_count)

    def record_error(self, source_id: str, error: str) -> None:
        """Record a fetch failure for a source."""
        with self._lock:
            rec = self._get_or_create(source_id)
            rec.record_error(error)

    def should_disable(self, source_id: str) -> bool:
        """
        Return True if this source should be auto-disabled.

        Triggers when consecutive_errors ≥ CONSECUTIVE_ERROR_THRESHOLD.
        This combines previous-run and current-run consecutive errors so
        a source that was already failing before this run is flagged
        immediately.
        """
        with self._lock:
            rec = self._records.get(source_id)
            consecutive = rec.consecutive_errors if rec else self._prev_consecutive.get(source_id, 0)
            return consecutive >= CONSECUTIVE_ERROR_THRESHOLD

    def get_record(self, source_id: str) -> Optional[SourceHealthRecord]:
        """Return the current health record for a source, or None."""
        with self._lock:
            return self._records.get(source_id)

    def all_records(self) -> list[SourceHealthRecord]:
        """Return all in-memory records, sorted by source_id."""
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.source_id)

    def flush(self) -> None:
        """
        Append today's health summary for all tracked sources to the JSONL file.

        Creates the file (and parent directories) if they don't exist.
        Silently swallows I/O errors so a flush failure never aborts a run.
        """
        with self._lock:
            records_snapshot = list(self._records.values())

        if not records_snapshot:
            return

        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for rec in records_snapshot:
                    line = json.dumps(rec.to_jsonl_dict(), ensure_ascii=False)
                    f.write(line + "\n")
        except OSError:
            pass  # Never crash the pipeline over a stats flush

    def load_history(self, days: int = 30) -> list[dict]:
        """
        Load historical health records from the JSONL file.

        Parameters
        ----------
        days:
            Maximum age of records to return (default: 30 days).
            Records older than this are excluded.

        Returns
        -------
        list[dict]
            Parsed JSONL rows, newest first. Malformed lines are skipped.
        """
        if not self._path.exists():
            return []

        cutoff = date.today().toordinal() - days
        records: list[dict] = []

        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    row_date = date.fromisoformat(row.get("date", "2000-01-01"))
                    if row_date.toordinal() >= cutoff:
                        records.append(row)
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return []

        return list(reversed(records))  # newest first

    def source_summary(self, days: int = 30) -> dict[str, dict]:
        """
        Aggregate the last N days of history into per-source stats.

        Returns
        -------
        dict[source_id → stats_dict]
            stats_dict contains:
              - total_attempts: int
              - total_successes: int
              - total_errors: int
              - success_rate: float (0–1)
              - avg_items: float
              - consecutive_errors: int  (from most recent row)
              - last_error: str | None
              - last_seen: str (most recent date)
              - days_tracked: int
        """
        rows = self.load_history(days=days)

        # Group by source_id (rows are newest-first, so first seen = most recent)
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            sid = row.get("source_id", "?")
            grouped.setdefault(sid, []).append(row)

        summary = {}
        for sid, source_rows in grouped.items():
            total_attempts = sum(r.get("attempts", 0) for r in source_rows)
            total_successes = sum(r.get("successes", 0) for r in source_rows)
            total_errors = sum(r.get("errors", 0) for r in source_rows)
            total_items = sum(r.get("last_item_count", 0) for r in source_rows)
            n = len(source_rows)

            most_recent = source_rows[0]  # already newest-first

            summary[sid] = {
                "total_attempts": total_attempts,
                "total_successes": total_successes,
                "total_errors": total_errors,
                "success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
                "avg_items": total_items / n if n > 0 else 0.0,
                "consecutive_errors": most_recent.get("consecutive_errors", 0),
                "last_error": most_recent.get("last_error"),
                "last_seen": most_recent.get("date", "?"),
                "days_tracked": n,
            }

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, source_id: str) -> SourceHealthRecord:
        """Return the record for source_id, creating it if needed."""
        if source_id not in self._records:
            prev_consecutive = self._prev_consecutive.get(source_id, 0)
            rec = SourceHealthRecord(source_id=source_id)
            rec.consecutive_errors = prev_consecutive
            self._records[source_id] = rec
        return self._records[source_id]

    def _load_prev_consecutive(self) -> dict[str, int]:
        """
        Load the most recent consecutive_errors value per source from disk.

        This ensures the rolling streak persists across pipeline runs —
        a source that was already at 4 errors will hit the threshold after
        just 1 more failure in the next run.
        """
        if not self._path.exists():
            return {}

        # Read all lines, keep the most recent per source_id
        latest: dict[str, dict] = {}
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line.strip())
                    sid = row.get("source_id")
                    if sid:
                        latest[sid] = row
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return {}

        return {
            sid: row.get("consecutive_errors", 0)
            for sid, row in latest.items()
        }
