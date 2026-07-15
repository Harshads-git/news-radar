"""
src/pipeline/cost_ledger.py
============================
Persistent AI API cost tracking across pipeline runs.

Every pipeline run that makes AI calls has its cost recorded to
``data/cost_log.jsonl`` — one JSON line per run. The ``--cost-report``
CLI command reads this file to display daily and weekly spend totals.

Why JSONL (not a database)?
  JSONL is append-only, portable, human-readable, and zero-dependency.
  A pipeline that runs once a day generates at most 365 lines/year —
  tiny even after years of use.

Entry format (one JSON line per run):
  {
    "date":             "2026-07-15",
    "run_id":           "2026-07-15T13:00:00Z",
    "model":            "gpt-4o-mini",
    "total_tokens":     1540,
    "prompt_tokens":    1200,
    "completion_tokens": 340,
    "total_calls":      42,
    "retry_calls":      1,
    "failed_calls":     0,
    "cost_usd":         0.000924,
    "dry_run":          false,
    "written_at":       "2026-07-15T13:05:22Z"
  }

Usage:
    from src.pipeline.cost_ledger import CostLedger

    ledger = CostLedger(data_dir)
    ledger.record(provider.cost_tracker, model=settings.ai_model, dry_run=dry_run)

    # For the CLI:
    report = ledger.daily_report(days=30)
    for row in report:
        print(row["date"], row["cost_usd"])
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retry import CostTracker

# Filename inside data_dir
_LEDGER_FILENAME = "cost_log.jsonl"


class CostLedger:
    """
    Append-only JSONL cost ledger for AI pipeline runs.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. Path("data")). The ledger is written to
        ``data_dir / "cost_log.jsonl"``.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / _LEDGER_FILENAME

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(
        self,
        cost_tracker: "CostTracker",
        *,
        model: str = "unknown",
        dry_run: bool = False,
        run_id: str | None = None,
    ) -> dict:
        """
        Append one cost entry from a CostTracker to the JSONL file.

        Parameters
        ----------
        cost_tracker:
            The CostTracker from the AI provider after a pipeline run.
        model:
            The AI model name (from settings.ai_model).
        dry_run:
            Whether this was a dry run (costs still real, but no output saved).
        run_id:
            Optional run identifier (ISO timestamp). Defaults to now-UTC.

        Returns
        -------
        dict
            The entry that was written (for testing / logging).

        Notes
        -----
        Silently ignores I/O errors so a write failure never aborts a run.
        """
        now_utc = datetime.now(timezone.utc)
        entry = {
            "date": now_utc.strftime("%Y-%m-%d"),
            "run_id": run_id or now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "total_tokens": cost_tracker.total_tokens,
            "prompt_tokens": sum(r.prompt_tokens for r in cost_tracker.records),
            "completion_tokens": sum(r.completion_tokens for r in cost_tracker.records),
            "total_calls": cost_tracker.total_calls,
            "retry_calls": cost_tracker.retry_calls,
            "failed_calls": cost_tracker.failed_calls,
            "cost_usd": round(cost_tracker.total_cost_usd, 8),
            "dry_run": dry_run,
            "written_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._append(entry)
        return entry

    # ------------------------------------------------------------------
    # Reading / Aggregation
    # ------------------------------------------------------------------

    def load_entries(self, days: int = 30) -> list[dict]:
        """
        Load cost entries from the last N days, newest first.

        Parameters
        ----------
        days:
            How many calendar days of history to return. Default: 30.

        Returns
        -------
        list[dict]
            Parsed JSONL rows within the date window. Malformed lines
            are silently skipped.
        """
        if not self._path.exists():
            return []

        cutoff_date = (date.today() - timedelta(days=days)).isoformat()
        entries: list[dict] = []

        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("date", "0000-00-00") >= cutoff_date:
                        entries.append(row)
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            return []

        return list(reversed(entries))  # newest first

    def daily_report(self, days: int = 30) -> list[dict]:
        """
        Aggregate entries by date and return daily cost summaries.

        Returns
        -------
        list[dict], one entry per date (newest first). Each dict contains:
          - date: str           — YYYY-MM-DD
          - runs: int           — number of pipeline runs that day
          - total_tokens: int
          - total_calls: int
          - cost_usd: float     — sum of all run costs for that day
          - dry_runs: int       — how many of those runs were dry-run
        """
        entries = self.load_entries(days=days)
        by_date: dict[str, dict] = {}

        for row in entries:
            d = row.get("date", "?")
            if d not in by_date:
                by_date[d] = {
                    "date": d,
                    "runs": 0,
                    "total_tokens": 0,
                    "total_calls": 0,
                    "cost_usd": 0.0,
                    "dry_runs": 0,
                }
            agg = by_date[d]
            agg["runs"] += 1
            agg["total_tokens"] += row.get("total_tokens", 0)
            agg["total_calls"] += row.get("total_calls", 0)
            agg["cost_usd"] = round(agg["cost_usd"] + row.get("cost_usd", 0.0), 8)
            if row.get("dry_run"):
                agg["dry_runs"] += 1

        # Sort newest date first
        return sorted(by_date.values(), key=lambda r: r["date"], reverse=True)

    def weekly_summary(self, weeks: int = 4) -> list[dict]:
        """
        Aggregate entries by ISO week and return weekly cost summaries.

        Returns
        -------
        list[dict], one entry per ISO week (newest first). Each dict:
          - week: str     — "YYYY-Www" (ISO week number)
          - runs: int
          - total_tokens: int
          - cost_usd: float
        """
        entries = self.load_entries(days=weeks * 7)
        by_week: dict[str, dict] = {}

        for row in entries:
            try:
                d = date.fromisoformat(row.get("date", "2000-01-01"))
                iso_year, iso_week, _ = d.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
            except ValueError:
                continue

            if week_key not in by_week:
                by_week[week_key] = {
                    "week": week_key,
                    "runs": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            agg = by_week[week_key]
            agg["runs"] += 1
            agg["total_tokens"] += row.get("total_tokens", 0)
            agg["cost_usd"] = round(agg["cost_usd"] + row.get("cost_usd", 0.0), 8)

        return sorted(by_week.values(), key=lambda r: r["week"], reverse=True)

    def total_spend(self, days: int = 30) -> float:
        """Return total USD spend across the last N days."""
        return round(sum(r.get("cost_usd", 0.0) for r in self.load_entries(days=days)), 8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, entry: dict) -> None:
        """Append a single dict as a JSON line. Silently swallows I/O errors."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
