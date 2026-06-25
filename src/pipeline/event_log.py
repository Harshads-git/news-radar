"""
src/pipeline/event_log.py
==========================
Structured JSON Lines event logger for pipeline runs.

Writes one JSON object per line to data/logs/YYYY-MM-DD.jsonl.
Each event has a standard envelope:

    {"ts": "2026-06-25T15:30:00Z", "level": "INFO", "stage": "fetch",
     "event": "scraper_done", "data": {"source": "hn-top", "count": 28}}

Why JSONL (JSON Lines)?
  - One event per line → safe for concurrent writes (append is atomic)
  - Trivially grep-able: `grep '"stage":"score"' data/logs/2026-06-25.jsonl`
  - Importable into any analytics tool (Pandas, DuckDB, BigQuery)
  - Human-readable with `cat` or `jq`
  - No schema migration needed — each event is self-describing

Usage:
    from src.pipeline.event_log import EventLog
    from src.config import settings

    log = EventLog(settings.data_dir)
    log.start_run("2026-06-25", dry_run=False)
    log.stage_start("fetch", sources=12)
    log.scraper_result("hn-top", count=28, duration_s=1.2)
    log.stage_end("fetch", count=28, duration_s=4.1)
    log.end_run(status="success", items=10, duration_s=45.2)
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    """
    Appends structured JSON events to data/logs/YYYY-MM-DD.jsonl.

    Thread-safe for single-process use (Python's file append is atomic
    at the OS level for writes smaller than PIPE_BUF ~4 KB on Linux).
    """

    def __init__(self, data_dir: Path | str, log_date: date | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.log_date = log_date or date.today()
        self._log_path: Path | None = None

    @property
    def log_path(self) -> Path:
        """Resolved path to today's JSONL log file."""
        if self._log_path is None:
            logs_dir = self.data_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = logs_dir / f"{self.log_date.isoformat()}.jsonl"
        return self._log_path

    # ------------------------------------------------------------------
    # High-level pipeline events
    # ------------------------------------------------------------------

    def start_run(self, briefing_date: str, *, dry_run: bool = False) -> None:
        """Log the start of a pipeline run."""
        self._write("INFO", "pipeline", "run_start", {
            "briefing_date": briefing_date,
            "dry_run": dry_run,
        })

    def end_run(
        self,
        *,
        status: str,
        items: int,
        duration_s: float,
        errors: list[str] | None = None,
    ) -> None:
        """Log the end of a pipeline run with final stats."""
        self._write("INFO", "pipeline", "run_end", {
            "status": status,
            "items_in_briefing": items,
            "duration_s": round(duration_s, 2),
            "errors": errors or [],
        })

    def stage_start(self, stage: str, **kwargs: Any) -> None:
        """Log that a pipeline stage has started."""
        self._write("INFO", stage, "stage_start", dict(kwargs))

    def stage_end(self, stage: str, **kwargs: Any) -> None:
        """Log that a pipeline stage completed."""
        self._write("INFO", stage, "stage_end", dict(kwargs))

    def stage_error(self, stage: str, error: str, **kwargs: Any) -> None:
        """Log an error within a pipeline stage."""
        self._write("ERROR", stage, "stage_error", {"error": error, **kwargs})

    # ------------------------------------------------------------------
    # Stage-specific events
    # ------------------------------------------------------------------

    def scraper_result(self, source_id: str, *, count: int, duration_s: float) -> None:
        """Log the result of a single scraper."""
        self._write("DEBUG", "fetch", "scraper_result", {
            "source_id": source_id,
            "count": count,
            "duration_s": round(duration_s, 2),
        })

    def scraper_error(self, source_id: str, error: str) -> None:
        """Log a scraper failure."""
        self._write("WARNING", "fetch", "scraper_error", {
            "source_id": source_id,
            "error": error,
        })

    def dedup_result(self, before: int, after: int) -> None:
        """Log deduplication results."""
        self._write("INFO", "dedup", "dedup_result", {
            "before": before,
            "after": after,
            "removed": before - after,
        })

    def score_result(self, total: int, passed: int, threshold: int) -> None:
        """Log AI scoring results."""
        self._write("INFO", "score", "score_result", {
            "total_scored": total,
            "passed_threshold": passed,
            "threshold": threshold,
        })

    def delivery_result(self, channel: str, *, success: bool, error: str = "") -> None:
        """Log a delivery channel result."""
        level = "INFO" if success else "WARNING"
        self._write(level, "delivery", "delivery_result", {
            "channel": channel,
            "success": success,
            "error": error,
        })

    # ------------------------------------------------------------------
    # Core writer
    # ------------------------------------------------------------------

    def _write(
        self,
        level: str,
        stage: str,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Append one JSON line to the log file."""
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": level,
            "stage": stage,
            "event": event,
            "data": data,
        }
        line = json.dumps(entry, ensure_ascii=False)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # Never crash the pipeline due to logging failures

    # ------------------------------------------------------------------
    # Reader helpers
    # ------------------------------------------------------------------

    def read_events(self) -> list[dict]:
        """Read all events from today's log file."""
        if not self.log_path.exists():
            return []
        events = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    @staticmethod
    def load_log(log_file: Path) -> list[dict]:
        """Load events from any JSONL log file."""
        if not log_file.exists():
            return []
        events = []
        for line in log_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    @staticmethod
    def list_log_files(data_dir: Path | str) -> list[Path]:
        """Return all JSONL log files sorted oldest → newest."""
        logs_dir = Path(data_dir) / "logs"
        if not logs_dir.exists():
            return []
        return sorted(logs_dir.glob("*.jsonl"))
