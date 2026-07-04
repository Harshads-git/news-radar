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

Day 19 additions:
  - RunTimeline: reconstructs per-stage durations from start/end events
  - aggregate_runs(): 7/30-day roll-up of success rate, avg duration, cost
  - ai_cost event: logs token usage and estimated USD cost per run
  - status_panel(): builds a Rich renderable timeline for --status

Usage:
    from src.pipeline.event_log import EventLog
    from src.config import settings

    log = EventLog(settings.data_dir)
    log.start_run("2026-06-25", dry_run=False)
    log.stage_start("fetch", sources=12)
    log.scraper_result("hn-top", count=28, duration_s=1.2)
    log.stage_end("fetch", count=28, duration_s=4.1)
    log.ai_cost("gpt-4o-mini", tokens=1200, cost_usd=0.00018, calls=8)
    log.end_run(status="success", items=10, duration_s=45.2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# RunTimeline — reconstructed from event log
# ---------------------------------------------------------------------------


@dataclass
class StageTimeline:
    """Timing information for one pipeline stage."""
    name: str
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    item_count: int = 0
    had_error: bool = False

    @property
    def label(self) -> str:
        """Human-readable stage label."""
        return {
            "fetch": "Fetch",
            "dedup": "Dedup",
            "score": "Score",
            "summarize": "Summarize",
            "build": "Build",
            "render": "Render",
            "delivery": "Delivery",
            "pipeline": "Total",
        }.get(self.name, self.name.capitalize())


@dataclass
class RunTimeline:
    """
    A parsed view of one pipeline run reconstructed from JSONL events.

    Why reconstruct from events rather than just reading run_log.json?
    The event log has per-stage granularity (fetch took 3.2s, score took 8.7s),
    whereas run_log.json only records total duration. This lets --status show
    a per-stage breakdown bar.
    """
    run_date: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_duration_s: float = 0.0
    status: str = "unknown"
    dry_run: bool = False
    items_in_briefing: int = 0
    errors: list[str] = field(default_factory=list)
    stages: list[StageTimeline] = field(default_factory=list)

    # AI cost (from ai_cost event)
    ai_tokens: int = 0
    ai_cost_usd: float = 0.0
    ai_calls: int = 0
    ai_model: str = ""

    @property
    def success(self) -> bool:
        return self.status == "success"

    def stage(self, name: str) -> StageTimeline | None:
        """Return the StageTimeline for a given stage name, or None."""
        for s in self.stages:
            if s.name == name:
                return s
        return None


def parse_run_timeline(events: list[dict]) -> RunTimeline | None:
    """
    Parse a list of JSONL events for one run into a RunTimeline.

    Assumes events are in chronological order (as written).
    Returns None if there are no pipeline events.
    """
    run = RunTimeline()
    stage_starts: dict[str, str] = {}  # stage → start timestamp
    stage_map: dict[str, StageTimeline] = {}

    for ev in events:
        evt = ev.get("event", "")
        stage = ev.get("stage", "")
        ts = ev.get("ts", "")
        data = ev.get("data", {})

        if evt == "run_start":
            run.started_at = ts
            run.run_date = data.get("briefing_date", "")
            run.dry_run = data.get("dry_run", False)

        elif evt == "run_end":
            run.finished_at = ts
            run.status = data.get("status", "unknown")
            run.items_in_briefing = data.get("items_in_briefing", 0)
            run.total_duration_s = data.get("duration_s", 0.0)
            run.errors = data.get("errors", [])

        elif evt == "stage_start":
            stage_starts[stage] = ts
            st = StageTimeline(name=stage, started_at=ts)
            stage_map[stage] = st

        elif evt == "stage_end":
            st = stage_map.get(stage)
            if st is None:
                st = StageTimeline(name=stage)
                stage_map[stage] = st
            st.ended_at = ts
            st.duration_s = data.get("duration_s", 0.0)
            st.item_count = data.get("count", 0)

        elif evt == "stage_error":
            st = stage_map.get(stage)
            if st:
                st.had_error = True

        elif evt == "ai_cost":
            run.ai_tokens = data.get("tokens", 0)
            run.ai_cost_usd = data.get("cost_usd", 0.0)
            run.ai_calls = data.get("calls", 0)
            run.ai_model = data.get("model", "")

    if not run.started_at:
        return None

    # Preserve pipeline stage order
    _STAGE_ORDER = ["fetch", "dedup", "score", "summarize", "build", "render", "delivery"]
    run.stages = [
        stage_map[s] for s in _STAGE_ORDER if s in stage_map
    ]
    # Append any unexpected stages not in the standard order
    for name, st in stage_map.items():
        if name not in _STAGE_ORDER:
            run.stages.append(st)

    return run


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------


@dataclass
class RunAggregate:
    """Aggregated statistics over N pipeline runs."""
    run_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_duration_s: float = 0.0
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0
    total_items: int = 0
    avg_items: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.run_count if self.run_count else 0.0


def aggregate_runs(
    data_dir: Path | str,
    days: int = 7,
) -> RunAggregate:
    """
    Aggregate run statistics from the last N days of JSONL event logs.

    Why aggregate from event logs rather than run_log.json?
    Event logs include per-run AI cost data (ai_cost events) which
    run_log.json also has, but event logs let us track trends over time
    more flexibly — future analysis could e.g. plot cost per day.

    Parameters
    ----------
    data_dir:
        Root data directory (parent of logs/).
    days:
        Number of days to look back.

    Returns
    -------
    RunAggregate with counts, averages, and totals.
    """
    agg = RunAggregate()
    log_files = EventLog.list_log_files(data_dir)

    # Keep only the last N files
    log_files = log_files[-days:]

    durations: list[float] = []
    items_list: list[int] = []

    for log_file in log_files:
        events = EventLog.load_log(log_file)
        timeline = parse_run_timeline(events)
        if timeline is None:
            continue

        agg.run_count += 1
        if timeline.success:
            agg.success_count += 1
        else:
            agg.error_count += 1

        if timeline.total_duration_s > 0:
            durations.append(timeline.total_duration_s)

        agg.total_items += timeline.items_in_briefing
        items_list.append(timeline.items_in_briefing)

        agg.total_cost_usd += timeline.ai_cost_usd
        agg.total_tokens += timeline.ai_tokens

    if durations:
        agg.avg_duration_s = sum(durations) / len(durations)
        agg.min_duration_s = min(durations)
        agg.max_duration_s = max(durations)

    if items_list:
        agg.avg_items = sum(items_list) / len(items_list)

    if agg.run_count > 0:
        agg.avg_cost_usd = agg.total_cost_usd / agg.run_count

    return agg


# ---------------------------------------------------------------------------
# EventLog class
# ---------------------------------------------------------------------------


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

    def ai_cost(
        self,
        model: str,
        *,
        tokens: int,
        cost_usd: float,
        calls: int,
        retries: int = 0,
    ) -> None:
        """
        Log AI provider cost for this run.

        Persisted so aggregate_runs() can sum total spend over N days
        without needing to re-query the AI provider.
        """
        self._write("INFO", "pipeline", "ai_cost", {
            "model": model,
            "tokens": tokens,
            "cost_usd": round(cost_usd, 6),
            "calls": calls,
            "retries": retries,
        })

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


# ---------------------------------------------------------------------------
# --status timeline panel builder
# ---------------------------------------------------------------------------


def build_status_panel(data_dir: Path | str) -> object | None:
    """
    Build a Rich renderable showing the last run's stage timeline.

    Returns None if no logs exist yet. Used by `--status` command.

    Timeline format (example):
        Stage        Duration   Items   Status
        ─────────────────────────────────────
        Fetch          3.2s      47    ✓
        Dedup          0.1s      38    ✓
        Score          8.7s      12    ✓
        Summarize     14.2s      12    ✓
        Build          0.3s      10    ✓
        ─────────────────────────────────────
        Total         26.5s      10    success  💰 $0.002
    """
    try:
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return None

    log_files = EventLog.list_log_files(data_dir)
    if not log_files:
        return None

    # Load the most recent log file
    events = EventLog.load_log(log_files[-1])
    if not events:
        return None

    # Find the LAST run_start in the file (a file may have multiple runs)
    last_run_start = -1
    for i, ev in enumerate(events):
        if ev.get("event") == "run_start":
            last_run_start = i

    if last_run_start < 0:
        return None

    run_events = events[last_run_start:]
    timeline = parse_run_timeline(run_events)
    if timeline is None:
        return None

    table = Table(
        title="Last Run Timeline",
        show_header=True,
        header_style="bold cyan",
        show_edge=True,
        min_width=50,
    )
    table.add_column("Stage", style="cyan", min_width=12)
    table.add_column("Duration", justify="right", min_width=9)
    table.add_column("Items", justify="right", min_width=6)
    table.add_column("Status", min_width=8)

    for st in timeline.stages:
        dur = f"{st.duration_s:.1f}s" if st.duration_s else "—"
        items = str(st.item_count) if st.item_count else "—"
        status_icon = "[red]✗[/red]" if st.had_error else "[green]✓[/green]"
        table.add_row(st.label, dur, items, status_icon)

    # Total row
    status_color = "green" if timeline.success else "red"
    cost_str = ""
    if timeline.ai_cost_usd > 0:
        cost_str = f"  💰 ${timeline.ai_cost_usd:.4f}"

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{timeline.total_duration_s:.1f}s[/bold]",
        f"[bold]{timeline.items_in_briefing}[/bold]",
        f"[bold {status_color}]{timeline.status}[/bold {status_color}]{cost_str}",
    )

    return table
