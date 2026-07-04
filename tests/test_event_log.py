"""
tests/test_event_log.py
========================
Tests for the upgraded pipeline event log (src/pipeline/event_log.py).

Coverage:
  - EventLog: write/read roundtrip, all event methods, malformed lines skipped
  - parse_run_timeline: happy path, no run_start returns None, stage ordering,
    ai_cost extraction, error stage flagging
  - StageTimeline.label: all known stage names
  - aggregate_runs: empty dir, single run, multi-run, success rate, cost sum
  - build_status_panel: returns None when no logs, returns Rich Table otherwise
  - RunAggregate.success_rate: edge cases

All tests use temp dirs — no real pipeline runs needed.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.pipeline.event_log import (
    EventLog,
    RunAggregate,
    RunTimeline,
    StageTimeline,
    aggregate_runs,
    build_status_panel,
    parse_run_timeline,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_events(
    *,
    status: str = "success",
    items: int = 10,
    duration_s: float = 30.0,
    fetch_count: int = 47,
    fetch_dur: float = 3.2,
    score_count: int = 12,
    score_dur: float = 8.7,
    ai_tokens: int = 1200,
    ai_cost: float = 0.00018,
    dry_run: bool = False,
    with_ai_cost: bool = True,
) -> list[dict]:
    """Build a minimal list of event dicts for a complete pipeline run."""
    events = [
        {"ts": "2026-07-04T10:00:00Z", "level": "INFO",  "stage": "pipeline", "event": "run_start",    "data": {"briefing_date": "2026-07-04", "dry_run": dry_run}},
        {"ts": "2026-07-04T10:00:01Z", "level": "INFO",  "stage": "fetch",    "event": "stage_start",  "data": {"sources": 5}},
        {"ts": "2026-07-04T10:00:04Z", "level": "INFO",  "stage": "fetch",    "event": "stage_end",    "data": {"count": fetch_count, "duration_s": fetch_dur}},
        {"ts": "2026-07-04T10:00:04Z", "level": "INFO",  "stage": "score",    "event": "stage_start",  "data": {}},
        {"ts": "2026-07-04T10:00:13Z", "level": "INFO",  "stage": "score",    "event": "stage_end",    "data": {"count": score_count, "duration_s": score_dur}},
    ]
    if with_ai_cost:
        events.append({"ts": "2026-07-04T10:00:28Z", "level": "INFO", "stage": "pipeline", "event": "ai_cost",
                        "data": {"model": "gpt-4o-mini", "tokens": ai_tokens, "cost_usd": ai_cost, "calls": 8, "retries": 0}})
    events.append({"ts": "2026-07-04T10:00:30Z", "level": "INFO",  "stage": "pipeline", "event": "run_end",
                   "data": {"status": status, "items_in_briefing": items, "duration_s": duration_s, "errors": []}})
    return events


# ===========================================================================
# EventLog — write/read
# ===========================================================================


class TestEventLogWriteRead:
    @pytest.mark.unit
    def test_creates_log_file_on_first_write(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.start_run("2026-07-04")
        assert ev.log_path.exists()

    @pytest.mark.unit
    def test_read_events_returns_empty_when_no_file(self, tmp_path):
        ev = EventLog(tmp_path)
        # Don't write anything
        assert ev.read_events() == []

    @pytest.mark.unit
    def test_write_and_read_roundtrip(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.start_run("2026-07-04")
        ev.end_run(status="success", items=10, duration_s=30.0)
        events = ev.read_events()
        assert len(events) == 2
        assert events[0]["event"] == "run_start"
        assert events[1]["event"] == "run_end"

    @pytest.mark.unit
    def test_event_has_required_fields(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.start_run("2026-07-04")
        events = ev.read_events()
        e = events[0]
        assert "ts" in e
        assert "level" in e
        assert "stage" in e
        assert "event" in e
        assert "data" in e

    @pytest.mark.unit
    def test_malformed_line_is_skipped(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.start_run("2026-07-04")
        # Manually corrupt the file
        ev.log_path.write_text(
            ev.log_path.read_text() + "NOT JSON\n",
            encoding="utf-8",
        )
        events = ev.read_events()
        # Should still get the valid line, skip the bad one
        assert len(events) == 1

    @pytest.mark.unit
    def test_load_log_on_missing_file_returns_empty(self, tmp_path):
        result = EventLog.load_log(tmp_path / "ghost.jsonl")
        assert result == []

    @pytest.mark.unit
    def test_list_log_files_returns_sorted(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "2026-07-01.jsonl").write_text("{}\n")
        (logs_dir / "2026-07-03.jsonl").write_text("{}\n")
        (logs_dir / "2026-07-02.jsonl").write_text("{}\n")
        files = EventLog.list_log_files(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)

    @pytest.mark.unit
    def test_list_log_files_empty_when_no_logs_dir(self, tmp_path):
        assert EventLog.list_log_files(tmp_path / "nonexistent") == []


class TestEventLogMethods:
    @pytest.mark.unit
    def test_start_run_writes_run_start_event(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.start_run("2026-07-04", dry_run=True)
        events = ev.read_events()
        assert events[0]["event"] == "run_start"
        assert events[0]["data"]["dry_run"] is True

    @pytest.mark.unit
    def test_end_run_writes_run_end_event(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.end_run(status="success", items=8, duration_s=45.0)
        events = ev.read_events()
        assert events[0]["event"] == "run_end"
        assert events[0]["data"]["items_in_briefing"] == 8

    @pytest.mark.unit
    def test_ai_cost_writes_ai_cost_event(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.ai_cost("gpt-4o-mini", tokens=1800, cost_usd=0.00027, calls=12, retries=1)
        events = ev.read_events()
        assert events[0]["event"] == "ai_cost"
        assert events[0]["data"]["model"] == "gpt-4o-mini"
        assert events[0]["data"]["tokens"] == 1800
        assert events[0]["data"]["retries"] == 1

    @pytest.mark.unit
    def test_scraper_result_writes_correct_event(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.scraper_result("hn-top", count=28, duration_s=1.2)
        events = ev.read_events()
        assert events[0]["event"] == "scraper_result"
        assert events[0]["data"]["source_id"] == "hn-top"

    @pytest.mark.unit
    def test_delivery_result_success_is_info(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.delivery_result("email", success=True)
        events = ev.read_events()
        assert events[0]["level"] == "INFO"

    @pytest.mark.unit
    def test_delivery_result_failure_is_warning(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.delivery_result("email", success=False, error="SMTP timeout")
        events = ev.read_events()
        assert events[0]["level"] == "WARNING"

    @pytest.mark.unit
    def test_stage_error_is_level_error(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.stage_error("fetch", "Connection refused")
        events = ev.read_events()
        assert events[0]["level"] == "ERROR"
        assert events[0]["data"]["error"] == "Connection refused"

    @pytest.mark.unit
    def test_dedup_result_calculates_removed(self, tmp_path):
        ev = EventLog(tmp_path)
        ev.dedup_result(before=50, after=38)
        events = ev.read_events()
        assert events[0]["data"]["removed"] == 12


# ===========================================================================
# parse_run_timeline
# ===========================================================================


class TestParseRunTimeline:
    @pytest.mark.unit
    def test_happy_path_returns_timeline(self):
        events = _make_events()
        timeline = parse_run_timeline(events)
        assert isinstance(timeline, RunTimeline)

    @pytest.mark.unit
    def test_status_parsed_correctly(self):
        timeline = parse_run_timeline(_make_events(status="success"))
        assert timeline.status == "success"
        assert timeline.success is True

    @pytest.mark.unit
    def test_error_status_success_false(self):
        timeline = parse_run_timeline(_make_events(status="error"))
        assert not timeline.success

    @pytest.mark.unit
    def test_returns_none_when_no_run_start(self):
        events = [{"ts": "T", "event": "run_end", "stage": "pipeline", "data": {"status": "success"}}]
        assert parse_run_timeline(events) is None

    @pytest.mark.unit
    def test_empty_events_returns_none(self):
        assert parse_run_timeline([]) is None

    @pytest.mark.unit
    def test_stage_duration_extracted(self):
        timeline = parse_run_timeline(_make_events(fetch_dur=4.5))
        fetch = timeline.stage("fetch")
        assert fetch is not None
        assert fetch.duration_s == 4.5

    @pytest.mark.unit
    def test_stage_item_count_extracted(self):
        timeline = parse_run_timeline(_make_events(fetch_count=55))
        fetch = timeline.stage("fetch")
        assert fetch.item_count == 55

    @pytest.mark.unit
    def test_ai_cost_extracted(self):
        timeline = parse_run_timeline(_make_events(ai_cost=0.00099, ai_tokens=3000))
        assert timeline.ai_cost_usd == 0.00099
        assert timeline.ai_tokens == 3000
        assert timeline.ai_model == "gpt-4o-mini"

    @pytest.mark.unit
    def test_no_ai_cost_event_gives_zeros(self):
        timeline = parse_run_timeline(_make_events(with_ai_cost=False))
        assert timeline.ai_cost_usd == 0.0
        assert timeline.ai_tokens == 0

    @pytest.mark.unit
    def test_stage_order_preserved(self):
        timeline = parse_run_timeline(_make_events())
        names = [s.name for s in timeline.stages]
        assert names.index("fetch") < names.index("score")

    @pytest.mark.unit
    def test_stage_error_flag_set(self):
        events = _make_events()
        events.append({"ts": "T", "level": "ERROR", "stage": "fetch", "event": "stage_error",
                        "data": {"error": "timeout"}})
        timeline = parse_run_timeline(events)
        fetch = timeline.stage("fetch")
        assert fetch.had_error is True

    @pytest.mark.unit
    def test_run_date_extracted(self):
        timeline = parse_run_timeline(_make_events())
        assert timeline.run_date == "2026-07-04"

    @pytest.mark.unit
    def test_dry_run_flag_extracted(self):
        timeline = parse_run_timeline(_make_events(dry_run=True))
        assert timeline.dry_run is True

    @pytest.mark.unit
    def test_total_duration_extracted(self):
        timeline = parse_run_timeline(_make_events(duration_s=99.5))
        assert timeline.total_duration_s == 99.5

    @pytest.mark.unit
    def test_items_in_briefing_extracted(self):
        timeline = parse_run_timeline(_make_events(items=15))
        assert timeline.items_in_briefing == 15


# ===========================================================================
# StageTimeline.label
# ===========================================================================


class TestStageTimelineLabel:
    @pytest.mark.unit
    def test_fetch_label(self):
        assert StageTimeline(name="fetch").label == "Fetch"

    @pytest.mark.unit
    def test_score_label(self):
        assert StageTimeline(name="score").label == "Score"

    @pytest.mark.unit
    def test_unknown_stage_capitalizes(self):
        assert StageTimeline(name="custom").label == "Custom"


# ===========================================================================
# aggregate_runs
# ===========================================================================


class TestAggregateRuns:
    @pytest.mark.unit
    def test_empty_dir_returns_zero_runs(self, tmp_path):
        agg = aggregate_runs(tmp_path, days=7)
        assert agg.run_count == 0

    @pytest.mark.unit
    def test_single_success_run(self, tmp_path):
        ev = EventLog(tmp_path)
        for e in _make_events(status="success", items=10, duration_s=30.0):
            ev._write(e["level"], e["stage"], e["event"], e["data"])
        agg = aggregate_runs(tmp_path, days=7)
        assert agg.run_count == 1
        assert agg.success_count == 1
        assert agg.success_rate == 1.0

    @pytest.mark.unit
    def test_single_error_run(self, tmp_path):
        ev = EventLog(tmp_path)
        for e in _make_events(status="error"):
            ev._write(e["level"], e["stage"], e["event"], e["data"])
        agg = aggregate_runs(tmp_path, days=7)
        assert agg.error_count == 1
        assert agg.success_rate == 0.0

    @pytest.mark.unit
    def test_cost_summed_across_runs(self, tmp_path):
        # Write 3 runs to 3 separate log files
        for i, day in enumerate(["2026-07-01", "2026-07-02", "2026-07-03"]):
            log_file = tmp_path / "logs" / f"{day}.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            ev = EventLog.__new__(EventLog)
            ev.data_dir = tmp_path
            ev.log_date = date.fromisoformat(day)
            ev._log_path = log_file
            for e in _make_events(ai_cost=0.001):
                ev._write(e["level"], e["stage"], e["event"], e["data"])
        agg = aggregate_runs(tmp_path, days=7)
        assert agg.run_count == 3
        assert abs(agg.total_cost_usd - 0.003) < 1e-9

    @pytest.mark.unit
    def test_days_parameter_limits_files(self, tmp_path):
        # Write 10 log files
        for i in range(1, 11):
            log_file = tmp_path / "logs" / f"2026-07-{i:02d}.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            ev = EventLog.__new__(EventLog)
            ev.data_dir = tmp_path
            ev.log_date = date(2026, 7, i)
            ev._log_path = log_file
            for e in _make_events():
                ev._write(e["level"], e["stage"], e["event"], e["data"])
        # Only last 3 days
        agg = aggregate_runs(tmp_path, days=3)
        assert agg.run_count == 3


# ===========================================================================
# RunAggregate.success_rate
# ===========================================================================


class TestRunAggregate:
    @pytest.mark.unit
    def test_success_rate_zero_when_no_runs(self):
        agg = RunAggregate()
        assert agg.success_rate == 0.0

    @pytest.mark.unit
    def test_success_rate_100_percent(self):
        agg = RunAggregate(run_count=5, success_count=5)
        assert agg.success_rate == 1.0

    @pytest.mark.unit
    def test_success_rate_partial(self):
        agg = RunAggregate(run_count=4, success_count=3)
        assert abs(agg.success_rate - 0.75) < 0.001


# ===========================================================================
# build_status_panel
# ===========================================================================


class TestBuildStatusPanel:
    @pytest.mark.unit
    def test_returns_none_when_no_log_files(self, tmp_path):
        panel = build_status_panel(tmp_path)
        assert panel is None

    @pytest.mark.unit
    def test_returns_rich_table_when_logs_exist(self, tmp_path):
        ev = EventLog(tmp_path)
        for e in _make_events():
            ev._write(e["level"], e["stage"], e["event"], e["data"])
        panel = build_status_panel(tmp_path)
        assert panel is not None

    @pytest.mark.unit
    def test_returns_none_when_log_file_empty(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "2026-07-04.jsonl").write_text("")
        panel = build_status_panel(tmp_path)
        assert panel is None

    @pytest.mark.unit
    def test_uses_last_run_when_multiple_runs_in_file(self, tmp_path):
        ev = EventLog(tmp_path)
        # Write two runs with different item counts
        for e in _make_events(items=5, status="error"):
            ev._write(e["level"], e["stage"], e["event"], e["data"])
        for e in _make_events(items=15, status="success"):
            ev._write(e["level"], e["stage"], e["event"], e["data"])
        panel = build_status_panel(tmp_path)
        # Should not crash and should return a table
        assert panel is not None
