"""
tests/test_event_log_and_cli.py
================================
Unit tests for:
  - EventLog: write, read, multiple events, file organization, load helpers
  - _handle_check(): validation logic with mocked settings and sources
  - _handle_status(): run history table display (no crash on edge cases)

All tests use tmp_path for filesystem isolation.
CLI tests mock settings to avoid requiring real API keys or sources.json.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# EventLog Tests
# ===========================================================================


class TestEventLogWrite:
    @pytest.mark.unit
    def test_creates_log_file_on_first_write(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.start_run("2026-06-25")
        log_file = tmp_path / "logs" / "2026-06-25.jsonl"
        assert log_file.exists()

    @pytest.mark.unit
    def test_log_file_named_by_date(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.start_run("2026-06-25")
        assert (tmp_path / "logs" / "2026-06-25.jsonl").exists()

    @pytest.mark.unit
    def test_each_write_is_one_json_line(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.start_run("2026-06-25")
        log.end_run(status="success", items=5, duration_s=30.0)
        lines = (tmp_path / "logs" / "2026-06-25.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        # Each line must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert "ts" in obj
            assert "event" in obj

    @pytest.mark.unit
    def test_event_has_required_envelope_fields(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.stage_start("fetch", sources=5)
        events = log.read_events()
        assert len(events) == 1
        e = events[0]
        assert "ts" in e
        assert "level" in e
        assert "stage" in e
        assert "event" in e
        assert "data" in e

    @pytest.mark.unit
    def test_start_run_event_has_briefing_date(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.start_run("2026-06-25", dry_run=True)
        events = log.read_events()
        assert events[0]["data"]["briefing_date"] == "2026-06-25"
        assert events[0]["data"]["dry_run"] is True

    @pytest.mark.unit
    def test_end_run_event_has_status_and_items(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.end_run(status="success", items=12, duration_s=42.5, errors=["minor"])
        events = log.read_events()
        e = events[0]["data"]
        assert e["status"] == "success"
        assert e["items_in_briefing"] == 12
        assert e["duration_s"] == 42.5
        assert e["errors"] == ["minor"]

    @pytest.mark.unit
    def test_scraper_result_event(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.scraper_result("hn-top", count=28, duration_s=1.2)
        events = log.read_events()
        assert events[0]["stage"] == "fetch"
        assert events[0]["data"]["source_id"] == "hn-top"
        assert events[0]["data"]["count"] == 28

    @pytest.mark.unit
    def test_dedup_result_event(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.dedup_result(before=50, after=38)
        events = log.read_events()
        data = events[0]["data"]
        assert data["before"] == 50
        assert data["after"] == 38
        assert data["removed"] == 12

    @pytest.mark.unit
    def test_score_result_event(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.score_result(total=38, passed=12, threshold=6)
        events = log.read_events()
        data = events[0]["data"]
        assert data["total_scored"] == 38
        assert data["passed_threshold"] == 12

    @pytest.mark.unit
    def test_delivery_result_success_event(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.delivery_result("discord", success=True)
        events = log.read_events()
        assert events[0]["level"] == "INFO"
        assert events[0]["data"]["success"] is True

    @pytest.mark.unit
    def test_delivery_result_failure_event(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log.delivery_result("email", success=False, error="SMTP timeout")
        events = log.read_events()
        assert events[0]["level"] == "WARNING"
        assert events[0]["data"]["error"] == "SMTP timeout"

    @pytest.mark.unit
    def test_events_appended_not_overwritten(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        for i in range(5):
            log.stage_start(f"stage-{i}")
        events = log.read_events()
        assert len(events) == 5

    @pytest.mark.unit
    def test_read_events_empty_when_no_file(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        assert log.read_events() == []

    @pytest.mark.unit
    def test_never_crashes_on_unwritable_path(self, tmp_path):
        """EventLog must silently ignore OS errors — never crash the pipeline."""
        from src.pipeline.event_log import EventLog
        # Point to a file-as-directory to provoke OSError
        bad_path = tmp_path / "not-a-dir.jsonl"
        bad_path.mkdir()  # Create a directory where a file should be
        log = EventLog(tmp_path, log_date=date(2026, 6, 25))
        log._log_path = bad_path / "nope.jsonl"  # subpath of a directory
        log.start_run("2026-06-25")  # must NOT raise


class TestEventLogHelpers:
    @pytest.mark.unit
    def test_list_log_files_returns_sorted_paths(self, tmp_path):
        from src.pipeline.event_log import EventLog
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "2026-06-23.jsonl").write_text("")
        (logs_dir / "2026-06-25.jsonl").write_text("")
        (logs_dir / "2026-06-24.jsonl").write_text("")
        files = EventLog.list_log_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["2026-06-23.jsonl", "2026-06-24.jsonl", "2026-06-25.jsonl"]

    @pytest.mark.unit
    def test_list_log_files_empty_when_no_logs(self, tmp_path):
        from src.pipeline.event_log import EventLog
        assert EventLog.list_log_files(tmp_path) == []

    @pytest.mark.unit
    def test_load_log_reads_events_from_path(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log_file = tmp_path / "test.jsonl"
        log_file.write_text(
            '{"ts":"2026-06-25T10:00:00Z","level":"INFO","stage":"fetch","event":"run_start","data":{}}\n'
            '{"ts":"2026-06-25T10:01:00Z","level":"INFO","stage":"pipeline","event":"run_end","data":{"status":"success"}}\n'
        )
        events = EventLog.load_log(log_file)
        assert len(events) == 2
        assert events[1]["data"]["status"] == "success"

    @pytest.mark.unit
    def test_load_log_skips_invalid_lines(self, tmp_path):
        from src.pipeline.event_log import EventLog
        log_file = tmp_path / "test.jsonl"
        log_file.write_text(
            '{"valid": true}\n'
            'NOT JSON\n'
            '{"also_valid": true}\n'
        )
        events = EventLog.load_log(log_file)
        assert len(events) == 2

    @pytest.mark.unit
    def test_load_log_returns_empty_when_no_file(self, tmp_path):
        from src.pipeline.event_log import EventLog
        result = EventLog.load_log(tmp_path / "nonexistent.jsonl")
        assert result == []


# ===========================================================================
# _handle_check Tests
# ===========================================================================


def make_check_settings(tmp_path: Path, **overrides) -> MagicMock:
    s = MagicMock()
    s.ai_model = "gpt-4o-mini"
    s.active_model_provider = "openai"
    s.validate_ai_config.return_value = []  # no warnings by default
    s.sources_file = tmp_path / "sources.json"
    s.data_dir = tmp_path / "data"
    s.docs_dir = tmp_path / "docs"
    s.has_email = False
    s.has_discord = False
    s.has_slack = False
    s.custom_webhook_url = ""
    s.github_pages_enabled = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def write_valid_sources(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": [{
        "id": "hn-test", "type": "hackernews",
        "name": "HN", "enabled": True, "limit": 5, "tags": [],
    }]}))


class TestHandleCheck:
    @pytest.mark.unit
    def test_returns_0_when_all_ok(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path)
        write_valid_sources(s.sources_file)
        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 0

    @pytest.mark.unit
    def test_returns_1_when_ai_key_missing(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path)
        s.validate_ai_config.return_value = ["AI_MODEL='gpt-4o-mini' requires OPENAI_API_KEY"]
        write_valid_sources(s.sources_file)
        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 1

    @pytest.mark.unit
    def test_returns_1_when_sources_missing(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path)
        # Don't create sources.json
        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 1

    @pytest.mark.unit
    def test_returns_1_when_sources_invalid(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path)
        s.sources_file.parent.mkdir(parents=True, exist_ok=True)
        s.sources_file.write_text("NOT JSON")
        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 1

    @pytest.mark.unit
    def test_reports_configured_delivery_channels(self, tmp_path, capsys):
        from src.main import _handle_check
        s = make_check_settings(tmp_path, has_discord=True)
        write_valid_sources(s.sources_file)
        log = MagicMock()
        # Should not raise even with discord configured
        result = _handle_check(s, log)
        assert result == 0

    @pytest.mark.unit
    def test_github_pages_enabled_reported(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path, github_pages_enabled=True)
        write_valid_sources(s.sources_file)
        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 0

    @pytest.mark.unit
    def test_data_dir_created_if_missing(self, tmp_path):
        from src.main import _handle_check
        s = make_check_settings(tmp_path)
        s.data_dir = tmp_path / "new_data_dir"  # doesn't exist yet
        write_valid_sources(s.sources_file)
        log = MagicMock()
        _handle_check(s, log)
        assert s.data_dir.exists()


# ===========================================================================
# _handle_status Tests (smoke tests — checks it doesn't crash)
# ===========================================================================


class TestHandleStatus:
    @pytest.mark.unit
    def test_status_with_no_run_log(self, tmp_path):
        """--status must not crash when run_log.json doesn't exist."""
        from src.main import _handle_status
        s = MagicMock()
        s.ai_model = "gpt-4o-mini"
        s.score_threshold = 6
        s.max_briefing_items = 20
        s.output_language = "English"
        s.log_level = "INFO"
        s.sources_file = "data/sources.json"
        s.data_dir = tmp_path / "data"
        s.docs_dir = tmp_path / "docs"
        s.has_openai = False
        s.has_gemini = False
        s.has_anthropic = False
        s.has_email = False
        s.has_discord = False
        s.has_slack = False
        s.github_pages_enabled = True
        log = MagicMock()
        _handle_status(s, log)  # must not raise

    @pytest.mark.unit
    def test_status_with_valid_run_log(self, tmp_path):
        """--status with a real run_log.json must display run history."""
        from src.main import _handle_status
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        run_log = data_dir / "run_log.json"
        run_log.write_text(json.dumps([{
            "date": "2026-06-25", "status": "success",
            "in_briefing": 10, "fetched": 80,
            "duration_s": 42.0, "dry_run": False, "errors": [],
        }]))

        s = MagicMock()
        s.ai_model = "gpt-4o-mini"
        s.score_threshold = 6
        s.max_briefing_items = 20
        s.output_language = "English"
        s.log_level = "INFO"
        s.sources_file = "data/sources.json"
        s.data_dir = data_dir
        s.docs_dir = tmp_path / "docs"
        s.has_openai = True
        s.has_gemini = False
        s.has_anthropic = False
        s.has_email = False
        s.has_discord = False
        s.has_slack = False
        s.github_pages_enabled = True
        log = MagicMock()
        _handle_status(s, log)  # must not raise

    @pytest.mark.unit
    def test_status_with_corrupted_run_log(self, tmp_path):
        """--status must not crash even if run_log.json is corrupted."""
        from src.main import _handle_status
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "run_log.json").write_text("BROKEN JSON {{{")

        s = MagicMock()
        s.ai_model = "gpt-4o-mini"
        s.score_threshold = 6
        s.max_briefing_items = 20
        s.output_language = "English"
        s.log_level = "INFO"
        s.sources_file = "data/sources.json"
        s.data_dir = data_dir
        s.docs_dir = tmp_path / "docs"
        s.has_openai = False
        s.has_gemini = False
        s.has_anthropic = False
        s.has_email = False
        s.has_discord = False
        s.has_slack = False
        s.github_pages_enabled = True
        log = MagicMock()
        _handle_status(s, log)  # must not raise
