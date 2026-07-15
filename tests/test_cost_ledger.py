"""
tests/test_cost_ledger.py
==========================
Tests for Day 26 cost tracking in src/pipeline/cost_ledger.py.

Coverage:
  - CostLedger.record: creates JSONL file, writes valid JSON line,
    correct field values (date, model, tokens, cost_usd, dry_run),
    returns the entry dict, silent on bad directory path
  - CostLedger.load_entries: empty when no file, round-trip, newest
    first, date cutoff filters old entries, malformed lines skipped,
    multiple entries all loaded
  - CostLedger.daily_report: empty when no data, aggregates runs by
    date, cost_usd summed per day, tokens summed, dry_runs counted,
    sorted newest first, multi-day produces multiple rows
  - CostLedger.weekly_summary: empty when no data, groups by ISO week,
    cost_usd summed per week, runs counted
  - CostLedger.total_spend: zero with no data, correct sum across days
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.cost_ledger import CostLedger


# ===========================================================================
# Helpers
# ===========================================================================


def _make_tracker(model: str = "gpt-4o-mini", n: int = 1, cost_per_call: float = 0.0001):
    """Build a MagicMock CostTracker with predictable attributes."""
    tracker = MagicMock()
    # Build a minimal record list
    record = MagicMock()
    record.prompt_tokens = 100
    record.completion_tokens = 50
    tracker.records = [record] * n
    tracker.total_tokens = 150 * n
    tracker.total_cost_usd = cost_per_call * n
    tracker.total_calls = n
    tracker.retry_calls = 0
    tracker.failed_calls = 0
    return tracker


def _write_raw_entry(ledger: CostLedger, entry: dict) -> None:
    """Directly append a raw dict to the ledger file (for test setup)."""
    ledger._data_dir.mkdir(parents=True, exist_ok=True)
    with ledger._path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


# ===========================================================================
# CostLedger.record
# ===========================================================================


class TestCostLedgerRecord:
    @pytest.mark.unit
    def test_creates_ledger_file(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        assert (tmp_path / "cost_log.jsonl").exists()

    @pytest.mark.unit
    def test_writes_valid_json_line(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert isinstance(d, dict)

    @pytest.mark.unit
    def test_entry_has_correct_date(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert d["date"] == date.today().isoformat()

    @pytest.mark.unit
    def test_entry_has_correct_model(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="claude-3-haiku-20240307")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert d["model"] == "claude-3-haiku-20240307"

    @pytest.mark.unit
    def test_entry_has_correct_token_counts(self, tmp_path):
        ledger = CostLedger(tmp_path)
        tracker = _make_tracker(n=3)
        ledger.record(tracker, model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert d["total_tokens"] == 450  # 150 * 3

    @pytest.mark.unit
    def test_entry_has_correct_cost(self, tmp_path):
        ledger = CostLedger(tmp_path)
        tracker = _make_tracker(cost_per_call=0.00042)
        ledger.record(tracker, model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert abs(d["cost_usd"] - 0.00042) < 1e-9

    @pytest.mark.unit
    def test_entry_dry_run_flag_true(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini", dry_run=True)
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert d["dry_run"] is True

    @pytest.mark.unit
    def test_entry_dry_run_flag_false_by_default(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        d = json.loads(lines[0])
        assert d["dry_run"] is False

    @pytest.mark.unit
    def test_returns_entry_dict(self, tmp_path):
        ledger = CostLedger(tmp_path)
        result = ledger.record(_make_tracker(), model="gpt-4o-mini")
        assert isinstance(result, dict)
        assert "cost_usd" in result

    @pytest.mark.unit
    def test_record_silent_on_bad_path(self):
        ledger = CostLedger(Path("/nonexistent/deep/path"))
        try:
            ledger.record(_make_tracker(), model="gpt-4o-mini")
        except Exception as e:
            pytest.fail(f"record() raised unexpectedly: {e}")

    @pytest.mark.unit
    def test_appends_multiple_entries(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        lines = (tmp_path / "cost_log.jsonl").read_text().splitlines()
        assert len(lines) == 2


# ===========================================================================
# CostLedger.load_entries
# ===========================================================================


class TestLoadEntries:
    @pytest.mark.unit
    def test_empty_when_no_file(self, tmp_path):
        ledger = CostLedger(tmp_path)
        assert ledger.load_entries() == []

    @pytest.mark.unit
    def test_round_trip_after_record(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        entries = ledger.load_entries(days=30)
        assert len(entries) == 1

    @pytest.mark.unit
    def test_newest_first(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": _days_ago(5), "cost_usd": 0.001})
        _write_raw_entry(ledger, {"date": _days_ago(1), "cost_usd": 0.002})
        entries = ledger.load_entries(days=30)
        assert entries[0]["date"] >= entries[-1]["date"]

    @pytest.mark.unit
    def test_date_cutoff_filters_old_entries(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": _days_ago(60), "cost_usd": 0.001})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.002})
        entries = ledger.load_entries(days=7)
        dates = [e["date"] for e in entries]
        assert _days_ago(60) not in dates
        assert date.today().isoformat() in dates

    @pytest.mark.unit
    def test_malformed_lines_skipped(self, tmp_path):
        ledger = CostLedger(tmp_path)
        (tmp_path / "cost_log.jsonl").write_text("NOT JSON\n")
        assert ledger.load_entries() == []

    @pytest.mark.unit
    def test_multiple_entries_all_loaded(self, tmp_path):
        ledger = CostLedger(tmp_path)
        for _ in range(5):
            ledger.record(_make_tracker(), model="gpt-4o-mini")
        entries = ledger.load_entries(days=30)
        assert len(entries) == 5


# ===========================================================================
# CostLedger.daily_report
# ===========================================================================


class TestDailyReport:
    @pytest.mark.unit
    def test_empty_when_no_data(self, tmp_path):
        ledger = CostLedger(tmp_path)
        assert ledger.daily_report() == []

    @pytest.mark.unit
    def test_aggregates_runs_for_same_date(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        daily = ledger.daily_report(days=30)
        assert len(daily) == 1
        assert daily[0]["runs"] == 2

    @pytest.mark.unit
    def test_cost_summed_per_day(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.001, "total_tokens": 100, "total_calls": 1, "dry_run": False})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.002, "total_tokens": 200, "total_calls": 2, "dry_run": False})
        daily = ledger.daily_report(days=30)
        assert abs(daily[0]["cost_usd"] - 0.003) < 1e-9

    @pytest.mark.unit
    def test_tokens_summed_per_day(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.0, "total_tokens": 100, "total_calls": 1, "dry_run": False})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.0, "total_tokens": 200, "total_calls": 1, "dry_run": False})
        daily = ledger.daily_report(days=30)
        assert daily[0]["total_tokens"] == 300

    @pytest.mark.unit
    def test_dry_runs_counted(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini", dry_run=True)
        ledger.record(_make_tracker(), model="gpt-4o-mini", dry_run=False)
        daily = ledger.daily_report(days=30)
        assert daily[0]["dry_runs"] == 1

    @pytest.mark.unit
    def test_sorted_newest_first(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": _days_ago(5), "cost_usd": 0.001, "total_tokens": 0, "total_calls": 1, "dry_run": False})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.002, "total_tokens": 0, "total_calls": 1, "dry_run": False})
        daily = ledger.daily_report(days=30)
        assert daily[0]["date"] > daily[-1]["date"]

    @pytest.mark.unit
    def test_multi_day_produces_multiple_rows(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": _days_ago(1), "cost_usd": 0.001, "total_tokens": 0, "total_calls": 1, "dry_run": False})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.002, "total_tokens": 0, "total_calls": 1, "dry_run": False})
        daily = ledger.daily_report(days=30)
        assert len(daily) == 2


# ===========================================================================
# CostLedger.weekly_summary
# ===========================================================================


class TestWeeklySummary:
    @pytest.mark.unit
    def test_empty_when_no_data(self, tmp_path):
        ledger = CostLedger(tmp_path)
        assert ledger.weekly_summary() == []

    @pytest.mark.unit
    def test_groups_by_iso_week(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(cost_per_call=0.001), model="gpt-4o-mini")
        ledger.record(_make_tracker(cost_per_call=0.002), model="gpt-4o-mini")
        weekly = ledger.weekly_summary(weeks=4)
        assert len(weekly) == 1
        assert weekly[0]["runs"] == 2

    @pytest.mark.unit
    def test_cost_summed_per_week(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.001, "total_tokens": 100})
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.003, "total_tokens": 200})
        weekly = ledger.weekly_summary(weeks=4)
        assert abs(weekly[0]["cost_usd"] - 0.004) < 1e-9

    @pytest.mark.unit
    def test_week_key_format(self, tmp_path):
        ledger = CostLedger(tmp_path)
        ledger.record(_make_tracker(), model="gpt-4o-mini")
        weekly = ledger.weekly_summary(weeks=4)
        assert len(weekly) == 1
        # Week key should be YYYY-Www
        week_key = weekly[0]["week"]
        parts = week_key.split("-W")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()


# ===========================================================================
# CostLedger.total_spend
# ===========================================================================


class TestTotalSpend:
    @pytest.mark.unit
    def test_zero_with_no_data(self, tmp_path):
        ledger = CostLedger(tmp_path)
        assert ledger.total_spend() == 0.0

    @pytest.mark.unit
    def test_sums_across_days(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.001})
        _write_raw_entry(ledger, {"date": _days_ago(2), "cost_usd": 0.002})
        total = ledger.total_spend(days=30)
        assert abs(total - 0.003) < 1e-9

    @pytest.mark.unit
    def test_excludes_old_entries(self, tmp_path):
        ledger = CostLedger(tmp_path)
        _write_raw_entry(ledger, {"date": date.today().isoformat(), "cost_usd": 0.001})
        _write_raw_entry(ledger, {"date": _days_ago(60), "cost_usd": 999.0})
        total = ledger.total_spend(days=7)
        assert abs(total - 0.001) < 1e-9
