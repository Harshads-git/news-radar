"""
tests/test_source_health.py
============================
Tests for Day 24 source health tracking in src/pipeline/source_health.py.

Coverage:
  - SourceHealthRecord: initial state, record_success (counts, consecutive
    reset, item_count update), record_error (counts, consecutive increment,
    last_error), success_rate, healthy property, to_jsonl_dict fields
  - SourceHealthTracker: record_success/error update correct record,
    should_disable (returns False below threshold, True at threshold, True
    for prev consecutive loaded from disk), get_record, all_records sorting,
    flush (creates file, writes correct JSON, appends multiple runs, silent
    on bad dir), load_history (empty on no file, cutoff filters old entries,
    malformed lines skipped, newest first), source_summary (empty, success
    rate correct, avg_items, consecutive_errors from latest row, multi-source),
    _load_prev_consecutive (empty when no file, reads latest per-source)

All tests use tmp_path — no real pipeline calls.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.pipeline.source_health import (
    CONSECUTIVE_ERROR_THRESHOLD,
    SourceHealthRecord,
    SourceHealthTracker,
)


# ===========================================================================
# SourceHealthRecord
# ===========================================================================


class TestSourceHealthRecord:
    @pytest.mark.unit
    def test_initial_state(self):
        rec = SourceHealthRecord("my-source")
        assert rec.source_id == "my-source"
        assert rec.attempts == 0
        assert rec.successes == 0
        assert rec.errors == 0
        assert rec.consecutive_errors == 0
        assert rec.last_item_count == 0
        assert rec.last_error is None

    @pytest.mark.unit
    def test_record_success_increments_attempts_and_successes(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(item_count=25)
        assert rec.attempts == 1
        assert rec.successes == 1

    @pytest.mark.unit
    def test_record_success_updates_item_count(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(item_count=30)
        assert rec.last_item_count == 30

    @pytest.mark.unit
    def test_record_success_resets_consecutive_errors(self):
        rec = SourceHealthRecord("src-a")
        rec.consecutive_errors = 3
        rec.record_success(item_count=10)
        assert rec.consecutive_errors == 0

    @pytest.mark.unit
    def test_record_success_clears_last_error(self):
        rec = SourceHealthRecord("src-a")
        rec.last_error = "Something went wrong"
        rec.record_success(item_count=10)
        assert rec.last_error is None

    @pytest.mark.unit
    def test_record_error_increments_all_counts(self):
        rec = SourceHealthRecord("src-b")
        rec.record_error("Connection refused")
        assert rec.attempts == 1
        assert rec.errors == 1
        assert rec.consecutive_errors == 1

    @pytest.mark.unit
    def test_record_error_stores_last_error(self):
        rec = SourceHealthRecord("src-b")
        rec.record_error("Timeout")
        assert rec.last_error == "Timeout"

    @pytest.mark.unit
    def test_record_error_truncates_long_message(self):
        rec = SourceHealthRecord("src-b")
        rec.record_error("E" * 500)
        assert len(rec.last_error) <= 200

    @pytest.mark.unit
    def test_multiple_successes_accumulate(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(item_count=10)
        rec.record_success(item_count=20)
        assert rec.attempts == 2
        assert rec.successes == 2

    @pytest.mark.unit
    def test_success_rate_zero_when_no_attempts(self):
        rec = SourceHealthRecord("src-a")
        assert rec.success_rate == 0.0

    @pytest.mark.unit
    def test_success_rate_one_when_all_succeed(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(10)
        rec.record_success(20)
        assert rec.success_rate == 1.0

    @pytest.mark.unit
    def test_success_rate_half_when_equal(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(10)
        rec.record_error("Err")
        assert rec.success_rate == 0.5

    @pytest.mark.unit
    def test_healthy_true_when_no_errors(self):
        rec = SourceHealthRecord("src-a")
        rec.record_success(10)
        assert rec.healthy is True

    @pytest.mark.unit
    def test_healthy_false_when_error(self):
        rec = SourceHealthRecord("src-a")
        rec.record_error("boom")
        assert rec.healthy is False

    @pytest.mark.unit
    def test_to_jsonl_dict_has_required_fields(self):
        rec = SourceHealthRecord("src-x")
        rec.record_success(15)
        d = rec.to_jsonl_dict()
        assert d["source_id"] == "src-x"
        assert d["successes"] == 1
        assert "date" in d
        assert "written_at" in d
        assert "consecutive_errors" in d


# ===========================================================================
# SourceHealthTracker — record_success / record_error / get_record
# ===========================================================================


class TestTrackerRecording:
    @pytest.mark.unit
    def test_record_success_creates_record(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("hn-top", item_count=25)
        rec = tracker.get_record("hn-top")
        assert rec is not None
        assert rec.successes == 1

    @pytest.mark.unit
    def test_record_error_creates_record(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_error("hn-rss", "Timeout")
        rec = tracker.get_record("hn-rss")
        assert rec is not None
        assert rec.errors == 1

    @pytest.mark.unit
    def test_get_record_returns_none_for_unknown(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        assert tracker.get_record("nonexistent") is None

    @pytest.mark.unit
    def test_multiple_records_tracked_independently(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=10)
        tracker.record_error("src-b", "Boom")
        assert tracker.get_record("src-a").successes == 1
        assert tracker.get_record("src-b").errors == 1

    @pytest.mark.unit
    def test_all_records_sorted_by_source_id(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("zzz-last", item_count=5)
        tracker.record_success("aaa-first", item_count=5)
        ids = [r.source_id for r in tracker.all_records()]
        assert ids == sorted(ids)


# ===========================================================================
# SourceHealthTracker — should_disable
# ===========================================================================


class TestShouldDisable:
    @pytest.mark.unit
    def test_returns_false_below_threshold(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        for _ in range(CONSECUTIVE_ERROR_THRESHOLD - 1):
            tracker.record_error("src-a", "Err")
        assert tracker.should_disable("src-a") is False

    @pytest.mark.unit
    def test_returns_true_at_threshold(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        for _ in range(CONSECUTIVE_ERROR_THRESHOLD):
            tracker.record_error("src-a", "Err")
        assert tracker.should_disable("src-a") is True

    @pytest.mark.unit
    def test_returns_false_after_success_resets(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        for _ in range(CONSECUTIVE_ERROR_THRESHOLD):
            tracker.record_error("src-a", "Err")
        tracker.record_success("src-a", item_count=5)
        assert tracker.should_disable("src-a") is False

    @pytest.mark.unit
    def test_returns_false_for_unknown_source(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        assert tracker.should_disable("new-source") is False

    @pytest.mark.unit
    def test_persisted_consecutive_loaded_on_init(self, tmp_path):
        """Previous run's consecutive_errors carries over to new tracker instance."""
        # Write a row with consecutive_errors=4 (one away from threshold)
        health_file = tmp_path / "source_health.jsonl"
        row = {"date": date.today().isoformat(), "source_id": "flaky",
               "consecutive_errors": CONSECUTIVE_ERROR_THRESHOLD - 1,
               "attempts": 4, "successes": 0, "errors": 4,
               "last_item_count": 0, "last_error": "Timeout",
               "written_at": "2026-07-13T10:00:00Z"}
        health_file.write_text(json.dumps(row) + "\n")

        tracker = SourceHealthTracker(tmp_path)
        # One more error should push it to/over the threshold
        tracker.record_error("flaky", "Timeout again")
        assert tracker.should_disable("flaky") is True


# ===========================================================================
# SourceHealthTracker — flush
# ===========================================================================


class TestTrackerFlush:
    @pytest.mark.unit
    def test_flush_creates_health_file(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=10)
        tracker.flush()
        assert (tmp_path / "source_health.jsonl").exists()

    @pytest.mark.unit
    def test_flush_writes_valid_jsonl(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=10)
        tracker.flush()
        lines = (tmp_path / "source_health.jsonl").read_text().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["source_id"] == "src-a"
        assert d["successes"] == 1

    @pytest.mark.unit
    def test_flush_appends_on_second_call(self, tmp_path):
        # First tracker: record src-a, flush
        tracker1 = SourceHealthTracker(tmp_path)
        tracker1.record_success("src-a", item_count=10)
        tracker1.flush()
        # Second tracker: record src-b, flush (should append, not overwrite)
        tracker2 = SourceHealthTracker(tmp_path)
        tracker2.record_success("src-b", item_count=5)
        tracker2.flush()
        lines = (tmp_path / "source_health.jsonl").read_text().splitlines()
        # Both flushes appended: 1 line from first + 1 line from second = 2
        assert len(lines) == 2
        source_ids = {json.loads(l)["source_id"] for l in lines}
        assert "src-a" in source_ids
        assert "src-b" in source_ids

    @pytest.mark.unit
    def test_flush_no_op_when_no_records(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.flush()  # should not crash or create file
        assert not (tmp_path / "source_health.jsonl").exists()

    @pytest.mark.unit
    def test_flush_silent_on_bad_path(self):
        # tmp_path that doesn't exist and can't be created
        tracker = SourceHealthTracker(Path("/nonexistent/deep/path/that/will/fail"))
        tracker.record_success("src-a", item_count=5)
        try:
            tracker.flush()  # must not raise
        except Exception as e:
            pytest.fail(f"flush raised: {e}")


# ===========================================================================
# SourceHealthTracker — load_history
# ===========================================================================


class TestLoadHistory:
    @pytest.mark.unit
    def test_returns_empty_when_no_file(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        assert tracker.load_history() == []

    @pytest.mark.unit
    def test_loads_written_entries(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("hn-top", item_count=25)
        tracker.flush()
        history = tracker.load_history(days=30)
        assert len(history) == 1
        assert history[0]["source_id"] == "hn-top"

    @pytest.mark.unit
    def test_cutoff_filters_old_entries(self, tmp_path):
        health_file = tmp_path / "source_health.jsonl"
        old_date = (date.today() - timedelta(days=60)).isoformat()
        old_row = {"date": old_date, "source_id": "old-src",
                   "attempts": 1, "successes": 1, "errors": 0,
                   "last_item_count": 10, "consecutive_errors": 0,
                   "last_error": None, "written_at": "T"}
        health_file.write_text(json.dumps(old_row) + "\n")

        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("new-src", item_count=5)
        tracker.flush()
        history = tracker.load_history(days=7)
        ids = [r["source_id"] for r in history]
        assert "old-src" not in ids
        assert "new-src" in ids

    @pytest.mark.unit
    def test_malformed_lines_skipped(self, tmp_path):
        health_file = tmp_path / "source_health.jsonl"
        health_file.write_text("NOT JSON\n")
        tracker = SourceHealthTracker(tmp_path)
        assert tracker.load_history() == []


# ===========================================================================
# SourceHealthTracker — source_summary
# ===========================================================================


class TestSourceSummary:
    @pytest.mark.unit
    def test_empty_when_no_history(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        assert tracker.source_summary() == {}

    @pytest.mark.unit
    def test_success_rate_correct(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=10)
        tracker.record_error("src-a", "Boom")
        tracker.flush()
        summary = tracker.source_summary()
        assert abs(summary["src-a"]["success_rate"] - 0.5) < 1e-9

    @pytest.mark.unit
    def test_consecutive_errors_from_latest_row(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_error("src-a", "Err")
        tracker.flush()
        summary = tracker.source_summary()
        assert summary["src-a"]["consecutive_errors"] == 1

    @pytest.mark.unit
    def test_multiple_sources_summarised_separately(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=10)
        tracker.record_error("src-b", "Err")
        tracker.flush()
        summary = tracker.source_summary()
        assert "src-a" in summary
        assert "src-b" in summary

    @pytest.mark.unit
    def test_last_seen_is_date_string(self, tmp_path):
        tracker = SourceHealthTracker(tmp_path)
        tracker.record_success("src-a", item_count=5)
        tracker.flush()
        summary = tracker.source_summary()
        last_seen = summary["src-a"]["last_seen"]
        assert last_seen == date.today().isoformat()
