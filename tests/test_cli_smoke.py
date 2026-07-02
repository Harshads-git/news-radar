"""
tests/test_cli_smoke.py
=======================
Smoke tests for the CLI entry point (src/main.py) and supporting modules.

These tests verify:
  1. The CLI parser accepts all documented flags without crashing
  2. --version prints the version string
  3. --status runs without error (uses default config)
  4. --dry-run fails gracefully when no API key is set
  5. The exception hierarchy is importable and well-formed
  6. The logger is importable and functional

All tests in this file are marked @pytest.mark.unit because they use no
real network I/O — they only test the CLI argument parsing and module
imports.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper: run the CLI as a subprocess so we test the full entry point
# ---------------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m src.main <args>` and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "src.main", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,  # prevent wizard from reading TTY
        cwd=".",  # project root
    )


# ---------------------------------------------------------------------------
# CLI Smoke Tests
# ---------------------------------------------------------------------------


class TestCLIParser:
    """Test that the argument parser handles all flags correctly."""

    @pytest.mark.unit
    def test_version_flag_prints_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "news-radar" in result.stdout
        # Should contain a version number like 0.1.0
        assert any(char.isdigit() for char in result.stdout)

    @pytest.mark.unit
    def test_help_flag_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "--run" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--setup" in result.stdout
        assert "--status" in result.stdout

    @pytest.mark.unit
    def test_no_args_exits_nonzero(self):
        """Running with no args should print usage error and exit non-zero."""
        result = run_cli()
        assert result.returncode != 0

    @pytest.mark.unit
    def test_status_flag_exits_zero(self):
        """--status should always succeed (reads config and prints a table)."""
        result = run_cli("--status")
        assert result.returncode == 0

    @pytest.mark.unit
    def test_status_shows_config_table(self):
        """--status output should contain key config field names."""
        result = run_cli("--status")
        output = result.stdout + result.stderr
        assert "AI Model" in output or "gpt-4o-mini" in output

    @pytest.mark.unit
    def test_setup_flag_exits_zero(self):
        """--setup should exit 0 (wizard placeholder for now)."""
        result = run_cli("--setup")
        assert result.returncode == 0

    @pytest.mark.unit
    def test_dry_run_without_api_key_exits_nonzero(self):
        """
        --dry-run with no API key should exit 1 with a clear error message.
        This validates the config check in _handle_run().
        """
        result = run_cli("--dry-run")
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        # Should mention API key configuration
        assert "API" in combined or "key" in combined.lower() or "error" in combined.lower()

    @pytest.mark.unit
    def test_mutually_exclusive_flags_rejected(self):
        """Providing two action flags at once should fail."""
        result = run_cli("--run", "--setup")
        assert result.returncode != 0

    @pytest.mark.unit
    def test_log_level_flag_accepted(self):
        """--status --log-level DEBUG should work fine."""
        result = run_cli("--status", "--log-level", "DEBUG")
        assert result.returncode == 0

    @pytest.mark.unit
    def test_invalid_log_level_rejected(self):
        """--log-level INVALID should fail with argparse error."""
        result = run_cli("--status", "--log-level", "VERBOSE")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Exception Hierarchy Tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the exception hierarchy is well-formed and importable."""

    @pytest.mark.unit
    def test_all_exceptions_importable(self):
        from src.exceptions import (
            AIError,
            AIProviderError,
            ConfigError,
            DeliveryError,
            FetchError,
            NewsRadarError,
            ParseError,
            RateLimitError,
            StorageError,
            TokenLimitError,
        )
        # All are subclasses of NewsRadarError
        for exc_class in [
            FetchError, RateLimitError, ParseError,
            AIError, TokenLimitError, AIProviderError,
            StorageError, ConfigError, DeliveryError,
        ]:
            assert issubclass(exc_class, NewsRadarError), \
                f"{exc_class.__name__} must inherit from NewsRadarError"

    @pytest.mark.unit
    def test_fetch_error_context_in_str(self):
        from src.exceptions import FetchError
        e = FetchError("Request failed", source_id="rss-1", url="https://ex.com")
        s = str(e)
        assert "rss-1" in s
        assert "https://ex.com" in s

    @pytest.mark.unit
    def test_rate_limit_error_has_retry_after(self):
        from src.exceptions import RateLimitError
        e = RateLimitError("429 Too Many Requests", source_id="reddit", retry_after=45)
        assert e.retry_after == 45
        assert e.status_code == 429

    @pytest.mark.unit
    def test_parse_error_truncates_raw_data(self):
        from src.exceptions import ParseError
        long_data = "x" * 1000
        e = ParseError("Bad XML", source_id="rss-1", raw_data=long_data)
        assert len(e.raw_data) <= 200

    @pytest.mark.unit
    def test_ai_error_carries_model(self):
        from src.exceptions import AIError
        e = AIError("API timeout", model="gpt-4o-mini", item_url="https://ex.com")
        assert e.model == "gpt-4o-mini"
        assert "gpt-4o-mini" in str(e)

    @pytest.mark.unit
    def test_config_error_carries_field(self):
        from src.exceptions import ConfigError
        e = ConfigError("Missing key", field="OPENAI_API_KEY", expected="sk-...")
        assert e.field == "OPENAI_API_KEY"

    @pytest.mark.unit
    def test_storage_error_carries_operation(self):
        from src.exceptions import StorageError
        e = StorageError("Disk full", path="/data/briefings/2026-06-15.json", operation="write")
        assert e.operation == "write"

    @pytest.mark.unit
    def test_delivery_error_carries_channel(self):
        from src.exceptions import DeliveryError
        e = DeliveryError("Webhook failed", channel="discord")
        assert e.channel == "discord"

    @pytest.mark.unit
    def test_exceptions_are_catchable_as_base(self):
        """All exceptions must be catchable as NewsRadarError."""
        from src.exceptions import FetchError, NewsRadarError, RateLimitError
        try:
            raise RateLimitError("too fast", source_id="hn", retry_after=60)
        except FetchError:
            pass  # caught as parent
        except Exception:
            pytest.fail("RateLimitError should be catchable as FetchError")

        try:
            raise FetchError("network down", source_id="reddit")
        except NewsRadarError:
            pass  # caught as base
        except Exception:
            pytest.fail("FetchError should be catchable as NewsRadarError")


# ---------------------------------------------------------------------------
# Logger Smoke Tests
# ---------------------------------------------------------------------------


class TestLogger:
    """Verify the logger module is functional."""

    @pytest.mark.unit
    def test_logger_importable(self):
        from src.logger import configure_logging, get_logger
        assert callable(configure_logging)
        assert callable(get_logger)

    @pytest.mark.unit
    def test_get_logger_returns_news_radar_logger(self):
        from src.logger import NewsRadarLogger, get_logger
        log = get_logger("test.module")
        assert isinstance(log, NewsRadarLogger)

    @pytest.mark.unit
    def test_logger_info_does_not_raise(self):
        from src.logger import get_logger
        log = get_logger("test.smoke")
        # Should not raise even with format args
        log.info("Test message %s", "arg1")

    @pytest.mark.unit
    def test_logger_warning_does_not_raise(self):
        from src.logger import get_logger
        log = get_logger("test.smoke")
        log.warning("Warning %d", 42)

    @pytest.mark.unit
    def test_logger_success_does_not_raise(self):
        from src.logger import get_logger
        log = get_logger("test.smoke")
        log.success("Operation completed in %.2fs", 1.23)

    @pytest.mark.unit
    def test_configure_logging_idempotent(self):
        """Calling configure_logging twice should not duplicate handlers."""
        import logging
        from src.logger import configure_logging
        before = len(logging.getLogger().handlers)
        configure_logging("INFO")
        configure_logging("DEBUG")  # second call should be no-op
        after = len(logging.getLogger().handlers)
        assert after == before
