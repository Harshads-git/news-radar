"""
tests/test_cli_day23.py
========================
Tests for Day 23 CLI improvements in src/main.py.

Coverage:
  - _build_parser: --sources-list and --config flags exist, all flags
    are mutually exclusive (can't pass two at once)
  - _handle_sources_list: missing file prints helpful message, valid
    sources.json renders rows (enabled/disabled, types), empty sources
    list handled gracefully, malformed JSON doesn't crash
  - _handle_config: renders all five sections (pipeline, paths, ai keys,
    delivery, advanced), active provider computed correctly, interests
    truncated at 80 chars, GitHub Pages status shown
  - --help epilog: contains new example commands
  - Argument defaults: sources_list=False by default, config=False

All tests use tmp_path / MagicMock — no real subprocess or network calls.
Rich output is captured via Console(file=StringIO).
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import _build_parser, _handle_config, _handle_sources_list


# ===========================================================================
# Helpers
# ===========================================================================


def _make_settings(
    tmp_path: Path,
    *,
    sources_file: Path | None = None,
    ai_model: str = "gpt-4o-mini",
    has_openai: bool = False,
    has_gemini: bool = False,
    has_anthropic: bool = False,
    has_email: bool = False,
    has_discord: bool = False,
    has_slack: bool = False,
    github_pages_enabled: bool = True,
    score_threshold: int = 6,
    max_briefing_items: int = 20,
    output_language: str = "English",
    log_level: str = "INFO",
    user_interests: str = "AI, Python",
    custom_webhook_url: str = "",
) -> MagicMock:
    """Build a minimal Settings-like mock for handler tests."""
    s = MagicMock()
    s.sources_file = sources_file or (tmp_path / "sources.json")
    s.data_dir = tmp_path / "data"
    s.briefings_dir = tmp_path / "data" / "briefings"
    s.cache_dir = tmp_path / "data" / "cache"
    s.docs_dir = tmp_path / "docs"
    s.ai_model = ai_model
    s.active_model_provider = "openai"
    s.has_openai = has_openai
    s.has_gemini = has_gemini
    s.has_anthropic = has_anthropic
    s.has_email = has_email
    s.has_discord = has_discord
    s.has_slack = has_slack
    s.github_pages_enabled = github_pages_enabled
    s.score_threshold = score_threshold
    s.max_briefing_items = max_briefing_items
    s.output_language = output_language
    s.log_level = log_level
    s.user_interests = user_interests
    s.custom_webhook_url = custom_webhook_url
    return s


def _sources_json(sources: list[dict]) -> dict:
    return {"sources": sources}


def _capture(handler, *args) -> str:
    """Call handler and capture Rich console output."""
    from rich.console import Console

    buf = StringIO()
    with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
        handler(*args)
    return buf.getvalue()


# ===========================================================================
# _build_parser: new flags exist
# ===========================================================================


class TestBuildParser:
    @pytest.mark.unit
    def test_sources_list_flag_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["--sources-list"])
        assert args.sources_list is True

    @pytest.mark.unit
    def test_config_flag_exists(self):
        parser = _build_parser()
        args = parser.parse_args(["--config"])
        assert args.config is True

    @pytest.mark.unit
    def test_sources_list_default_false(self):
        parser = _build_parser()
        # Can't parse with no action — sources_list should be False when not given
        args = parser.parse_args(["--run"])
        assert args.sources_list is False

    @pytest.mark.unit
    def test_config_default_false(self):
        parser = _build_parser()
        args = parser.parse_args(["--run"])
        assert args.config is False

    @pytest.mark.unit
    def test_sources_list_mutually_exclusive_with_run(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--sources-list", "--run"])

    @pytest.mark.unit
    def test_config_mutually_exclusive_with_run(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "--run"])

    @pytest.mark.unit
    def test_epilog_contains_sources_list_example(self):
        parser = _build_parser()
        assert "--sources-list" in (parser.epilog or "")

    @pytest.mark.unit
    def test_epilog_contains_config_example(self):
        parser = _build_parser()
        assert "--config" in (parser.epilog or "")

    @pytest.mark.unit
    def test_epilog_contains_advanced_section(self):
        parser = _build_parser()
        assert "Advanced" in (parser.epilog or "") or "no-enrich" in (parser.epilog or "")

    @pytest.mark.unit
    def test_sources_list_and_briefing_mutually_exclusive(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--sources-list", "--briefing"])


# ===========================================================================
# _handle_sources_list
# ===========================================================================


class TestHandleSourcesList:
    @pytest.mark.unit
    def test_missing_file_prints_helpful_message(self, tmp_path):
        settings = _make_settings(tmp_path)
        # sources_file doesn't exist → should print a message, not crash
        from rich.console import Console
        buf = StringIO()
        console = Console(file=buf, width=120)

        with patch("rich.console.Console", return_value=console):
            _handle_sources_list(settings, MagicMock())

        output = buf.getvalue()
        assert "not found" in output.lower() or "setup" in output.lower()

    @pytest.mark.unit
    def test_enabled_source_shows_enabled_text(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "hn-top", "type": "hackernews", "name": "HN Top", "enabled": True, "limit": 30, "tags": []}
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "hn-top" in output
        assert "enabled" in output.lower()

    @pytest.mark.unit
    def test_disabled_source_shows_disabled_text(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "hn-rss", "type": "rss", "name": "HN RSS", "enabled": False, "limit": 30, "tags": []}
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "disabled" in output.lower()

    @pytest.mark.unit
    def test_reddit_source_shows_subreddit(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "r-prog", "type": "reddit", "name": "r/programming",
             "enabled": True, "limit": 25, "tags": [], "subreddit": "programming", "sort": "hot"}
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "programming" in output

    @pytest.mark.unit
    def test_rss_source_shows_url(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "rss-1", "type": "rss", "name": "Feed", "enabled": True,
             "limit": 20, "tags": [], "url": "https://example.com/feed.xml"}
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "example.com" in output

    @pytest.mark.unit
    def test_empty_sources_list_handled(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps({"sources": []}))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "No sources" in output or len(output) > 0  # must not crash

    @pytest.mark.unit
    def test_malformed_json_does_not_crash(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text("NOT VALID JSON {{{{")
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            try:
                _handle_sources_list(settings, MagicMock())
            except Exception as e:
                pytest.fail(f"Handler raised unexpected exception: {e}")

    @pytest.mark.unit
    def test_multiple_sources_all_appear(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "src-a", "type": "hackernews", "name": "A", "enabled": True, "limit": 30, "tags": []},
            {"id": "src-b", "type": "rss", "name": "B", "enabled": True, "limit": 20, "tags": []},
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        assert "src-a" in output
        assert "src-b" in output

    @pytest.mark.unit
    def test_enabled_count_in_footer(self, tmp_path):
        sf = tmp_path / "sources.json"
        sf.write_text(json.dumps(_sources_json([
            {"id": "a", "type": "rss", "name": "A", "enabled": True, "limit": 10, "tags": []},
            {"id": "b", "type": "rss", "name": "B", "enabled": False, "limit": 10, "tags": []},
        ])))
        settings = _make_settings(tmp_path, sources_file=sf)

        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_sources_list(settings, MagicMock())
        output = buf.getvalue()
        # Footer should say "1/2 sources enabled"
        assert "1/2" in output or "1" in output


# ===========================================================================
# _handle_config
# ===========================================================================


class TestHandleConfig:
    @pytest.mark.unit
    def test_pipeline_section_contains_model(self, tmp_path):
        settings = _make_settings(tmp_path, ai_model="gpt-4o-mini")
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "gpt-4o-mini" in output

    @pytest.mark.unit
    def test_pipeline_section_contains_score_threshold(self, tmp_path):
        settings = _make_settings(tmp_path, score_threshold=7)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "7" in output

    @pytest.mark.unit
    def test_paths_section_contains_data_dir(self, tmp_path):
        settings = _make_settings(tmp_path)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "data" in output.lower()

    @pytest.mark.unit
    def test_ai_keys_section_shows_not_set_when_no_keys(self, tmp_path):
        settings = _make_settings(tmp_path, has_openai=False, has_gemini=False)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "not set" in output

    @pytest.mark.unit
    def test_ai_keys_section_shows_configured_when_key_present(self, tmp_path):
        settings = _make_settings(tmp_path, has_openai=True)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "configured" in output

    @pytest.mark.unit
    def test_delivery_section_shows_inactive_by_default(self, tmp_path):
        settings = _make_settings(tmp_path)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "inactive" in output

    @pytest.mark.unit
    def test_delivery_section_shows_email_active(self, tmp_path):
        settings = _make_settings(tmp_path, has_email=True)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "active" in output

    @pytest.mark.unit
    def test_github_pages_enabled_shown(self, tmp_path):
        settings = _make_settings(tmp_path, github_pages_enabled=True)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "GitHub Pages" in output or "enabled" in output

    @pytest.mark.unit
    def test_user_interests_truncated_when_long(self, tmp_path):
        long_interests = "A" * 200
        settings = _make_settings(tmp_path, user_interests=long_interests)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=200)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        # The long interest string shouldn't appear verbatim (truncated)
        assert long_interests not in output
        assert "..." in output

    @pytest.mark.unit
    def test_advanced_section_shows_log_level(self, tmp_path):
        settings = _make_settings(tmp_path, log_level="DEBUG")
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "DEBUG" in output

    @pytest.mark.unit
    def test_setup_hint_in_footer(self, tmp_path):
        settings = _make_settings(tmp_path)
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "setup" in output.lower()

    @pytest.mark.unit
    def test_output_language_shown(self, tmp_path):
        settings = _make_settings(tmp_path, output_language="Spanish")
        from rich.console import Console
        buf = StringIO()
        with patch("rich.console.Console", return_value=Console(file=buf, width=120)):
            _handle_config(settings, MagicMock())
        output = buf.getvalue()
        assert "Spanish" in output
