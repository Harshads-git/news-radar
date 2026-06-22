"""
tests/test_config_sources.py
=============================
Unit tests for:
  - Settings (config.py): field defaults, validation, properties
  - sources_loader.py: load_sources(), validate_sources_file(), get_sources_by_type()

Tests use temp files for sources.json to avoid depending on the real file.
Settings tests override env vars via monkeypatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.config import Settings
from src.exceptions import ConfigError


# ===========================================================================
# Settings Tests
# ===========================================================================


class TestSettingsDefaults:
    @pytest.mark.unit
    def test_default_ai_model(self):
        s = Settings()
        assert s.ai_model == "gpt-4o-mini"

    @pytest.mark.unit
    def test_default_score_threshold(self):
        s = Settings()
        assert s.score_threshold == 6

    @pytest.mark.unit
    def test_default_max_briefing_items(self):
        s = Settings()
        assert s.max_briefing_items == 20

    @pytest.mark.unit
    def test_default_user_interests_is_non_empty(self):
        s = Settings()
        assert s.user_interests
        assert len(s.user_interests) > 10

    @pytest.mark.unit
    def test_default_output_language(self):
        s = Settings()
        assert s.output_language == "English"

    @pytest.mark.unit
    def test_default_docs_dir(self):
        s = Settings()
        assert str(s.docs_dir) == "docs"

    @pytest.mark.unit
    def test_default_github_pages_enabled(self):
        s = Settings()
        assert s.github_pages_enabled is True

    @pytest.mark.unit
    def test_default_sources_file(self):
        s = Settings()
        assert "sources.json" in str(s.sources_file)

    @pytest.mark.unit
    def test_default_data_dir(self):
        s = Settings()
        assert str(s.data_dir) == "data"


class TestSettingsProperties:
    @pytest.mark.unit
    def test_briefings_dir_is_data_briefings(self):
        s = Settings()
        assert s.briefings_dir == Path("data") / "briefings"

    @pytest.mark.unit
    def test_cache_dir_is_data_cache(self):
        s = Settings()
        assert s.cache_dir == Path("data") / "cache"

    @pytest.mark.unit
    def test_has_openai_false_when_no_key(self):
        s = Settings(openai_api_key="")
        assert not s.has_openai

    @pytest.mark.unit
    def test_has_openai_true_when_key_set(self):
        s = Settings(openai_api_key="sk-test-1234")
        assert s.has_openai

    @pytest.mark.unit
    def test_has_gemini_false_when_no_key(self):
        s = Settings(gemini_api_key="")
        assert not s.has_gemini

    @pytest.mark.unit
    def test_has_gemini_true_when_key_set(self):
        s = Settings(gemini_api_key="AIza-test")
        assert s.has_gemini

    @pytest.mark.unit
    def test_has_anthropic_false_when_no_key(self):
        s = Settings(anthropic_api_key="")
        assert not s.has_anthropic

    @pytest.mark.unit
    def test_has_any_ai_key_false_when_all_empty(self):
        s = Settings(openai_api_key="", gemini_api_key="", anthropic_api_key="")
        assert not s.has_any_ai_key

    @pytest.mark.unit
    def test_has_any_ai_key_true_when_one_set(self):
        s = Settings(gemini_api_key="AIza-test")
        assert s.has_any_ai_key

    @pytest.mark.unit
    def test_has_email_false_when_not_configured(self):
        s = Settings()
        assert not s.has_email

    @pytest.mark.unit
    def test_has_email_true_when_all_configured(self):
        s = Settings(smtp_user="a@b.com", smtp_password="pw", email_to="c@d.com")
        assert s.has_email

    @pytest.mark.unit
    def test_has_discord_false_when_no_url(self):
        s = Settings(discord_webhook_url="")
        assert not s.has_discord

    @pytest.mark.unit
    def test_has_discord_true_when_url_set(self):
        s = Settings(discord_webhook_url="https://discord.com/api/webhooks/x/y")
        assert s.has_discord


class TestSettingsActiveModelProvider:
    @pytest.mark.unit
    def test_gpt_model_returns_openai(self):
        s = Settings(ai_model="gpt-4o-mini")
        assert s.active_model_provider == "openai"

    @pytest.mark.unit
    def test_o1_model_returns_openai(self):
        s = Settings(ai_model="o1-mini")
        assert s.active_model_provider == "openai"

    @pytest.mark.unit
    def test_gemini_model_returns_gemini(self):
        s = Settings(ai_model="gemini-1.5-flash")
        assert s.active_model_provider == "gemini"

    @pytest.mark.unit
    def test_claude_model_returns_anthropic(self):
        s = Settings(ai_model="claude-3-5-haiku-20241022")
        assert s.active_model_provider == "anthropic"

    @pytest.mark.unit
    def test_unknown_model_returns_unknown(self):
        s = Settings(ai_model="some-random-model")
        assert s.active_model_provider == "unknown"


class TestSettingsValidateAiConfig:
    @pytest.mark.unit
    def test_no_warning_when_key_matches_model(self):
        s = Settings(ai_model="gpt-4o-mini", openai_api_key="sk-test")
        warnings = s.validate_ai_config()
        assert warnings == []

    @pytest.mark.unit
    def test_warning_when_openai_model_but_no_key(self):
        s = Settings(ai_model="gpt-4o-mini", openai_api_key="")
        warnings = s.validate_ai_config()
        assert len(warnings) == 1
        assert "OPENAI_API_KEY" in warnings[0]

    @pytest.mark.unit
    def test_warning_when_gemini_model_but_no_key(self):
        s = Settings(ai_model="gemini-1.5-flash", gemini_api_key="")
        warnings = s.validate_ai_config()
        assert len(warnings) == 1
        assert "GEMINI_API_KEY" in warnings[0]

    @pytest.mark.unit
    def test_warning_when_anthropic_model_but_no_key(self):
        s = Settings(ai_model="claude-3-5-haiku-20241022", anthropic_api_key="")
        warnings = s.validate_ai_config()
        assert len(warnings) == 1
        assert "ANTHROPIC_API_KEY" in warnings[0]

    @pytest.mark.unit
    def test_no_warning_for_unknown_provider(self):
        s = Settings(ai_model="some-random-model")
        warnings = s.validate_ai_config()
        assert warnings == []


class TestSettingsScoreThresholdClamping:
    @pytest.mark.unit
    def test_threshold_clamped_at_max_10(self):
        s = Settings(score_threshold=15)
        assert s.score_threshold == 10

    @pytest.mark.unit
    def test_threshold_clamped_at_min_0(self):
        s = Settings(score_threshold=-5)
        assert s.score_threshold == 0

    @pytest.mark.unit
    def test_valid_threshold_unchanged(self):
        s = Settings(score_threshold=7)
        assert s.score_threshold == 7


# ===========================================================================
# sources_loader Tests
# ===========================================================================


def _write_sources(tmp_path: Path, data: dict) -> Path:
    """Write a sources.json to tmp_path and return the path."""
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


_MINIMAL_SOURCE = {
    "id": "hn-test",
    "type": "hackernews",
    "name": "HN Test",
    "enabled": True,
    "limit": 10,
    "tags": [],
}

_RSS_SOURCE = {
    "id": "rss-test",
    "type": "rss",
    "name": "RSS Test",
    "url": "https://example.com/feed.xml",
    "enabled": True,
    "limit": 10,
    "tags": [],
}

_REDDIT_SOURCE = {
    "id": "r-test",
    "type": "reddit",
    "name": "Reddit Test",
    "subreddit": "programming",
    "enabled": True,
    "limit": 10,
    "tags": [],
}


class TestLoadSources:
    @pytest.mark.unit
    def test_load_returns_sources_config(self, tmp_path):
        from src.setup.sources_loader import load_sources
        from src.models import SourcesConfig
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE]})
        cfg = load_sources(path)
        assert isinstance(cfg, SourcesConfig)

    @pytest.mark.unit
    def test_load_counts_sources_correctly(self, tmp_path):
        from src.setup.sources_loader import load_sources
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE, _RSS_SOURCE]})
        cfg = load_sources(path)
        assert len(cfg.sources) == 2

    @pytest.mark.unit
    def test_enabled_sources_excludes_disabled(self, tmp_path):
        from src.setup.sources_loader import load_sources
        disabled = {**_MINIMAL_SOURCE, "id": "disabled-hn", "enabled": False}
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE, disabled]})
        cfg = load_sources(path)
        assert len(cfg.enabled_sources) == 1

    @pytest.mark.unit
    def test_missing_file_raises_config_error(self, tmp_path):
        from src.setup.sources_loader import load_sources
        with pytest.raises(ConfigError, match="not found"):
            load_sources(tmp_path / "nonexistent.json")

    @pytest.mark.unit
    def test_invalid_json_raises_config_error(self, tmp_path):
        from src.setup.sources_loader import load_sources
        path = tmp_path / "sources.json"
        path.write_text("NOT JSON {{{", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_sources(path)

    @pytest.mark.unit
    def test_invalid_source_type_raises_config_error(self, tmp_path):
        from src.setup.sources_loader import load_sources
        bad = {**_MINIMAL_SOURCE, "type": "unsupported_type"}
        path = _write_sources(tmp_path, {"sources": [bad]})
        with pytest.raises(ConfigError, match="validation"):
            load_sources(path)

    @pytest.mark.unit
    def test_no_enabled_sources_logs_warning_by_default(self, tmp_path):
        from src.setup.sources_loader import load_sources
        disabled = {**_MINIMAL_SOURCE, "enabled": False}
        path = _write_sources(tmp_path, {"sources": [disabled]})
        # Should NOT raise — just warn
        cfg = load_sources(path)
        assert cfg.enabled_sources == []

    @pytest.mark.unit
    def test_no_enabled_sources_raises_when_required(self, tmp_path):
        from src.setup.sources_loader import load_sources
        disabled = {**_MINIMAL_SOURCE, "enabled": False}
        path = _write_sources(tmp_path, {"sources": [disabled]})
        with pytest.raises(ConfigError, match="No sources are enabled"):
            load_sources(path, require_enabled=True)

    @pytest.mark.unit
    def test_loads_real_sources_file(self):
        """Integration: real data/sources.json must load without errors."""
        from src.setup.sources_loader import load_sources
        cfg = load_sources("data/sources.json")
        assert len(cfg.sources) > 0
        assert len(cfg.enabled_sources) > 0


class TestGetSourcesByType:
    @pytest.mark.unit
    def test_filters_by_rss(self, tmp_path):
        from src.setup.sources_loader import load_sources, get_sources_by_type
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE, _RSS_SOURCE, _REDDIT_SOURCE]})
        cfg = load_sources(path)
        rss = get_sources_by_type(cfg, "rss")
        assert len(rss) == 1
        assert rss[0].type == "rss"

    @pytest.mark.unit
    def test_filters_by_hackernews(self, tmp_path):
        from src.setup.sources_loader import load_sources, get_sources_by_type
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE, _RSS_SOURCE]})
        cfg = load_sources(path)
        hn = get_sources_by_type(cfg, "hackernews")
        assert len(hn) == 1
        assert hn[0].type == "hackernews"

    @pytest.mark.unit
    def test_excludes_disabled_when_filtering(self, tmp_path):
        from src.setup.sources_loader import load_sources, get_sources_by_type
        disabled_rss = {**_RSS_SOURCE, "id": "disabled-rss", "enabled": False}
        path = _write_sources(tmp_path, {"sources": [_RSS_SOURCE, disabled_rss]})
        cfg = load_sources(path)
        rss = get_sources_by_type(cfg, "rss")
        assert len(rss) == 1  # only enabled one

    @pytest.mark.unit
    def test_unknown_type_returns_empty(self, tmp_path):
        from src.setup.sources_loader import load_sources, get_sources_by_type
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE]})
        cfg = load_sources(path)
        result = get_sources_by_type(cfg, "github")
        assert result == []


class TestValidateSourcesFile:
    @pytest.mark.unit
    def test_valid_file_returns_no_issues(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        path = _write_sources(tmp_path, {"sources": [_MINIMAL_SOURCE]})
        issues = validate_sources_file(path)
        assert issues == []

    @pytest.mark.unit
    def test_missing_file_returns_issue(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        issues = validate_sources_file(tmp_path / "missing.json")
        assert len(issues) == 1
        assert "not found" in issues[0]

    @pytest.mark.unit
    def test_invalid_json_returns_issue(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        path = tmp_path / "sources.json"
        path.write_text("BAD JSON", encoding="utf-8")
        issues = validate_sources_file(path)
        assert len(issues) == 1
        assert "Invalid JSON" in issues[0]

    @pytest.mark.unit
    def test_missing_sources_key_returns_issue(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        path = _write_sources(tmp_path, {"notSources": []})
        issues = validate_sources_file(path)
        assert any("sources" in i for i in issues)

    @pytest.mark.unit
    def test_bad_source_returns_specific_issue(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        bad = {**_MINIMAL_SOURCE, "type": "bad_type"}
        path = _write_sources(tmp_path, {"sources": [bad]})
        issues = validate_sources_file(path)
        assert len(issues) == 1
        assert "hn-test" in issues[0]

    @pytest.mark.unit
    def test_no_enabled_sources_warns(self, tmp_path):
        from src.setup.sources_loader import validate_sources_file
        disabled = {**_MINIMAL_SOURCE, "enabled": False}
        path = _write_sources(tmp_path, {"sources": [disabled]})
        issues = validate_sources_file(path)
        assert any("no sources are enabled" in i.lower() for i in issues)

    @pytest.mark.unit
    def test_real_sources_file_has_no_issues(self):
        """Integration: real sources.json should have zero issues."""
        from src.setup.sources_loader import validate_sources_file
        issues = validate_sources_file("data/sources.json")
        assert issues == []
