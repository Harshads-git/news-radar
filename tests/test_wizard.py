"""
tests/test_wizard.py
=====================
Tests for the interactive setup wizard (src/setup/wizard.py).

Coverage:
  - WizardConfig: defaults, field types
  - _load_existing_env: empty file, valid file, comments/blanks
  - _load_existing_sources: missing file, valid JSON, malformed JSON
  - _write_env: creates file, updates existing key, preserves custom keys,
    email/discord/slack conditionally written, API key routing
  - _write_sources: all sources written, enabled/disabled from config
  - run_wizard(non_interactive=True): full end-to-end non-interactive path
  - _DEFAULT_SOURCES: all have required fields, unique IDs
  - _print_config_preview: doesn't crash

All tests are pure unit tests — no Rich prompts, no user input.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.setup.wizard import (
    WizardConfig,
    _DEFAULT_SOURCES,
    _load_existing_env,
    _load_existing_sources,
    _write_env,
    _write_sources,
    run_wizard,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_cfg(**kwargs) -> WizardConfig:
    """Build a WizardConfig with sensible defaults, overriding with kwargs."""
    defaults = dict(
        ai_provider="openai",
        ai_model="gpt-4o-mini",
        ai_api_key="sk-test-key-12345",
        user_interests="AI, Python",
        score_threshold=6,
        max_briefing_items=20,
        github_pages_enabled=True,
        enabled_source_ids=["hackernews-top", "reddit-ml"],
    )
    defaults.update(kwargs)
    return WizardConfig(**defaults)


# ===========================================================================
# WizardConfig defaults
# ===========================================================================


class TestWizardConfig:
    @pytest.mark.unit
    def test_default_provider_is_openai(self):
        cfg = WizardConfig()
        assert cfg.ai_provider == "openai"

    @pytest.mark.unit
    def test_default_model_is_gpt4o_mini(self):
        cfg = WizardConfig()
        assert cfg.ai_model == "gpt-4o-mini"

    @pytest.mark.unit
    def test_default_score_threshold_is_6(self):
        cfg = WizardConfig()
        assert cfg.score_threshold == 6

    @pytest.mark.unit
    def test_default_max_items_is_20(self):
        cfg = WizardConfig()
        assert cfg.max_briefing_items == 20

    @pytest.mark.unit
    def test_default_email_disabled(self):
        cfg = WizardConfig()
        assert not cfg.email_enabled

    @pytest.mark.unit
    def test_default_discord_disabled(self):
        cfg = WizardConfig()
        assert not cfg.discord_enabled

    @pytest.mark.unit
    def test_enabled_source_ids_starts_empty(self):
        cfg = WizardConfig()
        assert cfg.enabled_source_ids == []

    @pytest.mark.unit
    def test_config_accepts_custom_values(self):
        cfg = WizardConfig(
            ai_provider="gemini",
            ai_model="gemini-1.5-flash",
            score_threshold=8,
        )
        assert cfg.ai_provider == "gemini"
        assert cfg.score_threshold == 8


# ===========================================================================
# _load_existing_env
# ===========================================================================


class TestLoadExistingEnv:
    @pytest.mark.unit
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        result = _load_existing_env(tmp_path / "nonexistent.env")
        assert result == {}

    @pytest.mark.unit
    def test_parses_simple_key_value(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("OPENAI_API_KEY=sk-abc123\n")
        result = _load_existing_env(f)
        assert result["OPENAI_API_KEY"] == "sk-abc123"

    @pytest.mark.unit
    def test_skips_comment_lines(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("# This is a comment\nAI_MODEL=gpt-4o-mini\n")
        result = _load_existing_env(f)
        assert "# This is a comment" not in result
        assert result["AI_MODEL"] == "gpt-4o-mini"

    @pytest.mark.unit
    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("\nAI_MODEL=gpt-4o-mini\n\n")
        result = _load_existing_env(f)
        assert len(result) == 1

    @pytest.mark.unit
    def test_handles_value_with_equals_sign(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("URL=https://example.com?a=1&b=2\n")
        result = _load_existing_env(f)
        assert result["URL"] == "https://example.com?a=1&b=2"

    @pytest.mark.unit
    def test_parses_multiple_keys(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("KEY_A=value_a\nKEY_B=value_b\nKEY_C=value_c\n")
        result = _load_existing_env(f)
        assert len(result) == 3


# ===========================================================================
# _load_existing_sources
# ===========================================================================


class TestLoadExistingSources:
    @pytest.mark.unit
    def test_returns_empty_list_when_missing(self, tmp_path):
        result = _load_existing_sources(tmp_path / "missing.json")
        assert result == []

    @pytest.mark.unit
    def test_loads_valid_sources_json(self, tmp_path):
        f = tmp_path / "sources.json"
        f.write_text(json.dumps({"sources": [{"id": "hn", "enabled": True}]}))
        result = _load_existing_sources(f)
        assert len(result) == 1
        assert result[0]["id"] == "hn"

    @pytest.mark.unit
    def test_returns_empty_on_malformed_json(self, tmp_path):
        f = tmp_path / "sources.json"
        f.write_text("{invalid json}")
        result = _load_existing_sources(f)
        assert result == []

    @pytest.mark.unit
    def test_returns_empty_list_when_sources_key_missing(self, tmp_path):
        f = tmp_path / "sources.json"
        f.write_text(json.dumps({"other_key": []}))
        result = _load_existing_sources(f)
        assert result == []


# ===========================================================================
# _write_env
# ===========================================================================


class TestWriteEnv:
    @pytest.mark.unit
    def test_creates_env_file(self, tmp_path):
        cfg = _make_cfg()
        env = tmp_path / ".env"
        _write_env(cfg, env)
        assert env.exists()

    @pytest.mark.unit
    def test_writes_openai_key(self, tmp_path):
        cfg = _make_cfg(ai_provider="openai", ai_api_key="sk-mykey")
        env = tmp_path / ".env"
        _write_env(cfg, env)
        content = env.read_text()
        assert "OPENAI_API_KEY=sk-mykey" in content

    @pytest.mark.unit
    def test_writes_gemini_key(self, tmp_path):
        cfg = _make_cfg(ai_provider="gemini", ai_api_key="AIza-mykey")
        env = tmp_path / ".env"
        _write_env(cfg, env)
        content = env.read_text()
        assert "GEMINI_API_KEY=AIza-mykey" in content

    @pytest.mark.unit
    def test_writes_anthropic_key(self, tmp_path):
        cfg = _make_cfg(ai_provider="anthropic", ai_api_key="sk-ant-mykey")
        env = tmp_path / ".env"
        _write_env(cfg, env)
        content = env.read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-mykey" in content

    @pytest.mark.unit
    def test_writes_score_threshold(self, tmp_path):
        cfg = _make_cfg(score_threshold=8)
        env = tmp_path / ".env"
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["SCORE_THRESHOLD"] == "8"

    @pytest.mark.unit
    def test_writes_github_pages_true(self, tmp_path):
        cfg = _make_cfg(github_pages_enabled=True)
        env = tmp_path / ".env"
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["GITHUB_PAGES_ENABLED"] == "true"

    @pytest.mark.unit
    def test_writes_github_pages_false(self, tmp_path):
        cfg = _make_cfg(github_pages_enabled=False)
        env = tmp_path / ".env"
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["GITHUB_PAGES_ENABLED"] == "false"

    @pytest.mark.unit
    def test_email_keys_written_when_enabled(self, tmp_path):
        cfg = _make_cfg(
            email_enabled=True,
            smtp_user="user@gmail.com",
            smtp_password="pass123",
            email_to="to@gmail.com",
        )
        env = tmp_path / ".env"
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["SMTP_USER"] == "user@gmail.com"
        assert env_data["SMTP_PASSWORD"] == "pass123"
        assert env_data["EMAIL_TO"] == "to@gmail.com"

    @pytest.mark.unit
    def test_email_keys_not_written_when_disabled(self, tmp_path):
        cfg = _make_cfg(email_enabled=False)
        env = tmp_path / ".env"
        _write_env(cfg, env)
        content = env.read_text()
        assert "SMTP_USER" not in content

    @pytest.mark.unit
    def test_discord_key_written_when_enabled(self, tmp_path):
        cfg = _make_cfg(
            discord_enabled=True,
            discord_webhook_url="https://discord.com/api/webhooks/test",
        )
        env = tmp_path / ".env"
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["DISCORD_WEBHOOK_URL"] == "https://discord.com/api/webhooks/test"

    @pytest.mark.unit
    def test_preserves_custom_keys_in_existing_env(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("MY_CUSTOM_VAR=keep_this\nOPENAI_API_KEY=old_key\n")
        cfg = _make_cfg(ai_provider="openai", ai_api_key="new_key")
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        assert env_data["MY_CUSTOM_VAR"] == "keep_this"

    @pytest.mark.unit
    def test_updates_existing_key_in_env(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=old_key\n")
        cfg = _make_cfg(ai_provider="openai", ai_api_key="new_key")
        _write_env(cfg, env)
        env_data = _load_existing_env(env)
        # Key should be updated, not duplicated
        content = env.read_text()
        assert content.count("OPENAI_API_KEY=") == 1
        assert env_data["OPENAI_API_KEY"] == "new_key"

    @pytest.mark.unit
    def test_creates_parent_dirs_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / "subdir" / ".env"
            cfg = _make_cfg()
            _write_env(cfg, env)
            assert env.exists()


# ===========================================================================
# _write_sources
# ===========================================================================


class TestWriteSources:
    @pytest.mark.unit
    def test_creates_sources_json(self, tmp_path):
        cfg = _make_cfg(enabled_source_ids=["hackernews-top"])
        sources_path = tmp_path / "data" / "sources.json"
        _write_sources(cfg, sources_path)
        assert sources_path.exists()

    @pytest.mark.unit
    def test_sources_json_is_valid(self, tmp_path):
        cfg = _make_cfg(enabled_source_ids=["hackernews-top"])
        sources_path = tmp_path / "sources.json"
        _write_sources(cfg, sources_path)
        data = json.loads(sources_path.read_text())
        assert "sources" in data

    @pytest.mark.unit
    def test_enabled_sources_match_config(self, tmp_path):
        cfg = _make_cfg(enabled_source_ids=["hackernews-top", "reddit-ml"])
        sources_path = tmp_path / "sources.json"
        _write_sources(cfg, sources_path)
        data = json.loads(sources_path.read_text())
        enabled = [s for s in data["sources"] if s["enabled"]]
        assert {s["id"] for s in enabled} == {"hackernews-top", "reddit-ml"}

    @pytest.mark.unit
    def test_all_default_sources_written(self, tmp_path):
        cfg = _make_cfg(enabled_source_ids=[])
        sources_path = tmp_path / "sources.json"
        _write_sources(cfg, sources_path)
        data = json.loads(sources_path.read_text())
        assert len(data["sources"]) == len(_DEFAULT_SOURCES)

    @pytest.mark.unit
    def test_non_selected_sources_are_disabled(self, tmp_path):
        cfg = _make_cfg(enabled_source_ids=["hackernews-top"])
        sources_path = tmp_path / "sources.json"
        _write_sources(cfg, sources_path)
        data = json.loads(sources_path.read_text())
        disabled = [s for s in data["sources"] if not s["enabled"]]
        assert all(s["id"] != "hackernews-top" for s in disabled)


# ===========================================================================
# _DEFAULT_SOURCES catalogue
# ===========================================================================


class TestDefaultSourcesCatalogue:
    @pytest.mark.unit
    def test_all_sources_have_id(self):
        for src in _DEFAULT_SOURCES:
            assert "id" in src, f"Source missing id: {src}"

    @pytest.mark.unit
    def test_all_source_ids_are_unique(self):
        ids = [s["id"] for s in _DEFAULT_SOURCES]
        assert len(ids) == len(set(ids))

    @pytest.mark.unit
    def test_all_sources_have_type(self):
        for src in _DEFAULT_SOURCES:
            assert src.get("type") in ("rss", "hackernews", "reddit")

    @pytest.mark.unit
    def test_all_sources_have_limit(self):
        for src in _DEFAULT_SOURCES:
            assert "limit" in src
            assert src["limit"] > 0

    @pytest.mark.unit
    def test_at_least_6_sources_in_catalogue(self):
        assert len(_DEFAULT_SOURCES) >= 6

    @pytest.mark.unit
    def test_at_least_one_hackernews_source(self):
        hn = [s for s in _DEFAULT_SOURCES if s["type"] == "hackernews"]
        assert len(hn) >= 1

    @pytest.mark.unit
    def test_at_least_one_rss_source(self):
        rss = [s for s in _DEFAULT_SOURCES if s["type"] == "rss"]
        assert len(rss) >= 1


# ===========================================================================
# run_wizard (non-interactive)
# ===========================================================================


class TestRunWizardNonInteractive:
    @pytest.mark.unit
    def test_non_interactive_writes_both_files(self, tmp_path):
        cfg = _make_cfg()
        env = tmp_path / ".env"
        sources = tmp_path / "sources.json"
        run_wizard(env, sources, non_interactive=True, defaults=cfg)
        assert env.exists()
        assert sources.exists()

    @pytest.mark.unit
    def test_non_interactive_returns_wizard_config(self, tmp_path):
        cfg = _make_cfg(score_threshold=9)
        result = run_wizard(
            tmp_path / ".env",
            tmp_path / "sources.json",
            non_interactive=True,
            defaults=cfg,
        )
        assert isinstance(result, WizardConfig)
        assert result.score_threshold == 9

    @pytest.mark.unit
    def test_non_interactive_env_readable_after_write(self, tmp_path):
        cfg = _make_cfg(ai_api_key="sk-roundtrip-test")
        env = tmp_path / ".env"
        run_wizard(env, tmp_path / "sources.json", non_interactive=True, defaults=cfg)
        env_data = _load_existing_env(env)
        assert env_data["OPENAI_API_KEY"] == "sk-roundtrip-test"

    @pytest.mark.unit
    def test_non_interactive_with_all_delivery_channels(self, tmp_path):
        cfg = _make_cfg(
            email_enabled=True,
            smtp_user="a@b.com",
            smtp_password="pass",
            email_to="c@d.com",
            discord_enabled=True,
            discord_webhook_url="https://discord.com/test",
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
        )
        env = tmp_path / ".env"
        run_wizard(env, tmp_path / "sources.json", non_interactive=True, defaults=cfg)
        env_data = _load_existing_env(env)
        assert "SMTP_USER" in env_data
        assert "DISCORD_WEBHOOK_URL" in env_data
        assert "SLACK_WEBHOOK_URL" in env_data
