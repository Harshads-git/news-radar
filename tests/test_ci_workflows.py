"""
tests/test_ci_workflows.py
===========================
Workflow validation and CI environment tests.

These tests verify:
  1. YAML workflow files are syntactically valid
  2. Required secrets/env vars are documented
  3. The pipeline works correctly with CI-style env injection
  4. .env.example has all required keys documented
  5. Docs bootstrap HTML is well-formed

Philosophy: CI should validate itself. If someone edits a workflow
file and introduces a syntax error, these tests catch it locally
before the broken YAML reaches GitHub.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOCS_INDEX = REPO_ROOT / "docs" / "index.html"


# ===========================================================================
# YAML Workflow Validation
# ===========================================================================


class TestWorkflowFiles:
    @pytest.mark.unit
    def test_ci_yml_exists(self):
        assert (WORKFLOWS_DIR / "ci.yml").exists(), "ci.yml missing"

    @pytest.mark.unit
    def test_daily_yml_exists(self):
        assert (WORKFLOWS_DIR / "daily.yml").exists(), "daily.yml missing"

    @pytest.mark.unit
    def test_ci_yml_is_valid_yaml(self):
        """Validate ci.yml parses as YAML without errors."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        doc = yaml.safe_load(content)
        assert isinstance(doc, dict), "ci.yml root must be a mapping"

    @pytest.mark.unit
    def test_daily_yml_is_valid_yaml(self):
        """Validate daily.yml parses as YAML without errors."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        doc = yaml.safe_load(content)
        assert isinstance(doc, dict), "daily.yml root must be a mapping"

    @pytest.mark.unit
    def test_ci_yml_has_push_trigger(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "push:" in content

    @pytest.mark.unit
    def test_ci_yml_has_pull_request_trigger(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "pull_request:" in content

    @pytest.mark.unit
    def test_daily_yml_has_cron_trigger(self):
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "schedule:" in content
        assert "cron:" in content

    @pytest.mark.unit
    def test_daily_yml_has_workflow_dispatch(self):
        """Manual trigger must be available for testing."""
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "workflow_dispatch:" in content

    @pytest.mark.unit
    def test_ci_yml_uses_uv(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "setup-uv" in content or "astral-sh" in content

    @pytest.mark.unit
    def test_daily_yml_uses_secrets_for_api_keys(self):
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "secrets.OPENAI_API_KEY" in content
        assert "secrets.GEMINI_API_KEY" in content

    @pytest.mark.unit
    def test_daily_yml_has_dry_run_input(self):
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "dry_run" in content

    @pytest.mark.unit
    def test_daily_yml_has_github_pages_deploy(self):
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "deploy-pages" in content or "actions/deploy-pages" in content

    @pytest.mark.unit
    def test_daily_yml_commits_docs_changes(self):
        content = (WORKFLOWS_DIR / "daily.yml").read_text(encoding="utf-8")
        assert "git commit" in content
        assert "docs/" in content

    @pytest.mark.unit
    def test_ci_yml_has_coverage_enforcement(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "cov-fail-under" in content

    @pytest.mark.unit
    def test_ci_yml_has_cancel_in_progress(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "cancel-in-progress" in content

    @pytest.mark.unit
    def test_ci_yml_tests_multiple_python_versions(self):
        content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "3.11" in content
        assert "3.12" in content


# ===========================================================================
# .env.example Completeness
# ===========================================================================


class TestEnvExample:
    @pytest.mark.unit
    def test_env_example_exists(self):
        assert ENV_EXAMPLE.exists(), ".env.example missing"

    @pytest.mark.unit
    def test_env_example_has_openai_key(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" in content

    @pytest.mark.unit
    def test_env_example_has_gemini_key(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "GEMINI_API_KEY" in content

    @pytest.mark.unit
    def test_env_example_has_discord_webhook(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "DISCORD_WEBHOOK_URL" in content

    @pytest.mark.unit
    def test_env_example_has_smtp_settings(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "SMTP_USER" in content
        assert "SMTP_PASSWORD" in content
        assert "EMAIL_TO" in content

    @pytest.mark.unit
    def test_env_example_has_github_pages_setting(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "GITHUB_PAGES_ENABLED" in content

    @pytest.mark.unit
    def test_env_example_has_user_interests(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "USER_INTERESTS" in content

    @pytest.mark.unit
    def test_env_example_has_score_threshold(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "SCORE_THRESHOLD" in content

    @pytest.mark.unit
    def test_env_example_has_log_level(self):
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "LOG_LEVEL" in content


# ===========================================================================
# Docs Bootstrap Validation
# ===========================================================================


class TestDocsBootstrap:
    @pytest.mark.unit
    def test_docs_index_exists(self):
        assert DOCS_INDEX.exists(), "docs/index.html missing"

    @pytest.mark.unit
    def test_docs_index_is_valid_html(self):
        content = DOCS_INDEX.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    @pytest.mark.unit
    def test_docs_index_has_meta_description(self):
        content = DOCS_INDEX.read_text(encoding="utf-8")
        assert 'name="description"' in content

    @pytest.mark.unit
    def test_docs_index_has_title(self):
        content = DOCS_INDEX.read_text(encoding="utf-8")
        assert "<title>" in content
        assert "News Radar" in content

    @pytest.mark.unit
    def test_docs_index_has_github_link(self):
        content = DOCS_INDEX.read_text(encoding="utf-8")
        assert "github.com/Harshads-git" in content

    @pytest.mark.unit
    def test_docs_index_is_dark_themed(self):
        content = DOCS_INDEX.read_text(encoding="utf-8")
        # Check for dark background color
        assert "#0d1117" in content or "dark" in content.lower()


# ===========================================================================
# CI Environment Simulation
# ===========================================================================


class TestCIEnvironmentSimulation:
    @pytest.mark.unit
    def test_settings_load_with_ci_env_vars(self, monkeypatch):
        """Settings must load cleanly with the dummy API keys injected in CI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-ci")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-ci")
        # Settings must not raise
        from src.config import Settings
        s = Settings()
        assert s.has_openai
        assert s.has_gemini

    @pytest.mark.unit
    def test_check_command_runs_with_ci_env(self, monkeypatch, tmp_path):
        """--check must pass (exit 0) with CI env vars."""
        import json
        from src.main import _handle_check
        from unittest.mock import MagicMock

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-ci")

        # Create minimal sources.json
        sources_file = tmp_path / "sources.json"
        sources_file.write_text(json.dumps({"sources": [{
            "id": "hn", "type": "hackernews",
            "name": "HN", "enabled": True, "limit": 5, "tags": [],
        }]}))

        s = MagicMock()
        s.ai_model = "gpt-4o-mini"
        s.active_model_provider = "openai"
        s.validate_ai_config.return_value = []
        s.sources_file = sources_file
        s.data_dir = tmp_path / "data"
        s.docs_dir = tmp_path / "docs"
        s.has_email = False
        s.has_discord = False
        s.has_slack = False
        s.custom_webhook_url = ""
        s.github_pages_enabled = True

        log = MagicMock()
        result = _handle_check(s, log)
        assert result == 0

    @pytest.mark.unit
    def test_secrets_md_exists(self):
        secrets_doc = REPO_ROOT / ".github" / "SECRETS.md"
        assert secrets_doc.exists(), ".github/SECRETS.md missing"

    @pytest.mark.unit
    def test_secrets_md_documents_all_required_secrets(self):
        secrets_doc = REPO_ROOT / ".github" / "SECRETS.md"
        content = secrets_doc.read_text(encoding="utf-8")
        required = [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "DISCORD_WEBHOOK_URL",
            "SMTP_USER",
            "SMTP_PASSWORD",
        ]
        for secret in required:
            assert secret in content, f"{secret} not documented in SECRETS.md"
