"""
src/config.py
=============
Centralised application configuration loaded from environment variables.

Uses Pydantic BaseSettings so every setting is:
  - Type-validated at startup (wrong type → clear error, not a mystery crash)
  - Documented in one place
  - Overridable via .env file or real environment variables

Usage:
    from src.config import settings
    print(settings.ai_model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All runtime configuration for News Radar.

    Values are loaded (in priority order) from:
      1. Real environment variables
      2. .env file in the project root
      3. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY == openai_api_key
        extra="ignore",  # ignore unknown env vars silently
    )

    # ------------------------------------------------------------------
    # AI Provider
    # ------------------------------------------------------------------

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (sk-...)",
    )

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key",
    )

    anthropic_api_key: str = Field(
        default="",
        description="Anthropic Claude API key",
    )

    ai_model: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model identifier to use for scoring & summarization. "
            "Examples: gpt-4o-mini, gemini-1.5-flash, claude-3-haiku-20240307"
        ),
    )

    # ------------------------------------------------------------------
    # Pipeline Tuning
    # ------------------------------------------------------------------

    score_threshold: int = Field(
        default=6,
        ge=0,
        le=10,
        description="Minimum AI score (0-10) for a story to appear in the briefing.",
    )

    max_briefing_items: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of stories in the daily briefing.",
    )

    output_language: str = Field(
        default="English",
        description="Language for AI-generated summaries and titles.",
    )

    user_interests: str = Field(
        default="AI, machine learning, Python, open source, developer tools, software engineering",
        description=(
            "Comma-separated description of what the reader cares about. "
            "Used in every AI scoring and summarization prompt. "
            "Example: 'AI, Python, startup funding, open source'"
        ),
    )

    # ------------------------------------------------------------------
    # Data & Storage
    # ------------------------------------------------------------------

    sources_file: Path = Field(
        default=Path("data/sources.json"),
        description="Path to the sources configuration JSON file.",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for all runtime data (briefings, cache, logs).",
    )

    # ------------------------------------------------------------------
    # Delivery Channels (all optional)
    # ------------------------------------------------------------------

    # Email
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server hostname.")
    smtp_port: int = Field(default=587, description="SMTP server port.")
    smtp_user: str = Field(default="", description="SMTP login username / sender address.")
    smtp_password: str = Field(default="", description="SMTP password or app password.")
    email_to: str = Field(default="", description="Recipient email address for briefings.")

    # Webhooks
    discord_webhook_url: str = Field(default="", description="Discord webhook URL.")
    slack_webhook_url: str = Field(default="", description="Slack webhook URL.")
    custom_webhook_url: str = Field(default="", description="Generic JSON webhook URL.")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    docs_dir: Path = Field(
        default=Path("docs"),
        description="Directory where GitHub Pages HTML/MD files are written.",
    )

    github_pages_enabled: bool = Field(
        default=True,
        description="If True, write rendered output to docs/ after each run.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log verbosity level.",
    )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @field_validator("score_threshold", mode="before")
    @classmethod
    def clamp_threshold(cls, v: int) -> int:
        """Ensure threshold stays in [0, 10] even if .env has a bad value."""
        return max(0, min(10, int(v)))

    @property
    def briefings_dir(self) -> Path:
        """Resolved path to the daily briefings folder."""
        return self.data_dir / "briefings"

    @property
    def cache_dir(self) -> Path:
        """Resolved path to the AI response cache folder."""
        return self.data_dir / "cache"

    @property
    def has_openai(self) -> bool:
        """True if an OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        """True if a Gemini API key is configured."""
        return bool(self.gemini_api_key)

    @property
    def has_anthropic(self) -> bool:
        """True if an Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_email(self) -> bool:
        """True if email delivery is fully configured."""
        return all([self.smtp_user, self.smtp_password, self.email_to])

    @property
    def has_discord(self) -> bool:
        """True if a Discord webhook URL is configured."""
        return bool(self.discord_webhook_url)

    @property
    def has_slack(self) -> bool:
        """True if a Slack webhook URL is configured."""
        return bool(self.slack_webhook_url)

    @property
    def has_any_ai_key(self) -> bool:
        """True if at least one AI API key is configured."""
        return self.has_openai or self.has_gemini or self.has_anthropic

    @property
    def active_model_provider(self) -> str:
        """Returns the provider name for the configured AI model."""
        m = self.ai_model.lower()
        if m.startswith(("gpt", "o1", "o3")):
            return "openai"
        if m.startswith("gemini"):
            return "gemini"
        if m.startswith("claude"):
            return "anthropic"
        return "unknown"

    def validate_ai_config(self) -> list[str]:
        """
        Validate that the configured AI model has a corresponding API key.

        Returns a list of warning strings (empty if everything is OK).
        Use at startup to give clear error messages before the pipeline runs.
        """
        warnings: list[str] = []
        provider = self.active_model_provider
        if provider == "openai" and not self.has_openai:
            warnings.append(
                f"AI_MODEL={self.ai_model!r} requires OPENAI_API_KEY (not set)"
            )
        if provider == "gemini" and not self.has_gemini:
            warnings.append(
                f"AI_MODEL={self.ai_model!r} requires GEMINI_API_KEY (not set)"
            )
        if provider == "anthropic" and not self.has_anthropic:
            warnings.append(
                f"AI_MODEL={self.ai_model!r} requires ANTHROPIC_API_KEY (not set)"
            )
        return warnings


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
