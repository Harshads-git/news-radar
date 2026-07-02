"""
src/setup/wizard.py
====================
Interactive setup wizard for News Radar.

Guides the user through configuring:
  1. AI provider + API key (OpenAI / Gemini / Anthropic)
  2. Personal interests (what to score highly)
  3. Score threshold and item count
  4. Delivery channels (Email, Discord, Slack, webhook)
  5. GitHub Pages toggle
  6. News sources to enable

Outputs:
  - .env file (or updates existing)
  - data/sources.json (pre-populated from bundled template)

Why a wizard instead of just docs?
  Editing raw .env files is error-prone — wrong variable names, missing
  values, copy-paste mistakes. The wizard validates each input immediately
  (e.g., checks that an API key starts with 'sk-' or 'AIza'), gives
  sensible defaults, and shows a preview before writing.

Usage:
    from src.setup.wizard import run_wizard
    run_wizard(env_path=Path(".env"), sources_path=Path("data/sources.json"))
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default sources catalogue
# The wizard picks from this list; user can toggle each on/off.
# ---------------------------------------------------------------------------

_DEFAULT_SOURCES: list[dict[str, Any]] = [
    {
        "id": "hackernews-top",
        "type": "hackernews",
        "name": "Hacker News Top",
        "url": None,
        "enabled": True,
        "limit": 30,
        "hn_story_type": "top",
        "tags": ["tech", "programming"],
    },
    {
        "id": "hackernews-new",
        "type": "hackernews",
        "name": "Hacker News New",
        "url": None,
        "enabled": False,
        "limit": 20,
        "hn_story_type": "new",
        "tags": ["tech"],
    },
    {
        "id": "reddit-ml",
        "type": "reddit",
        "name": "r/MachineLearning",
        "url": "https://www.reddit.com/r/MachineLearning",
        "subreddit": "MachineLearning",
        "enabled": True,
        "limit": 20,
        "tags": ["AI", "ML"],
    },
    {
        "id": "reddit-localllama",
        "type": "reddit",
        "name": "r/LocalLLaMA",
        "url": "https://www.reddit.com/r/LocalLLaMA",
        "subreddit": "LocalLLaMA",
        "enabled": True,
        "limit": 15,
        "tags": ["AI", "LLM"],
    },
    {
        "id": "reddit-python",
        "type": "reddit",
        "name": "r/Python",
        "url": "https://www.reddit.com/r/Python",
        "subreddit": "Python",
        "enabled": True,
        "limit": 15,
        "tags": ["Python"],
    },
    {
        "id": "openai-blog",
        "type": "rss",
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "enabled": True,
        "limit": 10,
        "tags": ["AI", "OpenAI"],
    },
    {
        "id": "deepmind-blog",
        "type": "rss",
        "name": "DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "enabled": False,
        "limit": 10,
        "tags": ["AI", "research"],
    },
    {
        "id": "huggingface-blog",
        "type": "rss",
        "name": "HuggingFace Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "enabled": True,
        "limit": 10,
        "tags": ["AI", "ML"],
    },
    {
        "id": "techcrunch",
        "type": "rss",
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "enabled": False,
        "limit": 15,
        "tags": ["tech", "startups"],
    },
    {
        "id": "ars-technica",
        "type": "rss",
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
        "enabled": True,
        "limit": 15,
        "tags": ["tech", "science"],
    },
    {
        "id": "github-blog",
        "type": "rss",
        "name": "GitHub Blog",
        "url": "https://github.blog/feed/",
        "enabled": True,
        "limit": 10,
        "tags": ["dev", "tools"],
    },
    {
        "id": "the-verge",
        "type": "rss",
        "name": "The Verge",
        "url": "https://www.theverge.com/rss/index.xml",
        "enabled": False,
        "limit": 15,
        "tags": ["tech", "consumer"],
    },
]


# ---------------------------------------------------------------------------
# Wizard configuration result
# ---------------------------------------------------------------------------


@dataclass
class WizardConfig:
    """Collects all answers from the wizard before writing output files."""

    # AI
    ai_provider: str = "openai"  # openai | gemini | anthropic
    ai_model: str = "gpt-4o-mini"
    ai_api_key: str = ""

    # Interests
    user_interests: str = "AI, machine learning, Python, open source"
    score_threshold: int = 6
    max_briefing_items: int = 20

    # Delivery
    email_enabled: bool = False
    smtp_user: str = ""
    smtp_password: str = ""
    email_to: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    discord_enabled: bool = False
    discord_webhook_url: str = ""

    slack_enabled: bool = False
    slack_webhook_url: str = ""

    github_pages_enabled: bool = False

    # Sources
    enabled_source_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wizard runner
# ---------------------------------------------------------------------------


def run_wizard(
    env_path: Path,
    sources_path: Path,
    *,
    non_interactive: bool = False,
    defaults: WizardConfig | None = None,
) -> WizardConfig:
    """
    Run the interactive setup wizard.

    Parameters
    ----------
    env_path:
        Path to write the .env file (e.g., Path(".env")).
    sources_path:
        Path to write sources.json (e.g., Path("data/sources.json")).
    non_interactive:
        If True, skip all prompts and use defaults/provided config.
        Used in tests and CI environments.
    defaults:
        Pre-filled config to use instead of prompting (non_interactive only).

    Returns
    -------
    WizardConfig
        The completed configuration.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm, IntPrompt, Prompt
        from rich.text import Text
    except ImportError:
        # Rich not available — fall back to plain prompts
        _run_plain_wizard(env_path, sources_path)
        return WizardConfig()

    console = Console()
    cfg = defaults or WizardConfig()

    if non_interactive:
        _write_outputs(cfg, env_path, sources_path, console)
        return cfg

    # ---- Welcome banner ----
    console.print()
    console.print(Panel.fit(
        "[bold cyan]📡 News Radar Setup Wizard[/bold cyan]\n"
        "[dim]This wizard will create your .env and sources.json files.[/dim]\n"
        "[dim]Press Enter to accept defaults shown in brackets.[/dim]",
        border_style="cyan",
    ))
    console.print()

    # ---- Detect existing config ----
    existing_env = _load_existing_env(env_path)
    if existing_env:
        console.print(f"[yellow]⚠[/yellow]  Found existing [bold]{env_path}[/bold] — "
                      "values will be used as defaults.")
        console.print()

    # ==================================================================
    # STEP 1: AI Provider
    # ==================================================================
    console.print("[bold]Step 1 of 5 — AI Provider[/bold]")
    console.print("[dim]News Radar uses an AI model to score and summarize stories.[/dim]")
    console.print()

    provider_map = {
        "1": ("openai", "gpt-4o-mini",   "OPENAI_API_KEY",   "sk-"),
        "2": ("gemini", "gemini-1.5-flash", "GEMINI_API_KEY", "AIza"),
        "3": ("anthropic", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY", "sk-ant-"),
    }

    console.print("  [1] OpenAI (gpt-4o-mini) — recommended, cheap, ~$0.02/day")
    console.print("  [2] Google Gemini (gemini-1.5-flash) — free tier available")
    console.print("  [3] Anthropic (claude-3-haiku) — high quality")
    console.print()

    default_provider_num = "1"
    existing_provider = existing_env.get("AI_PROVIDER", "openai")
    for num, (prov, _, _, _) in provider_map.items():
        if prov == existing_provider:
            default_provider_num = num
            break

    choice = Prompt.ask(
        "  Choose provider",
        choices=["1", "2", "3"],
        default=default_provider_num,
    )
    cfg.ai_provider, cfg.ai_model, key_var, key_prefix = provider_map[choice]

    # API key
    existing_key = existing_env.get(key_var, "")
    masked_key = f"{existing_key[:8]}..." if len(existing_key) > 8 else ""
    key_prompt = f"  Enter your {key_var}"
    if masked_key:
        key_prompt += f" (existing: {masked_key})"

    while True:
        raw_key = Prompt.ask(key_prompt, default=existing_key, password=True)
        if raw_key and (raw_key.startswith(key_prefix) or len(raw_key) > 20):
            cfg.ai_api_key = raw_key
            break
        if not raw_key and existing_key:
            cfg.ai_api_key = existing_key
            break
        console.print(
            f"  [red]✗[/red] Key looks wrong (expected prefix: {key_prefix}). Try again."
        )

    console.print(f"  [green]✓[/green] AI: {cfg.ai_model}\n")

    # ==================================================================
    # STEP 2: Interests & Scoring
    # ==================================================================
    console.print("[bold]Step 2 of 5 — Your Interests[/bold]")
    console.print("[dim]The AI uses this to score story relevance for you.[/dim]\n")

    default_interests = existing_env.get("USER_INTERESTS", cfg.user_interests)
    cfg.user_interests = Prompt.ask(
        "  Your interests (comma-separated topics)",
        default=default_interests,
    )

    default_threshold = int(existing_env.get("SCORE_THRESHOLD", cfg.score_threshold))
    cfg.score_threshold = IntPrompt.ask(
        "  Minimum score to include in briefing (1-10)",
        default=default_threshold,
    )
    cfg.score_threshold = max(1, min(10, cfg.score_threshold))

    default_max = int(existing_env.get("MAX_BRIEFING_ITEMS", cfg.max_briefing_items))
    cfg.max_briefing_items = IntPrompt.ask(
        "  Max stories per briefing",
        default=default_max,
    )
    console.print(
        f"  [green]✓[/green] Interests set, threshold={cfg.score_threshold}/10, "
        f"max={cfg.max_briefing_items} items\n"
    )

    # ==================================================================
    # STEP 3: Delivery
    # ==================================================================
    console.print("[bold]Step 3 of 5 — Delivery Channels[/bold]")
    console.print("[dim]Where should News Radar send your briefing? (all optional)[/dim]\n")

    # Email
    if Confirm.ask("  Enable email delivery?", default=bool(existing_env.get("SMTP_USER"))):
        cfg.email_enabled = True
        cfg.smtp_user = Prompt.ask(
            "    Gmail address (SMTP user)",
            default=existing_env.get("SMTP_USER", ""),
        )
        cfg.smtp_password = Prompt.ask(
            "    Gmail App Password",
            default=existing_env.get("SMTP_PASSWORD", ""),
            password=True,
        )
        cfg.email_to = Prompt.ask(
            "    Send briefing to (email)",
            default=existing_env.get("EMAIL_TO", cfg.smtp_user),
        )
        console.print(f"    [green]✓[/green] Email → {cfg.email_to}")

    # Discord
    if Confirm.ask("\n  Enable Discord delivery?",
                   default=bool(existing_env.get("DISCORD_WEBHOOK_URL"))):
        cfg.discord_enabled = True
        cfg.discord_webhook_url = Prompt.ask(
            "    Discord webhook URL",
            default=existing_env.get("DISCORD_WEBHOOK_URL", ""),
        )
        console.print("    [green]✓[/green] Discord webhook configured")

    # Slack
    if Confirm.ask("\n  Enable Slack delivery?",
                   default=bool(existing_env.get("SLACK_WEBHOOK_URL"))):
        cfg.slack_enabled = True
        cfg.slack_webhook_url = Prompt.ask(
            "    Slack webhook URL",
            default=existing_env.get("SLACK_WEBHOOK_URL", ""),
        )
        console.print("    [green]✓[/green] Slack webhook configured")

    # GitHub Pages
    default_gh = existing_env.get("GITHUB_PAGES_ENABLED", "true").lower() == "true"
    cfg.github_pages_enabled = Confirm.ask(
        "\n  Enable GitHub Pages briefing?",
        default=default_gh,
    )
    if cfg.github_pages_enabled:
        console.print("    [green]✓[/green] GitHub Pages will be updated after each run")

    console.print()

    # ==================================================================
    # STEP 4: Sources
    # ==================================================================
    console.print("[bold]Step 4 of 5 — News Sources[/bold]")
    console.print("[dim]Choose which sources to fetch from. "
                  "Defaults shown are recommended.[/dim]\n")

    existing_sources = _load_existing_sources(sources_path)
    existing_enabled_ids = {s["id"] for s in existing_sources if s.get("enabled")}

    enabled_ids: list[str] = []
    for src in _DEFAULT_SOURCES:
        default_on = src["id"] in existing_enabled_ids if existing_sources else src["enabled"]
        tag_str = ", ".join(src.get("tags", []))
        label = f"  {src['name']}"
        if tag_str:
            label += f" [dim]({tag_str})[/dim]"
        enabled = Confirm.ask(label, default=default_on)
        if enabled:
            enabled_ids.append(src["id"])

    cfg.enabled_source_ids = enabled_ids
    console.print(f"\n  [green]✓[/green] {len(enabled_ids)} sources selected\n")

    # ==================================================================
    # STEP 5: Preview & Write
    # ==================================================================
    console.print("[bold]Step 5 of 5 — Review & Save[/bold]\n")

    _print_config_preview(cfg, console)

    if not Confirm.ask("\n  Write these files now?", default=True):
        console.print("[yellow]Setup cancelled. No files written.[/yellow]")
        return cfg

    _write_outputs(cfg, env_path, sources_path, console)
    return cfg


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_env(cfg: WizardConfig, env_path: Path) -> None:
    """Write (or update) the .env file from the wizard config."""
    # Load existing lines to preserve any custom settings not covered by wizard
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    # Build dict of new key=value pairs
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    new_values: dict[str, str] = {
        provider_key_map[cfg.ai_provider]: cfg.ai_api_key,
        "AI_MODEL": cfg.ai_model,
        "USER_INTERESTS": cfg.user_interests,
        "SCORE_THRESHOLD": str(cfg.score_threshold),
        "MAX_BRIEFING_ITEMS": str(cfg.max_briefing_items),
        "GITHUB_PAGES_ENABLED": "true" if cfg.github_pages_enabled else "false",
    }

    if cfg.email_enabled:
        new_values["SMTP_USER"] = cfg.smtp_user
        new_values["SMTP_PASSWORD"] = cfg.smtp_password
        new_values["EMAIL_TO"] = cfg.email_to
        new_values["SMTP_HOST"] = cfg.smtp_host
        new_values["SMTP_PORT"] = str(cfg.smtp_port)

    if cfg.discord_enabled:
        new_values["DISCORD_WEBHOOK_URL"] = cfg.discord_webhook_url

    if cfg.slack_enabled:
        new_values["SLACK_WEBHOOK_URL"] = cfg.slack_webhook_url

    # Merge: update existing lines where key matches, append new keys
    updated_keys: set[str] = set()
    output_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            output_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in new_values:
            output_lines.append(f"{key}={new_values[key]}")
            updated_keys.add(key)
        else:
            output_lines.append(line)

    # Append any new keys not in existing file
    new_keys = [k for k in new_values if k not in updated_keys]
    if new_keys:
        if output_lines and output_lines[-1].strip():
            output_lines.append("")  # blank line separator
        output_lines.append("# Added by setup wizard")
        for key in new_keys:
            output_lines.append(f"{key}={new_values[key]}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _write_sources(cfg: WizardConfig, sources_path: Path) -> None:
    """Write sources.json with the user's enabled/disabled selections."""
    sources = []
    for src in _DEFAULT_SOURCES:
        s = dict(src)
        s["enabled"] = src["id"] in cfg.enabled_source_ids
        sources.append(s)

    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(
        json.dumps({"sources": sources}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_outputs(
    cfg: WizardConfig,
    env_path: Path,
    sources_path: Path,
    console: object,
) -> None:
    """Write both .env and sources.json, then print success message."""
    try:
        from rich.console import Console
        c: Console = console  # type: ignore[assignment]
    except ImportError:
        c = None  # type: ignore[assignment]

    _write_env(cfg, env_path)
    _write_sources(cfg, sources_path)

    if c:
        c.print(f"\n[green]✓[/green] Written: [bold]{env_path}[/bold]")
        c.print(f"[green]✓[/green] Written: [bold]{sources_path}[/bold]")
        c.print()
        c.print("[bold green]Setup complete![/bold green]")
        c.print(
            f"Run [bold cyan]news-radar --check[/bold cyan] to validate, "
            f"then [bold cyan]news-radar --dry-run[/bold cyan] to test."
        )


# ---------------------------------------------------------------------------
# Plain-text fallback (no Rich)
# ---------------------------------------------------------------------------


def _run_plain_wizard(env_path: Path, sources_path: Path) -> None:
    """Minimal plain-text wizard when Rich is not available."""
    print("\n=== News Radar Setup ===")
    print("Rich is not installed. Using basic setup.\n")

    cfg = WizardConfig()

    key = input("Enter your OPENAI_API_KEY (or press Enter to skip): ").strip()
    if key:
        cfg.ai_api_key = key
        cfg.ai_provider = "openai"

    interests = input(f"Your interests [{cfg.user_interests}]: ").strip()
    if interests:
        cfg.user_interests = interests

    cfg.enabled_source_ids = [s["id"] for s in _DEFAULT_SOURCES if s["enabled"]]

    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        console = None  # type: ignore[assignment]

    _write_outputs(cfg, env_path, sources_path, console)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_existing_env(env_path: Path) -> dict[str, str]:
    """Parse an existing .env file into a dict. Returns {} if not found."""
    if not env_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        result[key.strip()] = val.strip()
    return result


def _load_existing_sources(sources_path: Path) -> list[dict]:
    """Load existing sources.json. Returns [] if not found or invalid."""
    if not sources_path.exists():
        return []
    try:
        data = json.loads(sources_path.read_text(encoding="utf-8"))
        return data.get("sources", [])
    except (json.JSONDecodeError, OSError):
        return []


def _print_config_preview(cfg: WizardConfig, console: object) -> None:
    """Print a formatted preview of the wizard config before writing."""
    try:
        from rich.console import Console
        from rich.table import Table
        c: Console = console  # type: ignore[assignment]
    except ImportError:
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="white")

    table.add_row("AI Model", cfg.ai_model)
    table.add_row("Interests", cfg.user_interests[:60] + ("…" if len(cfg.user_interests) > 60 else ""))
    table.add_row("Score Threshold", f"{cfg.score_threshold}/10")
    table.add_row("Max Items", str(cfg.max_briefing_items))
    table.add_row("Email", cfg.email_to if cfg.email_enabled else "disabled")
    table.add_row("Discord", "enabled" if cfg.discord_enabled else "disabled")
    table.add_row("Slack", "enabled" if cfg.slack_enabled else "disabled")
    table.add_row("GitHub Pages", "enabled" if cfg.github_pages_enabled else "disabled")
    table.add_row("Sources", f"{len(cfg.enabled_source_ids)} enabled")

    c.print(table)  # type: ignore[attr-defined]
