"""
src/main.py
===========
CLI entry point for the News Radar pipeline.

Invocation:
    uv run python -m src.main --run
    uv run python -m src.main --dry-run
    uv run python -m src.main --setup
    uv run python -m src.main --version
    uv run python -m src.main --status

Or via the installed script:
    news-radar --run

The main() function is the only public API of this module.
It configures logging, validates the environment, then dispatches
to the appropriate sub-command handler.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="news-radar",
        description=(
            "📡 News Radar — AI-powered personal news briefing pipeline.\n"
            "Fetches, scores, and summarizes stories from RSS, HN, and Reddit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  news-radar --run           Run the full pipeline (fetch → score → summarize → deliver)
  news-radar --dry-run       Run pipeline but skip saving and delivery
  news-radar --setup         Launch the interactive setup wizard
  news-radar --status        Show last run info and current configuration
  news-radar --version       Print version and exit

  # Combine with log level:
  news-radar --run --log-level DEBUG
""",
    )

    # Mutually exclusive: only one action per run
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--run",
        action="store_true",
        help="Run the full news radar pipeline end-to-end.",
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline in dry-run mode: fetch and score but skip saves and notifications.",
    )
    action.add_argument(
        "--setup",
        action="store_true",
        help="Launch the interactive setup wizard to generate sources.json and .env.",
    )
    action.add_argument(
        "--status",
        action="store_true",
        help="Print last run status, current config summary, and source list.",
    )
    action.add_argument(
        "--version",
        action="store_true",
        help="Print the News Radar version and exit.",
    )

    # Optional modifiers
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Override the log verbosity level (default: from .env LOG_LEVEL).",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Target date for the briefing (default: today). Useful for backfills.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        metavar="FILE",
        help="Override the sources config file path (default: data/sources.json).",
    )

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _handle_version() -> None:
    """Print version string and exit."""
    try:
        from importlib.metadata import version

        ver = version("news-radar")
    except Exception:
        ver = "0.1.0-dev"
    print(f"news-radar {ver}")


def _handle_status(settings: object, log: object) -> None:
    """Print current configuration and last run status."""
    from src.config import Settings

    s: Settings = settings  # type: ignore[assignment]

    from rich.table import Table

    from src.logger import console

    # Config summary table
    table = Table(title="News Radar Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan", min_width=20)
    table.add_column("Value", style="white")

    table.add_row("AI Model", s.ai_model)
    table.add_row("Score Threshold", str(s.score_threshold))
    table.add_row("Max Items", str(s.max_briefing_items))
    table.add_row("Output Language", s.output_language)
    table.add_row("Log Level", s.log_level)
    table.add_row("Sources File", str(s.sources_file))
    table.add_row("Data Dir", str(s.data_dir))
    table.add_row("Has OpenAI Key", "[green]Yes[/green]" if s.has_openai else "[red]No[/red]")
    table.add_row("Has Gemini Key", "[green]Yes[/green]" if s.has_gemini else "[red]No[/red]")
    table.add_row(
        "Has Anthropic Key", "[green]Yes[/green]" if s.has_anthropic else "[red]No[/red]"
    )
    table.add_row("Email Delivery", "[green]On[/green]" if s.has_email else "[dim]Off[/dim]")
    table.add_row(
        "Discord Delivery", "[green]On[/green]" if s.has_discord else "[dim]Off[/dim]"
    )
    console.print(table)

    # Check last run
    run_log = s.data_dir / "run_log.json"
    if run_log.exists():
        import json

        with open(run_log) as f:
            runs = json.load(f)
        if runs:
            last = runs[-1]
            console.print(
                f"\n[bold]Last run:[/bold] {last.get('date', 'unknown')} — "
                f"{last.get('status', '?')} — "
                f"{last.get('items', 0)} items in {last.get('duration_s', 0):.1f}s"
            )
    else:
        console.print("\n[dim]No previous runs found.[/dim]")


def _handle_setup(log: object) -> None:
    """Launch the interactive setup wizard."""
    log.section("Setup Wizard")  # type: ignore[attr-defined]
    log.info("The setup wizard will be implemented on Day 18.")  # type: ignore[attr-defined]
    log.info(  # type: ignore[attr-defined]
        "For now, copy .env.example to .env and edit data/sources.json manually."
    )


async def _handle_run(settings: object, log: object, *, dry_run: bool = False, target_date: str | None = None, sources_file: str | None = None) -> int:
    """
    Main pipeline runner — invokes the Orchestrator.

    Returns the exit code (0 = success, 1 = error).
    """
    from datetime import date
    from pathlib import Path

    from src.config import Settings
    from src.orchestrator import Orchestrator

    s: Settings = settings  # type: ignore[assignment]

    # ---- Validate AI config ----
    warnings = s.validate_ai_config()
    for w in warnings:
        log.warning("%s", w)  # type: ignore[attr-defined]

    if not s.has_any_ai_key:
        log.error(  # type: ignore[attr-defined]
            "No AI API key configured. Set OPENAI_API_KEY, GEMINI_API_KEY, "
            "or ANTHROPIC_API_KEY in your .env file."
        )
        return 1

    # ---- Validate sources file ----
    sources_path = Path(sources_file) if sources_file else Path(s.sources_file)
    if not sources_path.exists():
        log.error(  # type: ignore[attr-defined]
            "Sources file not found: %s. Run --setup to create it.", sources_path
        )
        return 1

    # ---- Parse target date ----
    parsed_date: date | None = None
    if target_date:
        try:
            parsed_date = date.fromisoformat(target_date)
        except ValueError:
            log.error("Invalid date format: %r (expected YYYY-MM-DD)", target_date)  # type: ignore[attr-defined]
            return 1

    # ---- Run the pipeline ----
    orc = Orchestrator(s)
    briefing = await orc.run(
        dry_run=dry_run,
        target_date=parsed_date,
        sources_override=sources_path if sources_file else None,
    )

    if briefing is None:
        log.error("Pipeline failed — see logs above for details.")  # type: ignore[attr-defined]
        return 1

    return 0



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point — called by the ``news-radar`` script and by
    ``python -m src.main``.
    """
    # ---- Parse arguments ----
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Version is special: no config needed ----
    if args.version:
        _handle_version()
        sys.exit(0)

    # ---- Load config (after arg parsing so --log-level can override) ----
    from src.config import Settings
    from src.logger import configure_logging, get_logger

    settings = Settings()

    # CLI --log-level overrides .env LOG_LEVEL
    effective_level = args.log_level or settings.log_level
    configure_logging(effective_level)

    log = get_logger("src.main")

    # ---- Override sources file if provided ----
    if args.sources:
        log.info("Using custom sources file: %s", args.sources)

    # ---- Dispatch to sub-command ----
    try:
        if args.status:
            _handle_status(settings, log)

        elif args.setup:
            _handle_setup(log)

        elif args.run:
            exit_code = asyncio.run(_handle_run(
                settings, log,
                dry_run=False,
                target_date=args.date,
                sources_file=args.sources,
            ))
            sys.exit(exit_code)

        elif args.dry_run:
            exit_code = asyncio.run(_handle_run(
                settings, log,
                dry_run=True,
                target_date=args.date,
                sources_file=args.sources,
            ))
            sys.exit(exit_code)

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)

    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
