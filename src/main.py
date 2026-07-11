"""
src/main.py
===========
CLI entry point for the News Radar pipeline.

Invocation:
    uv run python -m src.main --run
    uv run python -m src.main --dry-run
    uv run python -m src.main --dry-run --no-enrich
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
  news-radar --briefing      Print the most recent briefing to the terminal
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
    action.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration, sources.json, and API keys. Exit 0 if all OK.",
    )
    action.add_argument(
        "--briefing",
        action="store_true",
        help="Print the most recent stored briefing to the terminal in Rich format.",
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
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        default=False,
        dest="no_enrich",
        help="Skip DuckDuckGo context enrichment (faster runs, weaker summaries).",
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
    """Print current configuration, last run info, and recent event log summary."""
    from src.config import Settings
    from src.pipeline.event_log import EventLog
    from rich.table import Table
    from src.logger import console

    s: Settings = settings  # type: ignore[assignment]

    # ---- Config summary ----
    table = Table(title="News Radar Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan", min_width=22)
    table.add_column("Value", style="white")

    table.add_row("AI Model", s.ai_model)
    table.add_row("Score Threshold", str(s.score_threshold))
    table.add_row("Max Items", str(s.max_briefing_items))
    table.add_row("Output Language", s.output_language)
    table.add_row("Log Level", s.log_level)
    table.add_row("Sources File", str(s.sources_file))
    table.add_row("Data Dir", str(s.data_dir))
    table.add_row("Docs Dir", str(s.docs_dir))
    table.add_row("Has OpenAI Key", "[green]Yes[/green]" if s.has_openai else "[red]No[/red]")
    table.add_row("Has Gemini Key", "[green]Yes[/green]" if s.has_gemini else "[red]No[/red]")
    table.add_row(
        "Has Anthropic Key", "[green]Yes[/green]" if s.has_anthropic else "[red]No[/red]"
    )
    table.add_row("Email Delivery", "[green]On[/green]" if s.has_email else "[dim]Off[/dim]")
    table.add_row(
        "Discord Delivery", "[green]On[/green]" if s.has_discord else "[dim]Off[/dim]"
    )
    table.add_row(
        "GitHub Pages", "[green]On[/green]" if s.github_pages_enabled else "[dim]Off[/dim]"
    )
    console.print(table)

    # ---- Run history ----
    run_log = s.data_dir / "run_log.json"
    if run_log.exists():
        import json
        try:
            runs = json.loads(run_log.read_text(encoding="utf-8"))
        except Exception:
            runs = []

        if runs:
            run_table = Table(title="Recent Runs (last 7)", show_header=True, header_style="bold magenta")
            run_table.add_column("Date", style="cyan", min_width=12)
            run_table.add_column("Status", min_width=10)
            run_table.add_column("Items", justify="right", min_width=6)
            run_table.add_column("Fetched", justify="right", min_width=8)
            run_table.add_column("Duration", justify="right", min_width=10)
            run_table.add_column("Errors", min_width=5)

            for run in runs[-7:]:
                status = run.get("status", "?")
                status_colored = (
                    f"[green]{status}[/green]" if status == "success"
                    else f"[red]{status}[/red]"
                )
                error_count = len(run.get("errors", []))
                error_str = f"[red]{error_count}[/red]" if error_count else "[dim]0[/dim]"
                dry = " [dim](dry)[/dim]" if run.get("dry_run") else ""
                run_table.add_row(
                    run.get("date", "?") + dry,
                    status_colored,
                    str(run.get("in_briefing", "?")),
                    str(run.get("fetched", "?")),
                    f"{run.get('duration_s', 0):.1f}s",
                    error_str,
                )
            console.print(run_table)
    else:
        console.print("\n[dim]No previous runs found.[/dim]")

    # ---- Last run timeline (from event log) ----
    from src.pipeline.event_log import build_status_panel, aggregate_runs
    timeline_panel = build_status_panel(s.data_dir)
    if timeline_panel is not None:
        console.print()
        console.print(timeline_panel)
    else:
        console.print("\n[dim]No run timeline available yet — run the pipeline first.[/dim]")

    # ---- 7-day aggregate stats ----
    agg = aggregate_runs(s.data_dir, days=7)
    if agg.run_count > 0:
        from rich.table import Table as _T
        agg_table = _T(title="7-Day Stats", show_header=False, box=None, padding=(0, 2))
        agg_table.add_column("Metric", style="dim")
        agg_table.add_column("Value", style="white")
        agg_table.add_row("Runs", str(agg.run_count))
        agg_table.add_row(
            "Success rate",
            f"[green]{agg.success_rate:.0%}[/green]" if agg.success_rate >= 0.8
            else f"[yellow]{agg.success_rate:.0%}[/yellow]",
        )
        agg_table.add_row("Avg duration", f"{agg.avg_duration_s:.0f}s")
        agg_table.add_row("Avg items", f"{agg.avg_items:.1f}")
        if agg.total_cost_usd > 0:
            agg_table.add_row("Total AI cost", f"${agg.total_cost_usd:.4f}")
            agg_table.add_row("Avg cost/run", f"${agg.avg_cost_usd:.4f}")
        console.print()
        console.print(agg_table)



def _handle_setup(log: object) -> None:
    """Launch the interactive setup wizard to create .env and sources.json."""
    import sys
    from pathlib import Path
    from src.setup.wizard import run_wizard

    env_path = Path(".env")
    sources_path = Path("data/sources.json")

    # Guard: Rich prompts require an interactive terminal.
    # In CI or subprocess test environments, stdin is a pipe — exit gracefully.
    if not sys.stdin.isatty():
        print("Setup wizard requires an interactive terminal.")
        print(f"Copy .env.example to {env_path} and edit it manually, or run:")
        print("  news-radar --setup")
        return

    run_wizard(env_path=env_path, sources_path=sources_path)



def _handle_check(settings: object, log: object) -> int:
    """
    Validate the full configuration and report any issues.

    Checks:
      1. AI API key matches the configured model
      2. sources.json exists and is valid
      3. At least one source is enabled
      4. data/ directory is writable
      5. Delivery channels configured (info only)

    Returns 0 if all required checks pass, 1 if any critical check fails.
    """
    from src.config import Settings
    from src.logger import console
    from src.setup.sources_loader import validate_sources_file

    s: Settings = settings  # type: ignore[assignment]

    log.section("Configuration Check")  # type: ignore[attr-defined]
    issues: list[str] = []
    warnings: list[str] = []

    # ---- Check 1: AI key ----
    ai_warnings = s.validate_ai_config()
    if ai_warnings:
        for w in ai_warnings:
            console.print(f"  [red]✗[/red] {w}")
            issues.append(w)
    else:
        console.print(f"  [green]✓[/green] AI key OK (provider: {s.active_model_provider}, model: {s.ai_model})")

    # ---- Check 2: sources.json ----
    src_issues = validate_sources_file(s.sources_file)
    if src_issues:
        for issue in src_issues:
            if "Warning" in issue:
                console.print(f"  [yellow]⚠[/yellow] {issue}")
                warnings.append(issue)
            else:
                console.print(f"  [red]✗[/red] {issue}")
                issues.append(issue)
    else:
        console.print(f"  [green]✓[/green] sources.json OK ({s.sources_file})")

    # ---- Check 3: data/ directory writable ----
    try:
        s.data_dir.mkdir(parents=True, exist_ok=True)
        test_file = s.data_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        console.print(f"  [green]✓[/green] data/ directory writable ({s.data_dir})")
    except OSError as e:
        msg = f"data/ directory not writable: {e}"
        console.print(f"  [red]✗[/red] {msg}")
        issues.append(msg)

    # ---- Check 4: Delivery channels (informational) ----
    channels = []
    if s.has_email:
        channels.append("email")
    if s.has_discord:
        channels.append("discord")
    if s.has_slack:
        channels.append("slack")
    if s.custom_webhook_url:
        channels.append("custom")
    if channels:
        console.print(f"  [green]✓[/green] Delivery: {', '.join(channels)}")
    else:
        console.print("  [dim]–[/dim] No delivery channels configured (briefing saved to docs/ only)")

    # ---- Check 5: GitHub Pages output dir ----
    if s.github_pages_enabled:
        console.print(f"  [green]✓[/green] GitHub Pages enabled → {s.docs_dir}")
    else:
        console.print("  [dim]–[/dim] GitHub Pages output disabled")

    # ---- Summary ----
    console.print()
    if issues:
        console.print(f"[red]✗ {len(issues)} issue(s) found. Fix before running --run.[/red]")
        return 1
    if warnings:
        console.print(f"[yellow]⚠ {len(warnings)} warning(s). Pipeline will run but may produce no results.[/yellow]")
    else:
        console.print("[green]✓ All checks passed. Ready to run: news-radar --run[/green]")
    return 0


async def _handle_run(
    settings: object,
    log: object,
    *,
    dry_run: bool = False,
    target_date: str | None = None,
    sources_file: str | None = None,
    enrich_context: bool = True,
) -> int:
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

    if not enrich_context:
        log.info("Context enrichment disabled (--no-enrich)")  # type: ignore[attr-defined]

    # ---- Run the pipeline ----
    orc = Orchestrator(s)
    briefing = await orc.run(
        dry_run=dry_run,
        target_date=parsed_date,
        sources_override=sources_path if sources_file else None,
        enrich_context=enrich_context,
    )

    if briefing is None:
        log.error("Pipeline failed — see logs above for details.")  # type: ignore[attr-defined]
        return 1

    return 0


# ---------------------------------------------------------------------------
# --briefing handler
# ---------------------------------------------------------------------------


def _handle_briefing(settings: "Settings", log: object) -> None:
    """
    Load and pretty-print the most recent briefing using Rich.

    Prints:
      - Briefing date and executive summary
      - Per-topic-cluster story groups with headlines and AI summaries
      - Footer stats (total fetched, scored, generated_at)
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich import print as rprint

    from src.storage.briefing_store import BriefingStore
    from src.briefing import BriefingBuilder

    console = Console()
    store = BriefingStore(settings.data_dir)
    briefing = store.load_latest()

    if briefing is None:
        console.print(
            "[yellow]No briefing found.[/yellow] "
            "Run [bold]news-radar --run[/bold] to generate one."
        )
        return

    # ---- Header ----
    console.print()
    console.print(Rule(f"[bold cyan]📡 News Radar Briefing — {briefing.date}[/bold cyan]"))
    console.print()

    # ---- Executive summary ----
    if briefing.executive_summary:
        console.print(
            Panel(
                briefing.executive_summary,
                title="[bold]Executive Summary[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()

    # ---- Topic clusters ----
    clusters = BriefingBuilder.cluster_items(briefing.items, top_n=5)

    if clusters:
        console.print(Rule("[bold]Stories by Topic[/bold]"))
        console.print()
        for cluster in clusters:
            console.print(f"[bold magenta]▸ {cluster.name}[/bold magenta] ({cluster.size} {'story' if cluster.size == 1 else 'stories'})")  # noqa: E501
            for si in cluster.items:
                score = si.scored.ai_score
                headline = si.ai_headline or si.title
                source = si.scored.item.source_name
                summary_text = (si.ai_summary or "")[:120].rstrip()
                if summary_text and len(si.ai_summary or "") > 120:
                    summary_text += "…"
                console.print(f"  [bold]{headline}[/bold] [dim]({source} · {score}/10)[/dim]")
                if summary_text:
                    console.print(f"  [dim]{summary_text}[/dim]")
                console.print(f"  [link={si.scored.item.url}]{si.scored.item.url}[/link]")
                console.print()
    else:
        # No clusters → flat list
        console.print(Rule("[bold]Stories[/bold]"))
        console.print()
        for si in briefing.items:
            headline = si.ai_headline or si.title
            score = si.scored.ai_score
            console.print(f"  [bold]{headline}[/bold] [dim]({si.scored.item.source_name} · {score}/10)[/dim]")
            console.print(f"  [link={si.scored.item.url}]{si.scored.item.url}[/link]")
            console.print()

    # ---- Footer ----
    console.print(Rule())
    topics_str = ", ".join(briefing.top_topics) or "(none)"
    console.print(
        f"[dim]Top topics: {topics_str}  |  "
        f"{briefing.total_fetched} fetched · {briefing.total_scored} scored · "
        f"{len(briefing.items)} in briefing  |  "
        f"Generated: {briefing.generated_at.strftime('%Y-%m-%d %H:%M UTC')}[/dim]"
    )
    console.print()



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

        elif args.briefing:
            _handle_briefing(settings, log)

        elif args.setup:
            _handle_setup(log)

        elif args.check:
            exit_code = _handle_check(settings, log)
            sys.exit(exit_code)

        elif args.run:
            exit_code = asyncio.run(_handle_run(
                settings, log,
                dry_run=False,
                target_date=args.date,
                sources_file=args.sources,
                enrich_context=not args.no_enrich,
            ))
            sys.exit(exit_code)

        elif args.dry_run:
            exit_code = asyncio.run(_handle_run(
                settings, log,
                dry_run=True,
                target_date=args.date,
                sources_file=args.sources,
                enrich_context=not args.no_enrich,
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
