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
  news-radar --run              Run the full pipeline (fetch → score → summarize → deliver)
  news-radar --dry-run          Run pipeline but skip saving and delivery
  news-radar --setup            Launch the interactive setup wizard
  news-radar --status           Show last run info and current configuration
  news-radar --briefing         Print the most recent briefing to the terminal
  news-radar --sources-list     Display all configured sources with status
  news-radar --config           Show full configuration and active delivery channels
  news-radar --source-stats     Show per-source fetch health and error history
  news-radar --cache-stats      Show AI score cache hit rate, size, and TTL
  news-radar --cost-report      Show AI API cost report: daily and weekly spend
  news-radar --retry-stats      Show circuit breaker events and throttle history
  news-radar --preview-email    Render latest briefing as email HTML, open in browser
  news-radar --check            Validate config and API keys (exit 0 if OK)
  news-radar --version          Print version and exit

  # Advanced:
  news-radar --run --date 2026-07-10        Re-run pipeline for a specific date
  news-radar --run --no-enrich             Skip web context fetching (faster)
  news-radar --run --log-level DEBUG       Verbose pipeline output
  news-radar --dry-run --sources custom.json  Test with a different sources file
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
    action.add_argument(
        "--sources-list",
        dest="sources_list",
        action="store_true",
        help="Display all configured news sources with type, status, and item limit.",
    )
    action.add_argument(
        "--config",
        action="store_true",
        help="Show full configuration: all settings, directories, and delivery channel status.",
    )
    action.add_argument(
        "--source-stats",
        dest="source_stats",
        action="store_true",
        help="Show per-source fetch health: attempts, errors, consecutive failures, and item counts.",
    )
    action.add_argument(
        "--cache-stats",
        dest="cache_stats",
        action="store_true",
        help="Show AI score cache statistics: hit rate, entry count, file size, and TTL.",
    )
    action.add_argument(
        "--cost-report",
        dest="cost_report",
        action="store_true",
        help="Show AI API cost report: daily and weekly spend from cost_log.jsonl.",
    )
    action.add_argument(
        "--retry-stats",
        dest="retry_stats",
        action="store_true",
        help="Show retry budget history: circuit breaker events and throttle changes.",
    )
    action.add_argument(
        "--preview-email",
        dest="preview_email",
        action="store_true",
        help="Render the latest briefing as an email HTML preview and open in browser.",
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
        try:
            from src.__version__ import __version__
            ver = __version__
        except Exception:
            ver = "1.0.0"
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


# ---------------------------------------------------------------------------
# --sources-list handler
# ---------------------------------------------------------------------------


def _handle_sources_list(settings: "Settings", log: object) -> None:
    """
    Display all configured news sources in a Rich table.

    Shows for each source:
      - Source ID
      - Type (rss / hackernews / reddit / github)
      - Display Name
      - Enabled / disabled status
      - Item limit (max stories to fetch)
      - Tags
      - URL or subreddit (where applicable)

    Reads from settings.sources_file. If the file does not exist,
    prints a helpful message prompting the user to run --setup.
    """
    import json
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule

    console = Console()
    sources_path = settings.sources_file

    console.print()
    console.print(Rule(f"[bold cyan]📡 Configured Sources — {sources_path}[/bold cyan]"))
    console.print()

    if not sources_path.exists():
        console.print(
            f"[yellow]Sources file not found:[/yellow] {sources_path}\n"
            "Run [bold]news-radar --setup[/bold] to create it."
        )
        return

    try:
        raw = json.loads(sources_path.read_text(encoding="utf-8"))
        sources = raw.get("sources", [])
    except Exception as e:
        console.print(f"[red]Failed to read sources file:[/red] {e}")
        return

    if not sources:
        console.print("[dim]No sources configured.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        row_styles=["", "dim"],
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", min_width=11)
    table.add_column("Name", min_width=18)
    table.add_column("Status", min_width=8, justify="center")
    table.add_column("Limit", min_width=5, justify="right")
    table.add_column("Tags", min_width=14)
    table.add_column("URL / Subreddit", overflow="fold")

    enabled_count = 0
    for src in sources:
        enabled = src.get("enabled", True)
        if enabled:
            enabled_count += 1

        status = "[green]● enabled[/green]" if enabled else "[red]○ disabled[/red]"
        tags = ", ".join(src.get("tags", [])) or "[dim]—[/dim]"
        limit = str(src.get("limit", 30))

        src_type = src.get("type", "?")
        if src_type == "reddit":
            location = f"r/{src.get('subreddit', '?')} ({src.get('sort', 'hot')})"
        elif src_type == "hackernews":
            location = "[dim]HN API[/dim]"
        elif src_type == "github":
            location = "[dim]GitHub Trending[/dim]"
        else:
            url = src.get("url", "")
            location = url if url else "[dim]—[/dim]"

        table.add_row(
            src.get("id", "?"),
            src_type,
            src.get("name", "?"),
            status,
            limit,
            tags,
            location,
        )

    console.print(table)
    console.print()
    console.print(
        f"[dim]{enabled_count}/{len(sources)} sources enabled  •  "
        f"Edit [bold]{sources_path}[/bold] to add or disable sources[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# --config handler
# ---------------------------------------------------------------------------


def _handle_config(settings: "Settings", log: object) -> None:
    """
    Display the full current configuration in a structured Rich layout.

    Organised into sections:
      1. Core Pipeline settings (model, threshold, limits, language)
      2. Storage & Paths (data dir, briefings dir, cache dir, docs dir)
      3. AI Provider status (which keys are present, active provider)
      4. Delivery Channels (email, Discord, Slack, webhook, GitHub Pages)
      5. Advanced (log level, user interests)

    Values come from settings (environment / .env file). API keys are
    shown as present/missing without exposing the actual key value.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.panel import Panel

    console = Console()
    s = settings

    console.print()
    console.print(Rule("[bold cyan]📡 News Radar Configuration[/bold cyan]"))
    console.print()

    # ---- Section 1: Core pipeline ----
    core = Table(title="⚙️  Pipeline", show_header=False, box=None, padding=(0, 2))
    core.add_column("Setting", style="dim", min_width=24)
    core.add_column("Value", style="white")
    core.add_row("AI Model", f"[bold]{s.ai_model}[/bold]")
    core.add_row("Active Provider", s.active_model_provider)
    core.add_row("Score Threshold", f"{s.score_threshold} / 10")
    core.add_row("Max Briefing Items", str(s.max_briefing_items))
    core.add_row("Output Language", s.output_language)
    console.print(core)
    console.print()

    # ---- Section 2: Storage & Paths ----
    paths = Table(title="📁  Storage & Paths", show_header=False, box=None, padding=(0, 2))
    paths.add_column("Name", style="dim", min_width=24)
    paths.add_column("Path", style="white")

    def _path_status(p: "Path") -> str:
        from pathlib import Path as P
        pp = P(p)
        return f"{pp}  [green]✓[/green]" if pp.exists() else f"{pp}  [dim](not yet created)[/dim]"

    paths.add_row("Sources File", _path_status(s.sources_file))
    paths.add_row("Data Dir", _path_status(s.data_dir))
    paths.add_row("Briefings Dir", _path_status(s.briefings_dir))
    paths.add_row("Cache Dir", _path_status(s.cache_dir))
    paths.add_row("Docs Dir", _path_status(s.docs_dir))
    console.print(paths)
    console.print()

    # ---- Section 3: AI Provider Keys ----
    ai = Table(title="🤖  AI Provider Keys", show_header=False, box=None, padding=(0, 2))
    ai.add_column("Provider", style="dim", min_width=24)
    ai.add_column("Status", style="white")

    def _key_status(present: bool, label: str) -> str:
        return f"[green]● configured[/green]  ({label})" if present else "[red]○ not set[/red]"

    ai.add_row("OpenAI", _key_status(s.has_openai, "OPENAI_API_KEY"))
    ai.add_row("Google Gemini", _key_status(s.has_gemini, "GEMINI_API_KEY"))
    ai.add_row("Anthropic Claude", _key_status(s.has_anthropic, "ANTHROPIC_API_KEY"))
    console.print(ai)
    console.print()

    # ---- Section 4: Delivery Channels ----
    delivery = Table(title="📬  Delivery Channels", show_header=False, box=None, padding=(0, 2))
    delivery.add_column("Channel", style="dim", min_width=24)
    delivery.add_column("Status", style="white")

    def _delivery(active: bool, hint: str = "") -> str:
        if active:
            return "[green]● active[/green]"
        note = f"  [dim]({hint})[/dim]" if hint else ""
        return f"[dim]○ inactive[/dim]{note}"

    delivery.add_row("Email (SMTP)", _delivery(s.has_email, "set SMTP_USER + SMTP_PASSWORD + EMAIL_TO"))
    delivery.add_row("Discord Webhook", _delivery(s.has_discord, "set DISCORD_WEBHOOK_URL"))
    delivery.add_row("Slack Webhook", _delivery(s.has_slack, "set SLACK_WEBHOOK_URL"))
    delivery.add_row(
        "Custom Webhook",
        _delivery(bool(s.custom_webhook_url), "set CUSTOM_WEBHOOK_URL"),
    )
    delivery.add_row(
        "GitHub Pages",
        "[green]● enabled[/green]" if s.github_pages_enabled else "[dim]○ disabled[/dim]",
    )
    console.print(delivery)
    console.print()

    # ---- Section 5: User Interests & Logging ----
    adv = Table(title="🔧  Advanced", show_header=False, box=None, padding=(0, 2))
    adv.add_column("Setting", style="dim", min_width=24)
    adv.add_column("Value", style="white")
    adv.add_row("Log Level", s.log_level)
    # Truncate very long interests for display
    interests_display = s.user_interests
    if len(interests_display) > 80:
        interests_display = interests_display[:77] + "..."
    adv.add_row("User Interests", interests_display)
    console.print(adv)

    console.print()
    console.print(
        "[dim]Settings are loaded from [bold].env[/bold] in the project root. "
        "Run [bold]news-radar --setup[/bold] to reconfigure.[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# --source-stats handler
# ---------------------------------------------------------------------------


def _handle_source_stats(settings: "Settings", log: object) -> None:
    """
    Display per-source fetch health statistics using a Rich table.

    Reads from ``data/source_health.jsonl`` (written by the pipeline on
    every run). Shows for the last 30 days:
      - Source ID
      - Total fetch attempts
      - Total successes / errors
      - Success rate (colour-coded: green ≥90%, yellow ≥70%, red <70%)
      - Average items per successful fetch
      - Consecutive error streak (red if ≥ threshold)
      - Last recorded error (truncated)
      - Last seen date

    If no data exists yet, prompts the user to run the pipeline first.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule

    from src.pipeline.source_health import (
        SourceHealthTracker,
        CONSECUTIVE_ERROR_THRESHOLD,
    )

    console = Console()
    health = SourceHealthTracker(settings.data_dir)
    summary = health.source_summary(days=30)

    console.print()
    console.print(Rule("[bold cyan]📡 Source Health Statistics (last 30 days)[/bold cyan]"))
    console.print()

    if not summary:
        console.print(
            "[yellow]No source health data found.[/yellow]\n"
            "Run [bold]news-radar --run[/bold] to collect statistics."
        )
        console.print()
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        row_styles=["", "dim"],
    )
    table.add_column("Source ID", style="cyan", no_wrap=True)
    table.add_column("Attempts", justify="right", min_width=8)
    table.add_column("OK / Err", justify="right", min_width=9)
    table.add_column("Rate", justify="right", min_width=7)
    table.add_column("Avg Items", justify="right", min_width=9)
    table.add_column("Consec Err", justify="right", min_width=10)
    table.add_column("Last Seen", min_width=10)
    table.add_column("Last Error", overflow="fold", min_width=20)

    for source_id, stats in sorted(summary.items()):
        rate = stats["success_rate"]
        rate_str = f"{rate:.0%}"
        if rate >= 0.9:
            rate_colored = f"[green]{rate_str}[/green]"
        elif rate >= 0.7:
            rate_colored = f"[yellow]{rate_str}[/yellow]"
        else:
            rate_colored = f"[red]{rate_str}[/red]"

        consec = stats["consecutive_errors"]
        if consec >= CONSECUTIVE_ERROR_THRESHOLD:
            consec_str = f"[bold red]{consec} ⚠[/bold red]"
        elif consec > 0:
            consec_str = f"[yellow]{consec}[/yellow]"
        else:
            consec_str = "[green]0[/green]"

        ok_err = f"{stats['total_successes']} / {stats['total_errors']}"
        last_error = stats.get("last_error") or "[dim]—[/dim]"
        if last_error and len(last_error) > 50:
            last_error = last_error[:47] + "..."

        table.add_row(
            source_id,
            str(stats["total_attempts"]),
            ok_err,
            rate_colored,
            f"{stats['avg_items']:.1f}",
            consec_str,
            stats.get("last_seen", "?"),
            last_error,
        )

    console.print(table)
    console.print()

    # Summary line
    total_sources = len(summary)
    healthy = sum(1 for s in summary.values() if s["consecutive_errors"] == 0)
    at_risk = sum(
        1 for s in summary.values()
        if 0 < s["consecutive_errors"] < CONSECUTIVE_ERROR_THRESHOLD
    )
    disabled_candidates = sum(
        1 for s in summary.values()
        if s["consecutive_errors"] >= CONSECUTIVE_ERROR_THRESHOLD
    )

    parts = [f"[dim]{total_sources} sources tracked"]
    parts.append(f"[green]{healthy} healthy[/green]")
    if at_risk:
        parts.append(f"[yellow]{at_risk} at risk[/yellow]")
    if disabled_candidates:
        parts.append(
            f"[bold red]{disabled_candidates} should be disabled[/bold red] "
            f"(≥{CONSECUTIVE_ERROR_THRESHOLD} consecutive errors)"
        )
    console.print("  " + "  •  ".join(parts) + "[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# --cache-stats handler
# ---------------------------------------------------------------------------


def _handle_cache_stats(settings: "Settings", log: object) -> None:
    """
    Display AI score cache statistics using Rich.

    Shows:
      - Cache file path and existence
      - Entry count (valid vs expired)
      - Hit rate (colour-coded: green ≥50%, yellow ≥20%, red <20%)
      - File size on disk
      - TTL setting
      - Top-10 most recently cached entries (url, score, topics, age)

    If the cache file does not exist yet, prompts the user to run the pipeline.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich.panel import Panel

    from src.storage.score_cache import ScoreCache, DEFAULT_TTL_HOURS

    console = Console()
    cache = ScoreCache(settings.data_dir)
    stats = cache.stats()

    console.print()
    console.print(Rule("[bold cyan]🗄️  AI Score Cache Statistics[/bold cyan]"))
    console.print()

    cache_path = settings.data_dir / "cache" / "score_cache.json"
    if not cache_path.exists():
        console.print(
            "[yellow]Cache file not found.[/yellow]\n"
            "Run [bold]news-radar --run[/bold] to populate the cache."
        )
        console.print()
        return

    # ---- Summary table ----
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="dim", min_width=22)
    summary.add_column("Value", style="white")

    summary.add_row("Cache File", str(cache_path))
    summary.add_row("Valid Entries", str(stats["entry_count"]))
    summary.add_row("Expired Entries", str(stats["expired_count"]))
    summary.add_row("File Size", f"{stats['file_size_kb']:.1f} KB")
    summary.add_row("TTL", f"{stats['ttl_hours']}h")

    hit_rate = stats["hit_rate"]
    hit_rate_str = f"{hit_rate:.0%}"
    if hit_rate >= 0.5:
        hit_rate_colored = f"[green]{hit_rate_str}[/green]"
    elif hit_rate >= 0.2:
        hit_rate_colored = f"[yellow]{hit_rate_str}[/yellow]"
    else:
        hit_rate_colored = f"[dim]{hit_rate_str}[/dim]"

    summary.add_row("Hit Rate (this session)", hit_rate_colored)
    summary.add_row("Hits / Misses", f"{stats['hit_count']} / {stats['miss_count']}")
    console.print(summary)
    console.print()

    # ---- Entries table (top 15 by most recently cached) ----
    with cache._lock:
        all_entries = list(cache._store.values())

    if not all_entries:
        console.print("[dim]No entries in cache.[/dim]")
        console.print()
        return

    # Sort by cached_at descending (most recent first)
    valid_entries = [e for e in all_entries if not e.is_expired]
    valid_entries.sort(key=lambda e: e.cached_at, reverse=True)
    display_entries = valid_entries[:15]

    entries_table = Table(
        title="Most Recent Cache Entries",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    entries_table.add_column("Score", justify="center", min_width=5)
    entries_table.add_column("Age", justify="right", min_width=7)
    entries_table.add_column("Hits", justify="right", min_width=4)
    entries_table.add_column("Topics", min_width=16)
    entries_table.add_column("URL", overflow="fold", min_width=30)

    for entry in display_entries:
        age_h = entry.age_hours
        if age_h < 1:
            age_str = f"{age_h * 60:.0f}m"
        else:
            age_str = f"{age_h:.1f}h"

        score = entry.ai_score
        score_colored = (
            f"[green]{score}[/green]" if score >= 7
            else f"[yellow]{score}[/yellow]" if score >= 4
            else f"[dim]{score}[/dim]"
        )
        topics_str = ", ".join(entry.ai_topics[:3]) if entry.ai_topics else "[dim]—[/dim]"

        entries_table.add_row(
            score_colored,
            age_str,
            str(entry.hits),
            topics_str,
            entry.url,
        )

    console.print(entries_table)
    console.print()
    console.print(
        f"[dim]Showing {len(display_entries)}/{len(valid_entries)} valid entries  •  "
        f"Cache auto-expires after {DEFAULT_TTL_HOURS}h[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# --cost-report handler
# ---------------------------------------------------------------------------


def _handle_cost_report(settings: "Settings", log: object) -> None:
    """
    Display AI API cost report using Rich tables.

    Shows:
      1. 30-day daily spend table (date, runs, tokens, calls, cost)
      2. 4-week weekly summary table
      3. 30-day total spend footer

    Cost values are colour-coded by relative daily spend:
      - Green  : ≤ avg spend / 2  (cheap day)
      - Yellow : ≤ avg spend × 1.5
      - Red    : > avg spend × 1.5 (expensive day)

    If cost_log.jsonl does not exist yet, prompts the user to run the pipeline.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule

    from src.pipeline.cost_ledger import CostLedger

    console = Console()
    ledger = CostLedger(settings.data_dir)

    console.print()
    console.print(Rule("[bold cyan]💰  AI Cost Report[/bold cyan]"))
    console.print()

    ledger_path = settings.data_dir / "cost_log.jsonl"
    if not ledger_path.exists():
        console.print(
            "[yellow]No cost data found.[/yellow]\n"
            "Run [bold]news-radar --run[/bold] to start tracking costs."
        )
        console.print()
        return

    # ---- Daily report ----
    daily = ledger.daily_report(days=30)
    if not daily:
        console.print("[dim]No cost entries in the last 30 days.[/dim]")
        console.print()
        return

    # Compute average daily cost for colour-coding
    avg_cost = sum(r["cost_usd"] for r in daily) / len(daily) if daily else 0.0

    daily_table = Table(
        title="Daily Spend (last 30 days)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    daily_table.add_column("Date", min_width=12)
    daily_table.add_column("Runs", justify="right", min_width=5)
    daily_table.add_column("Tokens", justify="right", min_width=9)
    daily_table.add_column("Calls", justify="right", min_width=7)
    daily_table.add_column("Cost (USD)", justify="right", min_width=11)
    daily_table.add_column("Dry", justify="center", min_width=4)

    for row in daily:
        cost = row["cost_usd"]
        cost_str = f"${cost:.6f}"
        if avg_cost > 0 and cost > avg_cost * 1.5:
            cost_col = f"[red]{cost_str}[/red]"
        elif avg_cost > 0 and cost > avg_cost / 2:
            cost_col = f"[yellow]{cost_str}[/yellow]"
        else:
            cost_col = f"[green]{cost_str}[/green]"

        dry_col = f"[dim]{row['dry_runs']}[/dim]" if row["dry_runs"] else "[dim]—[/dim]"

        daily_table.add_row(
            row["date"],
            str(row["runs"]),
            f"{row['total_tokens']:,}",
            str(row["total_calls"]),
            cost_col,
            dry_col,
        )

    console.print(daily_table)
    console.print()

    # ---- Weekly summary ----
    weekly = ledger.weekly_summary(weeks=4)
    if weekly:
        week_table = Table(
            title="Weekly Summary (last 4 weeks)",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        week_table.add_column("Week", min_width=10)
        week_table.add_column("Runs", justify="right", min_width=5)
        week_table.add_column("Tokens", justify="right", min_width=9)
        week_table.add_column("Cost (USD)", justify="right", min_width=11)

        for row in weekly:
            week_table.add_row(
                row["week"],
                str(row["runs"]),
                f"{row['total_tokens']:,}",
                f"${row['cost_usd']:.6f}",
            )

        console.print(week_table)
        console.print()

    # ---- 30-day total ----
    total_30d = ledger.total_spend(days=30)
    total_entries = ledger.load_entries(days=30)
    total_runs = sum(1 for e in total_entries)
    total_tokens = sum(e.get("total_tokens", 0) for e in total_entries)

    console.print(
        f"  [dim]30-day total: [bold]${total_30d:.6f}[/bold]  "
        f"({total_runs} runs · {total_tokens:,} tokens)[/dim]"
    )
    console.print(
        f"  [dim]Cost log: {ledger_path}[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# --retry-stats handler
# ---------------------------------------------------------------------------


def _handle_retry_stats(settings: "Settings", log: object) -> None:
    """
    Display retry budget and circuit breaker history using Rich.

    Shows:
      - Current circuit state for the configured AI provider
      - Event summary (throttle_down, throttle_up, circuit_open, circuit_close)
      - Last 30 days of circuit events in a chronological table

    If retry_budget.jsonl does not exist yet, prompts the user to run the pipeline.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule

    from src.pipeline.retry_budget import RetryBudget, CircuitState

    console = Console()

    # Use the configured AI model name as provider identifier
    provider_name = getattr(settings, "ai_model", "openai").split("/")[-1]
    budget = RetryBudget(settings.data_dir, provider_name=provider_name)

    console.print()
    console.print(Rule("[bold cyan]🔄  Retry Budget & Circuit Breaker History[/bold cyan]"))
    console.print()

    budget_path = settings.data_dir / "retry_budget.jsonl"
    if not budget_path.exists():
        console.print(
            "[yellow]No retry budget history found.[/yellow]\n"
            "Run [bold]news-radar --run[/bold] to start tracking circuit events."
        )
        console.print()
        return

    # ---- Event summary ----
    summary = budget.event_summary(days=30)
    if not summary:
        console.print("[dim]No circuit events in the last 30 days.[/dim]")
        console.print()
        return

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Event", style="dim", min_width=22)
    summary_table.add_column("Count", style="white")

    event_styles = {
        "circuit_open": "[bold red]",
        "circuit_close": "[green]",
        "circuit_half_open": "[yellow]",
        "throttle_down": "[yellow]",
        "throttle_up": "[green]",
        "probe_success": "[green]",
        "probe_failure": "[red]",
    }
    for event_type, count in sorted(summary.items()):
        style = event_styles.get(event_type, "")
        end = "[/]" if style else ""
        label = event_type.replace("_", " ").title()
        summary_table.add_row(label, f"{style}{count}{end}")

    console.print(summary_table)
    console.print()

    # ---- Event history table ----
    history = budget.load_history(days=30)

    events_table = Table(
        title="Circuit Events (last 30 days, newest first)",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    events_table.add_column("Timestamp", min_width=20)
    events_table.add_column("Provider", min_width=12)
    events_table.add_column("Event", min_width=16)
    events_table.add_column("Change", min_width=14)
    events_table.add_column("Reason", overflow="fold")

    for row in history[:40]:  # cap at 40 rows
        et = row.get("event_type", "?")
        style_open = event_styles.get(et, "")
        style_close = "[/]" if style_open else ""

        old_val = str(row.get("old_value", "?"))
        new_val = str(row.get("new_value", "?"))
        change = f"{old_val} → {new_val}"

        events_table.add_row(
            row.get("timestamp", "?"),
            row.get("provider", "?"),
            f"{style_open}{et}{style_close}",
            change,
            row.get("reason", ""),
        )

    console.print(events_table)
    console.print()
    console.print(
        f"  [dim]Budget file: {budget_path}[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# --preview-email handler
# ---------------------------------------------------------------------------


def _handle_preview_email(settings: "Settings", log: object) -> None:
    """
    Render the latest briefing as an email HTML preview and open in browser.

    Steps:
      1. Load the most recent briefing from the data store
      2. Render it with render_email_html() (inline styles, table layout)
      3. Save to data/email_preview.html
      4. Open in the default system browser via webbrowser.open()
      5. Print a Rich confirmation with the file path

    If no briefing exists yet, instruct the user to run the pipeline.
    """
    import webbrowser
    from rich.console import Console
    from rich.rule import Rule

    from src.delivery.email_template import render_email_html
    from src.storage import BriefingStore

    console = Console()
    console.print()
    console.print(Rule("[bold cyan]\U0001f4e7  Email Preview[/bold cyan]"))
    console.print()

    store = BriefingStore(settings.data_dir)
    briefing = store.load_latest()

    if briefing is None:
        console.print(
            "[yellow]No briefings found.[/yellow]\n"
            "Run [bold]news-radar --run[/bold] first to generate a briefing."
        )
        console.print()
        return

    # Render with email-optimized template
    html_str = render_email_html(briefing)

    # Save preview file
    preview_path = settings.data_dir / "email_preview.html"
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(html_str, encoding="utf-8")
    except OSError as e:
        console.print(f"[red]Could not write preview file: {e}[/red]")
        console.print()
        return

    # Open in browser
    file_url = preview_path.resolve().as_uri()
    opened = webbrowser.open(file_url)

    # Summary
    count = len(briefing.items)
    console.print(
        f"  [green]\u2713[/green] Email preview saved: [bold]{preview_path}[/bold]"
    )
    console.print(
        f"  [dim]Date: {briefing.date} \u00b7 {count} stories[/dim]"
    )
    if opened:
        console.print("  [dim]Opened in system browser.[/dim]")
    else:
        console.print(
            f"  [yellow]Could not auto-open browser.[/yellow] "
            f"Open manually: [bold]{file_url}[/bold]"
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

        elif args.sources_list:
            _handle_sources_list(settings, log)

        elif args.config:
            _handle_config(settings, log)

        elif args.source_stats:
            _handle_source_stats(settings, log)

        elif args.cache_stats:
            _handle_cache_stats(settings, log)

        elif args.cost_report:
            _handle_cost_report(settings, log)

        elif args.retry_stats:
            _handle_retry_stats(settings, log)

        elif args.preview_email:
            _handle_preview_email(settings, log)

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
