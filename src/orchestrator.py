"""
src/orchestrator.py
====================
The main pipeline orchestrator — wires all stages into one async run().

Pipeline Stages:
  1. FETCH    — Run all enabled scrapers concurrently → list[NewsItem]
  2. DEDUPE   — URL-normalize + Jaccard dedup → deduplicated list[NewsItem]
  3. SCORE    — AI scoring, filter by threshold → list[ScoredItem]
  4. SUMMARIZE— AI summarization → list[SummarizedItem]
  5. BUILD    — Assemble Briefing with exec summary → Briefing
  6. STORE    — Save Briefing as JSON to data/briefings/
  7. RENDER   — Write HTML + MD to docs/ (GitHub Pages)
  8. LOG      — Record run stats to data/run_log.json

Each stage reports timing. The orchestrator catches stage errors, logs
them, and (where possible) continues to the next stage so one failed
source doesn't abort the entire run.

Usage:
    from src.orchestrator import Orchestrator
    from src.config import settings

    orc = Orchestrator(settings)
    briefing = await orc.run()

Or for a dry run (no saves, no delivery):
    briefing = await orc.run(dry_run=True)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.ai import AIProviderFactory
from src.ai.scorer import NewsScorer
from src.ai.summarizer import NewsSummarizer
from src.briefing import BriefingBuilder
from src.delivery.dispatcher import DeliveryDispatcher
from src.deduplicator import Deduplicator
from src.logger import get_logger
from src.renderers.github_pages import GitHubPagesWriter
from src.scrapers import ScraperFactory
from src.setup.sources_loader import load_sources
from src.storage import BriefingStore

if TYPE_CHECKING:
    from src.config import Settings
    from src.models import Briefing, NewsItem, ScoredItem, SummarizedItem

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline run stats (stored to data/run_log.json)
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """Timing and count statistics for one pipeline run."""

    date: str = ""
    dry_run: bool = False
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    status: str = "pending"  # pending | success | error

    # Stage counts
    fetched: int = 0
    after_dedup: int = 0
    scored: int = 0
    in_briefing: int = 0

    # Stage timings (seconds)
    t_fetch: float = 0.0
    t_dedup: float = 0.0
    t_score: float = 0.0
    t_summarize: float = 0.0
    t_build: float = 0.0
    t_render: float = 0.0

    # Error notes
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Connects all pipeline stages and runs them in sequence.

    Parameters
    ----------
    settings:
        Application configuration (from src.config.settings).
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    async def run(
        self,
        *,
        dry_run: bool = False,
        target_date: date | None = None,
        sources_override: Path | None = None,
    ) -> "Briefing | None":
        """
        Run the full pipeline end-to-end.

        Parameters
        ----------
        dry_run:
            If True, skip saving and rendering (fetch + score + summarize only).
        target_date:
            Date for the briefing (default: today UTC).
        sources_override:
            Override the sources file path (for CLI --sources flag).

        Returns
        -------
        Briefing | None
            The completed briefing, or None if a fatal error occurred.
        """
        stats = RunStats(
            date=(target_date or date.today()).isoformat(),
            dry_run=dry_run,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        pipeline_start = time.monotonic()

        log.pipeline_start(stats.date, sources=0)
        if dry_run:
            log.warning("DRY-RUN: pipeline will not save or deliver output")

        try:
            briefing = await self._run_stages(stats, target_date, sources_override, dry_run)
            stats.status = "success"
            return briefing

        except Exception as e:
            log.exception("Fatal pipeline error: %s", e)
            stats.status = "error"
            stats.errors.append(str(e))
            return None

        finally:
            stats.duration_s = time.monotonic() - pipeline_start
            stats.finished_at = datetime.now(timezone.utc).isoformat()
            log.pipeline_end(items=stats.in_briefing, duration=stats.duration_s)
            self._record_run(stats)

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    async def _run_stages(
        self,
        stats: RunStats,
        target_date: date | None,
        sources_override: Path | None,
        dry_run: bool,
    ) -> "Briefing":
        """Execute all pipeline stages in sequence."""
        s = self.settings

        # ------ STAGE 1: FETCH ------
        news_items = await self._stage_fetch(stats, sources_override)

        # ------ STAGE 2: DEDUP ------
        deduped = self._stage_dedup(stats, news_items)

        # If no items, build an empty briefing
        if not deduped:
            log.warning("No items after deduplication — building empty briefing")
            return await self._build_empty_briefing(stats, target_date, dry_run)

        # ------ STAGE 3: SCORE ------
        provider = self._get_ai_provider()
        scored_items = await self._stage_score(stats, deduped, provider)

        if not scored_items:
            log.warning("No items passed score threshold — building empty briefing")
            return await self._build_empty_briefing(stats, target_date, dry_run)

        # ------ STAGE 4: SUMMARIZE ------
        web_contexts = await self._fetch_all_contexts(scored_items)
        summarized = await self._stage_summarize(stats, scored_items, web_contexts, provider)

        # ------ STAGE 5: BUILD BRIEFING ------
        briefing = await self._stage_build(stats, summarized, target_date, provider)

        # ------ STAGE 6: STORE (skip if dry_run) ------
        if not dry_run:
            self._stage_store(stats, briefing)

        # ------ STAGE 7: RENDER (skip if dry_run) ------
        if not dry_run:
            self._stage_render(stats, briefing)

        # ------ STAGE 8: DELIVER ------
        if not dry_run:
            await self._stage_deliver(briefing)

        return briefing

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    async def _stage_fetch(
        self, stats: RunStats, sources_override: Path | None
    ) -> list["NewsItem"]:
        """Stage 1: Run all scrapers concurrently."""
        # TODO(#12): investigate connection pooling - currently each scraper
        # creates its own httpx client, which means 14 separate TCP handshakes
        # even with asyncio.gather. Should use a shared AsyncClient.
        t0 = time.monotonic()
        sources_path = sources_override or self.settings.sources_file
        sources_config = load_sources(sources_path)
        enabled = sources_config.enabled_sources

        log.pipeline_start(stats.date, sources=len(enabled))
        log.section("Stage 1: Fetching")
        log.info("Running %d scrapers", len(enabled))

        # Build and run all scrapers concurrently
        scraper_tasks = []
        for src in enabled:
            try:
                scraper = ScraperFactory.create(src)
                scraper_tasks.append(scraper.fetch())
            except Exception as e:
                log.warning("Could not create scraper for '%s': %s", src.id, e)
                stats.errors.append(f"Scraper creation failed for {src.id}: {e}")

        if not scraper_tasks:
            log.warning("No scrapers could be created")
            stats.fetched = 0
            stats.t_fetch = time.monotonic() - t0
            return []

        # Gather results — continue even if some scrapers fail
        results = await asyncio.gather(*scraper_tasks, return_exceptions=True)

        all_items: list["NewsItem"] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                src_name = enabled[i].name if i < len(enabled) else f"source[{i}]"
                log.warning("Scraper '%s' failed: %s", src_name, result)
                stats.errors.append(f"Scraper {src_name}: {result}")
            elif isinstance(result, list):
                all_items.extend(result)

        stats.fetched = len(all_items)
        stats.t_fetch = time.monotonic() - t0
        log.success("Fetched %d items in %.1fs", stats.fetched, stats.t_fetch)
        return all_items

    def _stage_dedup(
        self, stats: RunStats, items: list["NewsItem"]
    ) -> list["NewsItem"]:
        """Stage 2: Deduplicate by URL and title similarity."""
        # FIXME(#2): Jaccard threshold is too aggressive when stories share
        # a common prefix like 'OpenAI releases X' vs 'OpenAI releases Y'
        # Consider weighting URL domain into the similarity calculation.
        t0 = time.monotonic()
        log.section("Stage 2: Deduplication")

        deduplicator = Deduplicator()
        deduped = deduplicator.deduplicate(items)

        stats.after_dedup = len(deduped)
        stats.t_dedup = time.monotonic() - t0
        log.success(
            "Deduped: %d → %d items (removed %d) in %.1fs",
            len(items),
            len(deduped),
            len(items) - len(deduped),
            stats.t_dedup,
        )
        return deduped

    async def _stage_score(
        self,
        stats: RunStats,
        items: list["NewsItem"],
        provider: object,
    ) -> list["ScoredItem"]:
        """Stage 3: AI scoring + threshold filtering."""
        # TODO(#5): add a disk cache for scores so reruns don't re-call the API
        # data/cache/YYYY-MM-DD-scores.json with 24h TTL
        t0 = time.monotonic()
        log.section("Stage 3: AI Scoring")

        scorer = NewsScorer(provider, self.settings)  # type: ignore[arg-type]
        scored = await scorer.score_all(items, fetch_context=True)

        stats.scored = len(scored)
        stats.t_score = time.monotonic() - t0
        log.success(
            "Scored: %d/%d items passed threshold in %.1fs",
            stats.scored,
            len(items),
            stats.t_score,
        )
        return scored

    async def _fetch_all_contexts(
        self, scored_items: list["ScoredItem"]
    ) -> dict[str, str]:
        """Pre-fetch web context for all scored items (best-effort)."""
        try:
            from src.search import fetch_web_context

            async def _ctx(item: "ScoredItem") -> tuple[str, str]:
                try:
                    ctx = await fetch_web_context(item.item.title)
                    return item.item.url, ctx
                except Exception:
                    return item.item.url, ""

            pairs = await asyncio.gather(*[_ctx(si) for si in scored_items])
            return dict(pairs)
        except Exception as e:
            log.debug("Web context fetch failed: %s", e)
            return {}

    async def _stage_summarize(
        self,
        stats: RunStats,
        scored_items: list["ScoredItem"],
        web_contexts: dict[str, str],
        provider: object,
    ) -> list["SummarizedItem"]:
        """Stage 4: AI summarization of scored items."""
        t0 = time.monotonic()
        log.section("Stage 4: AI Summarization")

        summarizer = NewsSummarizer(provider, self.settings)  # type: ignore[arg-type]
        summarized = await summarizer.summarize_all(scored_items, web_contexts=web_contexts)

        stats.t_summarize = time.monotonic() - t0
        log.success("Summarized %d items in %.1fs", len(summarized), stats.t_summarize)
        return summarized

    async def _stage_build(
        self,
        stats: RunStats,
        summarized: list["SummarizedItem"],
        target_date: date | None,
        provider: object,
    ) -> "Briefing":
        """Stage 5: Assemble Briefing with executive summary."""
        t0 = time.monotonic()
        log.section("Stage 5: Briefing Assembly")

        builder = BriefingBuilder(provider, self.settings)  # type: ignore[arg-type]
        briefing = await builder.build(
            summarized,
            total_fetched=stats.fetched,
            total_scored=stats.scored,
            briefing_date=target_date,
        )

        stats.in_briefing = len(briefing.items)
        stats.t_build = time.monotonic() - t0
        log.success("Briefing built: %d items in %.1fs", stats.in_briefing, stats.t_build)
        return briefing

    def _stage_store(self, stats: RunStats, briefing: "Briefing") -> None:
        """Stage 6: Save briefing to JSON storage."""
        try:
            store = BriefingStore(self.settings.data_dir)
            path = store.save(briefing)
            log.success("Briefing saved: %s", path.name)
        except Exception as e:
            log.error("Failed to save briefing: %s", e)
            stats.errors.append(f"Storage: {e}")

    def _stage_render(self, stats: RunStats, briefing: "Briefing") -> None:
        """Stage 7: Render to HTML + Markdown for GitHub Pages."""
        if not self.settings.github_pages_enabled:
            log.debug("GitHub Pages output disabled — skipping render")
            return

        t0 = time.monotonic()
        try:
            writer = GitHubPagesWriter(self.settings.docs_dir)
            outputs = writer.write(briefing)
            stats.t_render = time.monotonic() - t0
            log.success(
                "Rendered %d files in %.1fs (index.html updated)",
                len(outputs),
                stats.t_render,
            )
        except Exception as e:
            log.error("Render failed: %s", e)
            stats.errors.append(f"Render: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _stage_deliver(self, briefing: "Briefing") -> None:
        """Stage 8: Deliver briefing to all configured channels."""
        dispatcher = DeliveryDispatcher(self.settings)
        if not dispatcher.has_any_channel():
            log.info("No delivery channels configured — skipping")
            return
        await dispatcher.dispatch(briefing)

    def _get_ai_provider(self) -> object:
        """Initialize the AI provider from settings."""
        provider = AIProviderFactory.from_model(self.settings.ai_model)
        log.debug("AI provider: %s (model: %s)", type(provider).__name__, self.settings.ai_model)
        return provider

    async def _build_empty_briefing(
        self,
        stats: RunStats,
        target_date: date | None,
        dry_run: bool,
    ) -> "Briefing":
        """Build and optionally store an empty briefing (no items passed)."""
        from src.models import Briefing

        briefing = Briefing(
            date=(target_date or date.today()).isoformat(),
            items=[],
            executive_summary="No stories met the relevance threshold today.",
            total_fetched=stats.fetched,
            total_scored=0,
            generated_at=datetime.now(timezone.utc),
        )

        if not dry_run:
            self._stage_store(stats, briefing)

        return briefing

    def _record_run(self, stats: RunStats) -> None:
        """Append run stats to data/run_log.json for --status display."""
        run_log_path = self.settings.data_dir / "run_log.json"
        try:
            run_log_path.parent.mkdir(parents=True, exist_ok=True)
            runs: list[dict] = []
            if run_log_path.exists():
                try:
                    runs = json.loads(run_log_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    runs = []

            runs.append(stats.to_dict())
            # Keep only last 90 runs
            runs = runs[-90:]
            run_log_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        except Exception as e:
            log.debug("Could not write run_log.json: %s", e)
