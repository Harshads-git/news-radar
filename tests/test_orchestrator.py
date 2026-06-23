"""
tests/test_orchestrator.py
===========================
Unit tests for the pipeline Orchestrator.

Strategy: mock every stage's external dependency (AI, scrapers, storage)
so the orchestrator's coordination logic can be tested in isolation.

Tests cover:
  - RunStats dataclass serialization
  - Orchestrator.run(): success path, dry_run, empty results, errors
  - Stage isolation: one scraper failing doesn't abort the whole run
  - _record_run(): run_log.json written correctly
  - main.py _handle_run(): arg forwarding and exit codes
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator import Orchestrator, RunStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> MagicMock:
    """Return a mock Settings object wired to a temp dir."""
    s = MagicMock()
    s.ai_model = "gpt-4o-mini"
    s.score_threshold = 6
    s.max_briefing_items = 20
    s.user_interests = "AI, Python"
    s.output_language = "English"
    s.sources_file = tmp_path / "sources.json"
    s.data_dir = tmp_path / "data"
    s.docs_dir = tmp_path / "docs"
    s.github_pages_enabled = True
    s.has_openai = True
    s.has_gemini = False
    s.has_anthropic = False
    s.has_any_ai_key = True
    s.validate_ai_config.return_value = []
    s.active_model_provider = "openai"
    return s


def make_minimal_sources_json(path: Path) -> None:
    """Write a minimal sources.json with one HN source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sources": [{
            "id": "hn-test",
            "type": "hackernews",
            "name": "HN Test",
            "enabled": True,
            "limit": 5,
            "tags": [],
        }]
    }), encoding="utf-8")


def make_news_item(url: str = "https://example.com/story") -> MagicMock:
    item = MagicMock()
    item.url = url
    item.title = f"Story at {url}"
    item.summary = "Summary text"
    item.source_name = "HN"
    item.source_type = "hackernews"
    item.score = 100
    item.comment_count = 50
    item.comments_url = None
    item.published_at = datetime(2026, 6, 23, 10, 0, 0, tzinfo=timezone.utc)
    item.tags = []
    return item


def make_scored_item(url: str = "https://example.com/story", ai_score: int = 8) -> MagicMock:
    si = MagicMock()
    si.ai_score = ai_score
    si.item = make_news_item(url)
    si.ai_topics = ["AI"]
    si.ai_reason = "Relevant"
    si.model_used = "gpt-4o-mini"
    return si


def make_summarized_item(url: str = "https://example.com/story") -> MagicMock:
    su = MagicMock()
    su.ai_headline = "Test Headline"
    su.ai_summary = "Summary."
    su.key_points = ["Point 1"]
    su.model_used = "gpt-4o-mini"
    su.scored = make_scored_item(url)
    su.scored.item.url = url
    return su


def make_briefing(date_str: str = "2026-06-23") -> MagicMock:
    b = MagicMock()
    b.date = date_str
    b.items = [make_summarized_item()]
    b.executive_summary = "Test exec summary"
    b.top_topics = ["AI"]
    b.total_fetched = 10
    b.total_scored = 3
    b.model_dump_json.return_value = json.dumps({"date": date_str, "items": []})
    return b


# ===========================================================================
# RunStats Tests
# ===========================================================================


class TestRunStats:
    @pytest.mark.unit
    def test_to_dict_has_required_keys(self):
        stats = RunStats(date="2026-06-23", dry_run=False)
        d = stats.to_dict()
        assert "date" in d
        assert "dry_run" in d
        assert "duration_s" in d
        assert "fetched" in d
        assert "status" in d
        assert "errors" in d

    @pytest.mark.unit
    def test_to_dict_is_json_serializable(self):
        stats = RunStats(date="2026-06-23", errors=["some error"])
        d = stats.to_dict()
        json_str = json.dumps(d)  # must not raise
        assert "2026-06-23" in json_str

    @pytest.mark.unit
    def test_default_status_is_pending(self):
        stats = RunStats()
        assert stats.status == "pending"

    @pytest.mark.unit
    def test_errors_start_empty(self):
        stats = RunStats()
        assert stats.errors == []

    @pytest.mark.unit
    def test_can_append_to_errors(self):
        stats = RunStats()
        stats.errors.append("something went wrong")
        assert len(stats.errors) == 1


# ===========================================================================
# Orchestrator.run() — Happy Path
# ===========================================================================


class TestOrchestratorHappyPath:
    @pytest.mark.unit
    async def test_run_returns_briefing_on_success(self, tmp_path):
        """Full happy-path: all stages succeed → returns a Briefing."""
        settings = make_settings(tmp_path)
        make_minimal_sources_json(settings.sources_file)

        news_items = [make_news_item(f"https://example.com/{i}") for i in range(3)]
        scored_items = [make_scored_item(f"https://example.com/{i}") for i in range(3)]
        summarized_items = [make_summarized_item(f"https://example.com/{i}") for i in range(3)]
        briefing = make_briefing()

        with (
            patch("src.orchestrator.ScraperFactory") as MockSF,
            patch("src.orchestrator.Deduplicator") as MockDedup,
            patch("src.orchestrator.NewsScorer") as MockScorer,
            patch("src.orchestrator.NewsSummarizer") as MockSumm,
            patch("src.orchestrator.BriefingBuilder") as MockBuilder,
            patch("src.orchestrator.BriefingStore") as MockStore,
            patch("src.orchestrator.GitHubPagesWriter") as MockWriter,
            patch("src.orchestrator.AIProviderFactory") as MockAI,
        ):
            # Wire mocks
            scraper = MagicMock()
            scraper.fetch = AsyncMock(return_value=news_items)
            MockSF.create.return_value = scraper
            MockDedup.return_value.deduplicate.return_value = news_items
            MockScorer.return_value.score_all = AsyncMock(return_value=scored_items)
            MockSumm.return_value.summarize_all = AsyncMock(return_value=summarized_items)
            MockBuilder.return_value.build = AsyncMock(return_value=briefing)
            store_inst = MagicMock()
            store_inst.save.return_value = tmp_path / "2026-06-23.json"
            MockStore.return_value = store_inst
            MockWriter.return_value.write.return_value = {}

            orc = Orchestrator(settings)
            result = await orc.run()

        assert result is not None
        assert result is briefing

    @pytest.mark.unit
    async def test_dry_run_skips_store_and_render(self, tmp_path):
        """dry_run=True must skip BriefingStore.save() and GitHubPagesWriter.write()."""
        settings = make_settings(tmp_path)
        make_minimal_sources_json(settings.sources_file)

        news_items = [make_news_item()]
        scored_items = [make_scored_item()]
        summarized_items = [make_summarized_item()]
        briefing = make_briefing()

        with (
            patch("src.orchestrator.ScraperFactory") as MockSF,
            patch("src.orchestrator.Deduplicator") as MockDedup,
            patch("src.orchestrator.NewsScorer") as MockScorer,
            patch("src.orchestrator.NewsSummarizer") as MockSumm,
            patch("src.orchestrator.BriefingBuilder") as MockBuilder,
            patch("src.orchestrator.BriefingStore") as MockStore,
            patch("src.orchestrator.GitHubPagesWriter") as MockWriter,
            patch("src.orchestrator.AIProviderFactory"),
        ):
            scraper = MagicMock()
            scraper.fetch = AsyncMock(return_value=news_items)
            MockSF.create.return_value = scraper
            MockDedup.return_value.deduplicate.return_value = news_items
            MockScorer.return_value.score_all = AsyncMock(return_value=scored_items)
            MockSumm.return_value.summarize_all = AsyncMock(return_value=summarized_items)
            MockBuilder.return_value.build = AsyncMock(return_value=briefing)

            orc = Orchestrator(settings)
            result = await orc.run(dry_run=True)

        assert result is not None
        # Store and writer should NOT have been called
        MockStore.return_value.save.assert_not_called()
        MockWriter.return_value.write.assert_not_called()


# ===========================================================================
# Orchestrator — Error Handling and Edge Cases
# ===========================================================================


class TestOrchestratorErrors:
    @pytest.mark.unit
    async def test_one_failed_scraper_does_not_abort_run(self, tmp_path):
        """If one scraper fails, remaining results are still processed."""
        settings = make_settings(tmp_path)
        # Two sources in sources.json
        settings.sources_file.parent.mkdir(parents=True, exist_ok=True)
        settings.sources_file.write_text(json.dumps({
            "sources": [
                {"id": "hn-1", "type": "hackernews", "name": "HN 1", "enabled": True, "limit": 5, "tags": []},
                {"id": "hn-2", "type": "hackernews", "name": "HN 2", "enabled": True, "limit": 5, "tags": []},
            ]
        }), encoding="utf-8")

        good_scraper = MagicMock()
        good_scraper.fetch = AsyncMock(return_value=[make_news_item()])
        bad_scraper = MagicMock()
        bad_scraper.fetch = AsyncMock(side_effect=Exception("network failure"))

        briefing = make_briefing()

        with (
            patch("src.orchestrator.ScraperFactory") as MockSF,
            patch("src.orchestrator.Deduplicator") as MockDedup,
            patch("src.orchestrator.NewsScorer") as MockScorer,
            patch("src.orchestrator.NewsSummarizer") as MockSumm,
            patch("src.orchestrator.BriefingBuilder") as MockBuilder,
            patch("src.orchestrator.BriefingStore"),
            patch("src.orchestrator.GitHubPagesWriter"),
            patch("src.orchestrator.AIProviderFactory"),
        ):
            MockSF.create.side_effect = [good_scraper, bad_scraper]
            MockDedup.return_value.deduplicate.return_value = [make_news_item()]
            MockScorer.return_value.score_all = AsyncMock(return_value=[make_scored_item()])
            MockSumm.return_value.summarize_all = AsyncMock(return_value=[make_summarized_item()])
            MockBuilder.return_value.build = AsyncMock(return_value=briefing)

            orc = Orchestrator(settings)
            result = await orc.run()

        # Pipeline must complete despite one failed scraper
        assert result is not None

    @pytest.mark.unit
    async def test_no_items_after_dedup_returns_empty_briefing(self, tmp_path):
        """When dedup removes all items, an empty briefing is returned."""
        settings = make_settings(tmp_path)
        make_minimal_sources_json(settings.sources_file)

        with (
            patch("src.orchestrator.ScraperFactory") as MockSF,
            patch("src.orchestrator.Deduplicator") as MockDedup,
            patch("src.orchestrator.BriefingStore"),
            patch("src.orchestrator.GitHubPagesWriter"),
            patch("src.orchestrator.AIProviderFactory"),
        ):
            scraper = MagicMock()
            scraper.fetch = AsyncMock(return_value=[make_news_item()])
            MockSF.create.return_value = scraper
            MockDedup.return_value.deduplicate.return_value = []  # all dupes

            orc = Orchestrator(settings)
            result = await orc.run()

        assert result is not None
        assert result.items == [] or len(result.items) == 0

    @pytest.mark.unit
    async def test_no_items_after_scoring_returns_empty_briefing(self, tmp_path):
        """When all items score below threshold, an empty briefing is returned."""
        settings = make_settings(tmp_path)
        make_minimal_sources_json(settings.sources_file)

        with (
            patch("src.orchestrator.ScraperFactory") as MockSF,
            patch("src.orchestrator.Deduplicator") as MockDedup,
            patch("src.orchestrator.NewsScorer") as MockScorer,
            patch("src.orchestrator.BriefingStore"),
            patch("src.orchestrator.GitHubPagesWriter"),
            patch("src.orchestrator.AIProviderFactory"),
        ):
            scraper = MagicMock()
            scraper.fetch = AsyncMock(return_value=[make_news_item()])
            MockSF.create.return_value = scraper
            MockDedup.return_value.deduplicate.return_value = [make_news_item()]
            MockScorer.return_value.score_all = AsyncMock(return_value=[])  # nothing passed

            orc = Orchestrator(settings)
            result = await orc.run()

        assert result is not None

    @pytest.mark.unit
    async def test_run_returns_none_on_fatal_error(self, tmp_path):
        """A truly fatal error (e.g. sources file missing) returns None."""
        settings = make_settings(tmp_path)
        # Don't create sources.json → loader will raise

        with patch("src.orchestrator.AIProviderFactory"):
            orc = Orchestrator(settings)
            result = await orc.run()

        assert result is None


# ===========================================================================
# _record_run Tests
# ===========================================================================


class TestRecordRun:
    @pytest.mark.unit
    def test_creates_run_log_json(self, tmp_path):
        settings = make_settings(tmp_path)
        settings.data_dir = tmp_path / "data"

        orc = Orchestrator(settings)
        stats = RunStats(date="2026-06-23", status="success", in_briefing=3)
        orc._record_run(stats)

        run_log = tmp_path / "data" / "run_log.json"
        assert run_log.exists()

    @pytest.mark.unit
    def test_run_log_contains_run_data(self, tmp_path):
        settings = make_settings(tmp_path)
        settings.data_dir = tmp_path / "data"

        orc = Orchestrator(settings)
        stats = RunStats(date="2026-06-23", status="success", in_briefing=5)
        orc._record_run(stats)

        data = json.loads((tmp_path / "data" / "run_log.json").read_text())
        assert len(data) == 1
        assert data[0]["date"] == "2026-06-23"
        assert data[0]["status"] == "success"
        assert data[0]["in_briefing"] == 5

    @pytest.mark.unit
    def test_run_log_appends_multiple_runs(self, tmp_path):
        settings = make_settings(tmp_path)
        settings.data_dir = tmp_path / "data"
        orc = Orchestrator(settings)

        orc._record_run(RunStats(date="2026-06-21", status="success"))
        orc._record_run(RunStats(date="2026-06-22", status="success"))
        orc._record_run(RunStats(date="2026-06-23", status="error"))

        data = json.loads((tmp_path / "data" / "run_log.json").read_text())
        assert len(data) == 3
        assert data[-1]["date"] == "2026-06-23"

    @pytest.mark.unit
    def test_run_log_keeps_at_most_90_runs(self, tmp_path):
        settings = make_settings(tmp_path)
        settings.data_dir = tmp_path / "data"
        orc = Orchestrator(settings)

        for i in range(95):
            orc._record_run(RunStats(date=f"run-{i:03}", status="success"))

        data = json.loads((tmp_path / "data" / "run_log.json").read_text())
        assert len(data) == 90
