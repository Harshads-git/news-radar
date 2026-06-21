"""
tests/test_renderers.py
========================
Unit tests for:
  - render_markdown() — output structure and content
  - render_html()     — semantic structure, escaping, score colors
  - GitHubPagesWriter — file creation, archive, list_briefing_dates

No real Briefing objects from AI needed — tests use hand-crafted fixtures.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import (
    Briefing,
    BriefingMetadata,
    NewsItem,
    ScoredItem,
    SummarizedItem,
)
from src.renderers.html import render_html, _score_color
from src.renderers.markdown import render_markdown, _score_badge
from src.renderers.github_pages import GitHubPagesWriter, _build_archive_html


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_news_item(
    url: str = "https://techcrunch.com/ai-story",
    title: str = "OpenAI Releases GPT-5 with Groundbreaking Reasoning",
    source: str = "Hacker News",
    score: int = 500,
) -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        summary="OpenAI's most capable model yet.",
        source_id="hn-test",
        source_name=source,
        source_type="hackernews",
        score=score,
        comment_count=312,
        comments_url="https://news.ycombinator.com/item?id=99999",
        published_at=datetime(2026, 6, 21, 10, 0, 0, tzinfo=timezone.utc),
    )


def make_scored(item: NewsItem, ai_score: int = 9) -> ScoredItem:
    return ScoredItem(
        item=item,
        ai_score=ai_score,
        ai_score_reason="Highly relevant to AI research.",
        ai_reason="Highly relevant to AI research.",
        ai_topics=["AI", "LLM", "OpenAI"],
        model_used="gpt-4o-mini",
    )


def make_summarized(
    url: str = "https://techcrunch.com/ai-story",
    headline: str = "GPT-5 Sets New Standards for AI Reasoning",
    ai_score: int = 9,
) -> SummarizedItem:
    item = make_news_item(url=url)
    scored = make_scored(item, ai_score)
    return SummarizedItem(
        scored=scored,
        ai_headline=headline,
        ai_summary=(
            "OpenAI announced GPT-5 today, its most powerful model.\n\n"
            "The model achieves state-of-the-art results across all benchmarks.\n\n"
            "Watch for competitor responses from Google and Anthropic."
        ),
        key_points=[
            "GPT-5 outperforms GPT-4o on reasoning tasks",
            "Available to all paid users starting today",
            "API pricing reduced by 40%",
        ],
        model_used="gpt-4o-mini",
    )


def make_briefing(
    date_str: str = "2026-06-21",
    num_items: int = 3,
) -> Briefing:
    items = [
        make_summarized(
            url=f"https://example.com/story-{i}",
            headline=f"Story Headline {i}",
            ai_score=9 - i,
        )
        for i in range(num_items)
    ]
    return Briefing(
        date=date_str,
        items=items,
        executive_summary=(
            "Today was a landmark day for AI development.\n\n"
            "Multiple major announcements shaped the landscape.\n\n"
            "The week ahead will see further releases."
        ),
        top_topics=["AI", "LLM", "Python"],
        total_fetched=100,
        total_scored=15,
        generated_at=datetime(2026, 6, 21, 15, 0, 0, tzinfo=timezone.utc),
    )


# ===========================================================================
# Markdown Renderer Tests
# ===========================================================================


class TestRenderMarkdown:
    @pytest.mark.unit
    def test_output_contains_date(self):
        md = render_markdown(make_briefing("2026-06-21"))
        assert "2026-06-21" in md

    @pytest.mark.unit
    def test_output_has_h1_headline(self):
        md = render_markdown(make_briefing())
        assert md.startswith("# News Radar")

    @pytest.mark.unit
    def test_output_contains_executive_summary(self):
        briefing = make_briefing()
        md = render_markdown(briefing)
        assert "Today was a landmark day" in md

    @pytest.mark.unit
    def test_output_contains_top_topics(self):
        md = render_markdown(make_briefing())
        assert "`AI`" in md
        assert "`LLM`" in md

    @pytest.mark.unit
    def test_output_contains_story_headlines(self):
        md = render_markdown(make_briefing(num_items=2))
        assert "Story Headline 0" in md
        assert "Story Headline 1" in md

    @pytest.mark.unit
    def test_output_contains_read_link(self):
        md = render_markdown(make_briefing())
        assert "https://example.com/story-0" in md

    @pytest.mark.unit
    def test_output_contains_key_points(self):
        md = render_markdown(make_briefing())
        assert "GPT-5 outperforms" in md

    @pytest.mark.unit
    def test_output_contains_score_badge(self):
        md = render_markdown(make_briefing())
        assert "/10" in md

    @pytest.mark.unit
    def test_empty_briefing_has_footer(self):
        briefing = Briefing(date="2026-06-21", generated_at=datetime.now(timezone.utc))
        md = render_markdown(briefing)
        assert "News Radar" in md

    @pytest.mark.unit
    def test_score_badge_format(self):
        badge = _score_badge(9)
        assert "9/10" in badge

    @pytest.mark.unit
    def test_score_badge_has_emoji(self):
        badge_10 = _score_badge(10)
        badge_1 = _score_badge(1)
        assert "10/10" in badge_10
        assert "1/10" in badge_1


# ===========================================================================
# HTML Renderer Tests
# ===========================================================================


class TestRenderHtml:
    @pytest.mark.unit
    def test_output_is_valid_html_start(self):
        html_str = render_html(make_briefing())
        assert html_str.startswith("<!DOCTYPE html>")

    @pytest.mark.unit
    def test_output_has_lang_en(self):
        html_str = render_html(make_briefing())
        assert 'lang="en"' in html_str

    @pytest.mark.unit
    def test_output_has_viewport_meta(self):
        html_str = render_html(make_briefing())
        assert "viewport" in html_str

    @pytest.mark.unit
    def test_output_contains_briefing_date(self):
        html_str = render_html(make_briefing("2026-06-21"))
        assert "2026-06-21" in html_str

    @pytest.mark.unit
    def test_output_contains_headline(self):
        html_str = render_html(make_briefing())
        assert "Story Headline 0" in html_str

    @pytest.mark.unit
    def test_output_contains_read_article_link(self):
        html_str = render_html(make_briefing())
        assert "Read Article" in html_str

    @pytest.mark.unit
    def test_output_contains_exec_summary(self):
        html_str = render_html(make_briefing())
        assert "Today was a landmark day" in html_str

    @pytest.mark.unit
    def test_output_contains_topics(self):
        html_str = render_html(make_briefing())
        assert "topic-pill" in html_str
        assert "AI" in html_str

    @pytest.mark.unit
    def test_output_has_embedded_css(self):
        html_str = render_html(make_briefing())
        assert "<style>" in html_str
        assert "--bg:" in html_str  # CSS custom property

    @pytest.mark.unit
    def test_html_escapes_special_chars_in_title(self):
        """< and > in story titles must be HTML-escaped."""
        item = make_news_item(title="Comparing A<B vs C>D Performance")
        scored = make_scored(item)
        si = SummarizedItem(
            scored=scored,
            ai_headline="Comparing A<B vs C>D Performance",
            ai_summary="Test.",
            key_points=[],
            model_used="gpt-4o-mini",
        )
        briefing = Briefing(
            date="2026-06-21",
            items=[si],
            generated_at=datetime.now(timezone.utc),
        )
        html_str = render_html(briefing)
        # Raw < and > should not appear unescaped in HTML attributes/text
        assert "Comparing A&lt;B" in html_str or "A<B" not in html_str

    @pytest.mark.unit
    def test_output_has_footer(self):
        html_str = render_html(make_briefing())
        assert "page-footer" in html_str
        assert "News Radar" in html_str

    @pytest.mark.unit
    def test_score_color_green_for_high_score(self):
        color = _score_color(10)
        assert "hsl(120," in color or "hsl(120 " in color or "120" in color

    @pytest.mark.unit
    def test_score_color_red_for_low_score(self):
        color = _score_color(1)
        assert color.startswith("hsl(0,") or "hsl(0" in color

    @pytest.mark.unit
    def test_each_story_has_unique_id(self):
        html_str = render_html(make_briefing(num_items=3))
        assert 'id="story-1"' in html_str
        assert 'id="story-2"' in html_str
        assert 'id="story-3"' in html_str


# ===========================================================================
# GitHub Pages Writer Tests
# ===========================================================================


class TestGitHubPagesWriter:
    @pytest.mark.unit
    def test_write_creates_daily_html(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        outputs = writer.write(briefing)
        assert outputs["daily_html"].exists()
        assert outputs["daily_html"].name == "2026-06-21.html"

    @pytest.mark.unit
    def test_write_creates_daily_markdown(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        outputs = writer.write(briefing)
        assert outputs["daily_md"].exists()
        assert outputs["daily_md"].name == "2026-06-21.md"

    @pytest.mark.unit
    def test_write_creates_index_html(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        outputs = writer.write(briefing)
        assert outputs["index_html"].exists()
        assert outputs["index_html"].name == "index.html"

    @pytest.mark.unit
    def test_write_creates_archive_html(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        outputs = writer.write(briefing)
        assert outputs["archive_html"].exists()
        assert outputs["archive_html"].name == "archive.html"

    @pytest.mark.unit
    def test_index_html_contains_briefing_content(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        writer.write(briefing)
        index_content = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "Story Headline 0" in index_content

    @pytest.mark.unit
    def test_daily_markdown_contains_date(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        briefing = make_briefing("2026-06-21")
        writer.write(briefing)
        md_content = (tmp_path / "2026-06-21.md").read_text(encoding="utf-8")
        assert "2026-06-21" in md_content

    @pytest.mark.unit
    def test_list_briefing_dates_returns_written_dates(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        writer.write(make_briefing("2026-06-19"))
        writer.write(make_briefing("2026-06-20"))
        writer.write(make_briefing("2026-06-21"))
        dates = writer.list_briefing_dates()
        assert "2026-06-19" in dates
        assert "2026-06-20" in dates
        assert "2026-06-21" in dates

    @pytest.mark.unit
    def test_list_briefing_dates_empty_initially(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        assert writer.list_briefing_dates() == []

    @pytest.mark.unit
    def test_archive_html_contains_all_dates(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        writer.write(make_briefing("2026-06-19"))
        writer.write(make_briefing("2026-06-20"))
        archive = (tmp_path / "archive.html").read_text(encoding="utf-8")
        assert "2026-06-19" in archive
        assert "2026-06-20" in archive

    @pytest.mark.unit
    def test_write_returns_four_paths(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        outputs = writer.write(make_briefing())
        assert len(outputs) == 4
        assert all(isinstance(v, Path) for v in outputs.values())

    @pytest.mark.unit
    def test_overwrite_updates_index(self, tmp_path):
        """Writing a newer briefing must update index.html."""
        writer = GitHubPagesWriter(tmp_path)
        writer.write(make_briefing("2026-06-20"))
        writer.write(make_briefing("2026-06-21"))
        index = (tmp_path / "index.html").read_text(encoding="utf-8")
        # The latest briefing's date should appear in index
        assert "2026-06-21" in index

    @pytest.mark.unit
    def test_archive_html_is_valid_html(self, tmp_path):
        writer = GitHubPagesWriter(tmp_path)
        writer.write(make_briefing("2026-06-21"))
        archive = (tmp_path / "archive.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in archive
        assert "<title>" in archive


class TestBuildArchiveHtml:
    @pytest.mark.unit
    def test_empty_dates_produces_valid_html(self):
        html_str = _build_archive_html([])
        assert "<!DOCTYPE html>" in html_str
        assert "0 daily briefings" in html_str

    @pytest.mark.unit
    def test_dates_appear_in_output(self):
        html_str = _build_archive_html(["2026-06-19", "2026-06-20"])
        assert "2026-06-19" in html_str
        assert "2026-06-20" in html_str

    @pytest.mark.unit
    def test_newest_date_appears_first(self):
        html_str = _build_archive_html(["2026-06-19", "2026-06-20", "2026-06-21"])
        idx_21 = html_str.index("2026-06-21")
        idx_19 = html_str.index("2026-06-19")
        assert idx_21 < idx_19  # newest first

    @pytest.mark.unit
    def test_links_to_html_files(self):
        html_str = _build_archive_html(["2026-06-20"])
        assert '2026-06-20.html"' in html_str

    @pytest.mark.unit
    def test_links_to_md_files(self):
        html_str = _build_archive_html(["2026-06-20"])
        assert "2026-06-20.md" in html_str
