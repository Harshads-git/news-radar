"""
tests/test_briefing_clustering.py
===================================
Tests for Day 22 briefing upgrades in src/briefing.py.

Coverage:
  - TopicCluster dataclass: name, size, top_item, short_description
  - BriefingBuilder.cluster_items(): empty list, no-topics items, single topic,
    multi-topic dedup (each item in only one cluster), Other cluster for
    unclustered items, sorted by size desc, Other always last, top_n limit,
    display label mapping (ai → AI, llm → LLM etc.)
  - BriefingBuilder._fallback_exec_summary(): no items, with clusters (shows
    themes), without clusters (legacy behavior), top story headline present,
    date in output, cluster parts excluding Other
  - BriefingBuilder._parse_exec_response(): valid JSON, strips markdown fences,
    empty summary raises ValueError, no JSON raises ValueError, themes extracted
  - BriefingBuilder._extract_top_topics(): empty items, top N limit, most
    frequent topics first, case normalisation
  - BriefingBuilder._build_exec_prompt(): contains cluster lines, contains
    story list, contains interests

All tests are pure in-memory — no AI provider calls, no I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.briefing import BriefingBuilder, TopicCluster
from src.models import NewsItem, ScoredItem, SummarizedItem


# ===========================================================================
# Helpers
# ===========================================================================


def make_si(
    title: str,
    topics: list[str],
    score: int = 7,
    source: str = "Test Feed",
    url: str | None = None,
) -> SummarizedItem:
    """Factory: create a SummarizedItem with given topics and score."""
    item = NewsItem(
        source_id="test",
        source_name=source,
        source_type="rss",
        title=title,
        url=url or f"https://example.com/{abs(hash(title)) % 100000}",
    )
    scored = ScoredItem(item=item, ai_score=score, ai_topics=topics)
    return SummarizedItem(
        scored=scored,
        ai_summary=f"Summary of {title}.",
        ai_headline=title,
        ai_key_takeaways=[],
    )


AI_STORY_1 = make_si("OpenAI releases GPT-5", ["AI", "LLM", "OpenAI"], score=9)
AI_STORY_2 = make_si("Gemini 2.0 announced", ["AI", "Google"], score=8)
PYTHON_STORY = make_si("Python 4.0 released", ["Python", "Programming"], score=7)
RUST_STORY = make_si("Rust 2.0 ships", ["Rust", "Programming"], score=6)
NO_TOPIC_STORY = make_si("Random old story", [], score=3)


# ===========================================================================
# TopicCluster
# ===========================================================================


class TestTopicCluster:
    @pytest.mark.unit
    def test_name_stored(self):
        c = TopicCluster(name="AI")
        assert c.name == "AI"

    @pytest.mark.unit
    def test_size_empty(self):
        c = TopicCluster(name="AI")
        assert c.size == 0

    @pytest.mark.unit
    def test_size_with_items(self):
        c = TopicCluster(name="AI", items=[AI_STORY_1, AI_STORY_2])
        assert c.size == 2

    @pytest.mark.unit
    def test_top_item_none_when_empty(self):
        c = TopicCluster(name="AI")
        assert c.top_item is None

    @pytest.mark.unit
    def test_top_item_first_when_items(self):
        c = TopicCluster(name="AI", items=[AI_STORY_1, AI_STORY_2])
        assert c.top_item is AI_STORY_1

    @pytest.mark.unit
    def test_short_description_empty(self):
        c = TopicCluster(name="AI")
        desc = c.short_description()
        assert "AI" in desc
        assert "(no items)" in desc

    @pytest.mark.unit
    def test_short_description_one_story(self):
        c = TopicCluster(name="AI", items=[AI_STORY_1])
        desc = c.short_description()
        assert "1 story" in desc
        assert "OpenAI" in desc

    @pytest.mark.unit
    def test_short_description_plural_stories(self):
        c = TopicCluster(name="AI", items=[AI_STORY_1, AI_STORY_2])
        desc = c.short_description()
        assert "2 stories" in desc


# ===========================================================================
# BriefingBuilder.cluster_items
# ===========================================================================


class TestClusterItems:
    @pytest.mark.unit
    def test_empty_list_returns_empty(self):
        assert BriefingBuilder.cluster_items([]) == []

    @pytest.mark.unit
    def test_all_no_topics_returns_other_cluster(self):
        items = [NO_TOPIC_STORY, make_si("Another story", [])]
        clusters = BriefingBuilder.cluster_items(items)
        assert len(clusters) == 1
        assert clusters[0].name == "Other"
        assert clusters[0].size == 2

    @pytest.mark.unit
    def test_single_topic_creates_one_cluster(self):
        items = [AI_STORY_1, AI_STORY_2]
        clusters = BriefingBuilder.cluster_items(items, top_n=1)
        names = [c.name for c in clusters]
        assert "AI" in names

    @pytest.mark.unit
    def test_each_item_in_at_most_one_cluster(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY, RUST_STORY]
        clusters = BriefingBuilder.cluster_items(items, top_n=5)
        all_items = [si for c in clusters for si in c.items]
        # No item should appear twice
        assert len(all_items) == len(set(id(si) for si in all_items))

    @pytest.mark.unit
    def test_all_items_accounted_for(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY, RUST_STORY, NO_TOPIC_STORY]
        clusters = BriefingBuilder.cluster_items(items, top_n=5)
        total = sum(c.size for c in clusters)
        assert total == len(items)

    @pytest.mark.unit
    def test_other_cluster_contains_no_topic_items(self):
        items = [AI_STORY_1, NO_TOPIC_STORY]
        clusters = BriefingBuilder.cluster_items(items)
        other = next((c for c in clusters if c.name == "Other"), None)
        assert other is not None
        assert NO_TOPIC_STORY in other.items

    @pytest.mark.unit
    def test_other_cluster_is_last(self):
        items = [AI_STORY_1, NO_TOPIC_STORY]
        clusters = BriefingBuilder.cluster_items(items)
        assert clusters[-1].name == "Other"

    @pytest.mark.unit
    def test_sorted_by_size_descending(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY]
        clusters = BriefingBuilder.cluster_items(items, top_n=5)
        main_clusters = [c for c in clusters if c.name != "Other"]
        sizes = [c.size for c in main_clusters]
        assert sizes == sorted(sizes, reverse=True)

    @pytest.mark.unit
    def test_top_n_limits_cluster_count(self):
        items = [
            make_si("AI story", ["AI"], score=9),
            make_si("Python story", ["Python"], score=8),
            make_si("Rust story", ["Rust"], score=7),
            make_si("Go story", ["Go"], score=6),
            make_si("JS story", ["JavaScript"], score=5),
        ]
        clusters = BriefingBuilder.cluster_items(items, top_n=2)
        main_clusters = [c for c in clusters if c.name != "Other"]
        assert len(main_clusters) <= 2

    @pytest.mark.unit
    def test_display_label_ai_uppercase(self):
        items = [make_si("AI item", ["ai"])]
        clusters = BriefingBuilder.cluster_items(items)
        main = [c for c in clusters if c.name != "Other"]
        assert any(c.name == "AI" for c in main)

    @pytest.mark.unit
    def test_display_label_llm_uppercase(self):
        items = [make_si("LLM item", ["llm"])]
        clusters = BriefingBuilder.cluster_items(items)
        main = [c for c in clusters if c.name != "Other"]
        assert any(c.name == "LLM" for c in main)

    @pytest.mark.unit
    def test_no_other_cluster_when_all_matched(self):
        items = [AI_STORY_1, AI_STORY_2]
        clusters = BriefingBuilder.cluster_items(items)
        names = [c.name for c in clusters]
        assert "Other" not in names


# ===========================================================================
# BriefingBuilder._fallback_exec_summary
# ===========================================================================


class TestFallbackExecSummary:
    @pytest.mark.unit
    def test_empty_items_returns_no_stories_message(self):
        result = BriefingBuilder._fallback_exec_summary([], "2026-07-11")
        assert "No significant stories" in result
        assert "2026-07-11" in result

    @pytest.mark.unit
    def test_date_appears_in_output(self):
        result = BriefingBuilder._fallback_exec_summary([AI_STORY_1], "2026-07-11")
        assert "2026-07-11" in result

    @pytest.mark.unit
    def test_top_story_headline_in_output(self):
        result = BriefingBuilder._fallback_exec_summary([AI_STORY_1], "2026-07-11")
        assert "OpenAI" in result or "GPT" in result

    @pytest.mark.unit
    def test_with_clusters_shows_themes(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY]
        clusters = BriefingBuilder.cluster_items(items)
        result = BriefingBuilder._fallback_exec_summary(items, "2026-07-11", clusters)
        assert "AI" in result  # dominant cluster

    @pytest.mark.unit
    def test_without_clusters_no_themes_line(self):
        result = BriefingBuilder._fallback_exec_summary([AI_STORY_1], "2026-07-11")
        assert "themes:" not in result

    @pytest.mark.unit
    def test_other_cluster_excluded_from_themes(self):
        items = [NO_TOPIC_STORY, make_si("Another", [])]
        clusters = BriefingBuilder.cluster_items(items)
        result = BriefingBuilder._fallback_exec_summary(items, "2026-07-11", clusters)
        assert "Other" not in result or "themes" not in result

    @pytest.mark.unit
    def test_item_count_in_output(self):
        items = [AI_STORY_1, PYTHON_STORY, RUST_STORY]
        result = BriefingBuilder._fallback_exec_summary(items, "2026-07-11")
        assert "3" in result


# ===========================================================================
# BriefingBuilder._parse_exec_response
# ===========================================================================


class TestParseExecResponse:
    @pytest.mark.unit
    def test_valid_json_extracts_summary(self):
        raw = '{"executive_summary": "Great AI day!", "top_themes": ["AI", "ML"]}'
        summary, themes = BriefingBuilder._parse_exec_response(raw)
        assert summary == "Great AI day!"
        assert themes == ["AI", "ML"]

    @pytest.mark.unit
    def test_strips_markdown_code_fences(self):
        raw = "```json\n{\"executive_summary\": \"Good day.\", \"top_themes\": []}\n```"
        summary, themes = BriefingBuilder._parse_exec_response(raw)
        assert summary == "Good day."

    @pytest.mark.unit
    def test_missing_exec_summary_raises(self):
        raw = '{"top_themes": ["AI"]}'
        with pytest.raises(ValueError, match="Missing"):
            BriefingBuilder._parse_exec_response(raw)

    @pytest.mark.unit
    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON"):
            BriefingBuilder._parse_exec_response("This is not JSON at all")

    @pytest.mark.unit
    def test_empty_themes_list_ok(self):
        raw = '{"executive_summary": "Good day.", "top_themes": []}'
        summary, themes = BriefingBuilder._parse_exec_response(raw)
        assert themes == []

    @pytest.mark.unit
    def test_themes_stripped_of_whitespace(self):
        raw = '{"executive_summary": "Today.", "top_themes": ["  AI  ", " ML "]}'
        _, themes = BriefingBuilder._parse_exec_response(raw)
        assert themes == ["AI", "ML"]


# ===========================================================================
# BriefingBuilder._extract_top_topics
# ===========================================================================


class TestExtractTopTopics:
    @pytest.mark.unit
    def test_empty_items_returns_empty(self):
        assert BriefingBuilder._extract_top_topics([]) == []

    @pytest.mark.unit
    def test_most_frequent_topic_first(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY]
        topics = BriefingBuilder._extract_top_topics(items)
        assert topics[0] == "ai"  # 'ai' appears in both AI_STORY_1 and AI_STORY_2

    @pytest.mark.unit
    def test_respects_top_n(self):
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY, RUST_STORY]
        topics = BriefingBuilder._extract_top_topics(items, top_n=2)
        assert len(topics) <= 2

    @pytest.mark.unit
    def test_no_topics_items_contribute_nothing(self):
        items = [NO_TOPIC_STORY]
        assert BriefingBuilder._extract_top_topics(items) == []

    @pytest.mark.unit
    def test_topics_normalised_to_lowercase(self):
        item = make_si("AI Story", ["AI", "LLM"])
        topics = BriefingBuilder._extract_top_topics([item])
        assert all(t == t.lower() for t in topics)


# ===========================================================================
# BriefingBuilder._build_exec_prompt
# ===========================================================================


class TestBuildExecPrompt:
    @pytest.mark.unit
    def test_prompt_contains_cluster_lines(self):
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.user_interests = "AI, Python"

        builder = BriefingBuilder(provider=MagicMock(), settings=settings)
        items = [AI_STORY_1, AI_STORY_2, PYTHON_STORY]
        clusters = BriefingBuilder.cluster_items(items)
        prompt = builder._build_exec_prompt(items, clusters)
        assert "AI" in prompt
        assert "cluster" in prompt.lower() or "stories" in prompt.lower()

    @pytest.mark.unit
    def test_prompt_contains_story_list(self):
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.user_interests = "AI"

        builder = BriefingBuilder(provider=MagicMock(), settings=settings)
        items = [AI_STORY_1]
        clusters = BriefingBuilder.cluster_items(items)
        prompt = builder._build_exec_prompt(items, clusters)
        assert "OpenAI" in prompt or "GPT" in prompt

    @pytest.mark.unit
    def test_prompt_contains_user_interests(self):
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.user_interests = "Rust programming language"

        builder = BriefingBuilder(provider=MagicMock(), settings=settings)
        items = [RUST_STORY]
        clusters = BriefingBuilder.cluster_items(items)
        prompt = builder._build_exec_prompt(items, clusters)
        assert "Rust programming language" in prompt
