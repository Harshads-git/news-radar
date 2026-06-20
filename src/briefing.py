"""
src/briefing.py
===============
BriefingBuilder — assembles the final daily Briefing from SummarizedItems.

The BriefingBuilder is the last stage before delivery/storage. It:
  1. Groups summarized items by topic cluster
  2. Generates briefing-level metadata (date, item count, top topics)
  3. Generates an AI-written executive summary of the entire day's news
  4. Returns a complete Briefing object ready for rendering

Architecture:
  NewsItems → [Scraper] → [Deduplicator] → [Scorer] → [Summarizer] → [BriefingBuilder] → Briefing

The Briefing model (defined in src/models.py) is:
  Briefing:
    date: str                    (YYYY-MM-DD)
    items: list[SummarizedItem]
    executive_summary: str       (AI-written 2-3 paragraph overview)
    top_topics: list[str]        (most frequent AI topics across items)
    total_fetched: int
    total_scored: int
    generated_at: datetime

Usage:
    from src.briefing import BriefingBuilder
    builder = BriefingBuilder(provider, settings)
    briefing = await builder.build(summarized_items, stats)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.ai.base import BaseAIProvider
    from src.config import Settings
    from src.models import Briefing, SummarizedItem

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for the executive summary
# ---------------------------------------------------------------------------

_EXEC_SUMMARY_PROMPT = """\
You are an editor writing the opening section of a daily AI/tech news briefing.
The reader is interested in: {interests}

Today's top {item_count} stories (by relevance score):
{story_list}

Write a concise executive summary in this JSON format (no markdown, no preamble):
{{
  "executive_summary": "<2-3 paragraphs: paragraph 1 = today's top theme, paragraph 2 = notable developments, paragraph 3 = what to watch>",
  "top_themes": ["<theme1>", "<theme2>", "<theme3>"]
}}

Guidelines:
- Para 1: What was the dominant theme or story today? Be specific.
- Para 2: What other significant things happened? Connect the dots.
- Para 3: What should the reader pay attention to tomorrow?
- top_themes: 3 overarching themes (e.g. "LLM advancements", "open source tools").
- Keep total length under 300 words.
"""


class BriefingBuilder:
    """
    Assembles the final daily Briefing from a list of SummarizedItems.

    Can optionally generate an AI-written executive summary.
    If AI generation fails, falls back to a template-based summary.
    """

    def __init__(
        self,
        provider: "BaseAIProvider",
        settings: "Settings",
    ) -> None:
        self.provider = provider
        self.settings = settings

    async def build(
        self,
        summarized_items: list["SummarizedItem"],
        *,
        total_fetched: int = 0,
        total_scored: int = 0,
        briefing_date: date | None = None,
        generate_exec_summary: bool = True,
    ) -> "Briefing":
        """
        Build a complete Briefing from summarized items.

        Parameters
        ----------
        summarized_items:
            Items from the summarizer, ordered by relevance score.
        total_fetched:
            Total raw items fetched across all sources (for metadata).
        total_scored:
            Items that passed the score threshold (for metadata).
        briefing_date:
            Date of the briefing. Defaults to today UTC.
        generate_exec_summary:
            If True, uses AI to write the executive summary.
            Set False for tests or when no AI key is available.

        Returns
        -------
        Briefing
            Complete briefing object ready for storage and delivery.
        """
        from src.models import Briefing

        today = briefing_date or date.today()
        date_str = today.isoformat()

        log.section("Phase 4: Briefing Assembly")
        log.info("Building briefing for %s with %d items", date_str, len(summarized_items))

        # Derive top topics from all scored items
        top_topics = self._extract_top_topics(summarized_items)

        # Generate executive summary
        if summarized_items and generate_exec_summary:
            exec_summary = await self._generate_exec_summary(summarized_items)
        else:
            exec_summary = self._fallback_exec_summary(summarized_items, date_str)

        briefing = Briefing(
            date=date_str,
            items=summarized_items,
            executive_summary=exec_summary,
            top_topics=top_topics,
            total_fetched=total_fetched,
            total_scored=total_scored,
            generated_at=datetime.now(timezone.utc),
        )

        log.success(
            "Briefing ready: %d items, %d top topics",
            len(summarized_items),
            len(top_topics),
        )
        return briefing

    # ------------------------------------------------------------------
    # Executive summary generation
    # ------------------------------------------------------------------

    async def _generate_exec_summary(
        self, summarized_items: list["SummarizedItem"]
    ) -> str:
        """Use AI to generate a briefing-level executive summary."""
        prompt = self._build_exec_prompt(summarized_items)

        try:
            raw = await self.provider.complete(
                prompt,
                max_tokens=500,
                temperature=0.5,
            )
            summary, _ = self._parse_exec_response(raw)
            return summary
        except Exception as e:
            log.warning("Executive summary generation failed: %s", e)
            return self._fallback_exec_summary(
                summarized_items, date.today().isoformat()
            )

    def _build_exec_prompt(self, summarized_items: list["SummarizedItem"]) -> str:
        """Build the executive summary prompt from story list."""
        lines = []
        for i, si in enumerate(summarized_items[:10], 1):  # cap at top 10
            lines.append(
                f"  {i}. [{si.scored_item.ai_score}/10] {si.ai_headline}"
                f" — {si.scored_item.item.source_name}"
            )
        story_list = "\n".join(lines)

        return _EXEC_SUMMARY_PROMPT.format(
            interests=self.settings.user_interests,
            item_count=len(summarized_items),
            story_list=story_list,
        )

    @staticmethod
    def _parse_exec_response(raw: str) -> tuple[str, list[str]]:
        """Parse the executive summary JSON response."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in exec summary response: {text[:100]!r}")

        data = json.loads(match.group())
        summary = str(data.get("executive_summary", "")).strip()
        themes = [str(t).strip() for t in data.get("top_themes", []) if t]

        if not summary:
            raise ValueError("Missing 'executive_summary' in response")

        return summary, themes

    @staticmethod
    def _fallback_exec_summary(
        summarized_items: list["SummarizedItem"], date_str: str
    ) -> str:
        """Template-based fallback when AI exec summary fails."""
        if not summarized_items:
            return f"No significant stories found for {date_str}."

        top = summarized_items[0]
        sources = list({si.scored_item.item.source_name for si in summarized_items})
        sources_str = ", ".join(sources[:3])

        return (
            f"Today's briefing for {date_str} features {len(summarized_items)} "
            f"curated stories from {sources_str} and other sources.\n\n"
            f"Top story: {top.ai_headline}\n\n"
            f"Browse the items below for full summaries and key takeaways."
        )

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_top_topics(
        summarized_items: list["SummarizedItem"],
        top_n: int = 5,
    ) -> list[str]:
        """
        Aggregate AI-extracted topics across all items and return the top N.

        Uses a Counter so the most frequently occurring topics across
        all items bubble up to the top. This gives a quick "themes of
        the day" snapshot.
        """
        counter: Counter[str] = Counter()
        for si in summarized_items:
            for topic in si.scored_item.ai_topics:
                if topic:
                    counter[topic.lower().strip()] += 1

        # Return topics sorted by frequency, then alphabetically for ties
        return [topic for topic, _ in counter.most_common(top_n)]
