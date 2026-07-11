"""
src/briefing.py
===============
BriefingBuilder — assembles the final daily Briefing from SummarizedItems.

The BriefingBuilder is the last stage before delivery/storage. It:
  1. Groups summarized items by topic cluster        [upgraded Day 22]
  2. Generates briefing-level metadata (date, item count, top topics)
  3. Generates an AI-written executive summary enriched with cluster context
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

Day 22 additions:
  - TopicCluster dataclass: a named group of related SummarizedItems
  - cluster_items(): groups items by their dominant AI topic using
    topic frequency, returning ordered TopicCluster list
  - Exec summary prompt now includes cluster context for better AI output
  - Improved fallback summary shows per-cluster story counts

Usage:
    from src.briefing import BriefingBuilder
    builder = BriefingBuilder(provider, settings)
    briefing = await builder.build(summarized_items, stats)

    # Access clusters independently:
    clusters = BriefingBuilder.cluster_items(summarized_items, top_n=5)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.ai.base import BaseAIProvider
    from src.config import Settings
    from src.models import Briefing, SummarizedItem

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TopicCluster — a named group of related SummarizedItems
# ---------------------------------------------------------------------------


@dataclass
class TopicCluster:
    """
    A named group of SummarizedItems sharing a dominant topic.

    Items can belong to more than one cluster if they have multiple topics,
    but each item appears in at most one cluster (the first matched).

    Why cluster rather than just listing top topics?
    Clustering creates a narrative structure for the exec summary:
    instead of "10 stories about AI, Python, and Rust", we can say
    "3 AI stories led by OpenAI's GPT-5 announcement; 2 Python stories
    focused on the new release; 1 Rust story about performance gains."
    """

    name: str
    items: list["SummarizedItem"] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def top_item(self) -> "SummarizedItem | None":
        """Return the highest-scored item in this cluster."""
        return self.items[0] if self.items else None

    def short_description(self) -> str:
        """One-line description for prompt/display use."""
        if not self.items:
            return f"{self.name}: (no items)"
        top = self.top_item
        headline = (top.ai_headline or top.title) if top else ""
        return f"{self.name} ({self.size} stor{'y' if self.size == 1 else 'ies'}): {headline}"


# ---------------------------------------------------------------------------
# Prompt template for the executive summary
# ---------------------------------------------------------------------------

_EXEC_SUMMARY_PROMPT = """\
You are an editor writing the opening section of a daily AI/tech news briefing.
The reader is interested in: {interests}

Today's {item_count} curated stories are grouped into {cluster_count} topic cluster(s):
{cluster_lines}

Top stories (by relevance score):
{story_list}

Write a concise executive summary in this JSON format (no markdown, no preamble):
{{
  "executive_summary": "<2-3 paragraphs: paragraph 1 = today's dominant theme with specific story, paragraph 2 = other notable clusters/developments, paragraph 3 = what to watch tomorrow>",
  "top_themes": ["<theme1>", "<theme2>", "<theme3>"]
}}

Guidelines:
- Para 1: Name the dominant cluster and its top story specifically.
- Para 2: Mention other active clusters with a story example each.
- Para 3: Forward-looking: what should the reader follow tomorrow?
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

        # Cluster items by dominant topic
        clusters = self.cluster_items(summarized_items, top_n=5)
        log.info(
            "Topic clusters: %s",
            ", ".join(f"{c.name}({c.size})" for c in clusters) or "none",
        )

        # Generate executive summary
        if summarized_items and generate_exec_summary:
            exec_summary = await self._generate_exec_summary(summarized_items, clusters)
        else:
            exec_summary = self._fallback_exec_summary(summarized_items, date_str, clusters)

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
            "Briefing ready: %d items, %d clusters, %d top topics",
            len(summarized_items),
            len(clusters),
            len(top_topics),
        )
        return briefing

    # ------------------------------------------------------------------
    # Topic Clustering (Day 22)
    # ------------------------------------------------------------------

    @staticmethod
    def cluster_items(
        summarized_items: list["SummarizedItem"],
        top_n: int = 5,
    ) -> list["TopicCluster"]:
        """
        Group summarized items by their dominant topic.

        Algorithm:
          1. Count topic frequency across all items to find the top N topics
          2. For each top topic (in frequency order), collect items whose
             ai_topics list contains that topic
          3. Each item is assigned to at most one cluster (first match wins)
             to avoid double-counting

        Why this approach vs. TF-IDF clustering?
        We already have AI-extracted topics (ai_topics) for each item from
        the scoring stage. Using these is faster (O(n)) and more semantically
        accurate than re-clustering titles with TF-IDF. The AI already did
        the hard work of labelling each story.

        Parameters
        ----------
        summarized_items:
            List of SummarizedItems, ordered by score descending.
        top_n:
            Maximum number of clusters to create. Default: 5.

        Returns
        -------
        list[TopicCluster]
            Clusters ordered by size (largest first). Items not matched
            by any top topic are collected into a final "Other" cluster
            only if there are any unclustered items.
        """
        if not summarized_items:
            return []

        # Count topic frequency (case-insensitive, normalised)
        counter: Counter[str] = Counter()
        for si in summarized_items:
            for topic in (si.scored.ai_topics or []):
                if topic:
                    counter[topic.strip().lower()] += 1

        # Take the top N topics as cluster labels (title-case for display)
        top_topics = [t for t, _ in counter.most_common(top_n)]

        # Build topic → canonical display label mapping
        # e.g. "ai" → "AI", "python" → "Python"
        def display_label(t: str) -> str:
            label_map = {
                "ai": "AI",
                "llm": "LLM",
                "api": "API",
                "ml": "ML",
                "oss": "OSS",
            }
            return label_map.get(t.lower(), t.title())

        # Assign items to clusters (greedy: first matching top topic wins)
        assigned: set[int] = set()  # ids of assigned items
        clusters: list[TopicCluster] = []

        for topic in top_topics:
            cluster_items_list = []
            for si in summarized_items:
                if id(si) in assigned:
                    continue
                item_topics = [t.strip().lower() for t in (si.scored.ai_topics or [])]
                if topic in item_topics:
                    cluster_items_list.append(si)
                    assigned.add(id(si))
            if cluster_items_list:
                clusters.append(TopicCluster(
                    name=display_label(topic),
                    items=cluster_items_list,
                ))

        # Collect remaining unclustered items into "Other"
        unclustered = [si for si in summarized_items if id(si) not in assigned]
        if unclustered:
            clusters.append(TopicCluster(name="Other", items=unclustered))

        # Sort by cluster size (largest first), Other always last
        main_clusters = [c for c in clusters if c.name != "Other"]
        other_cluster = [c for c in clusters if c.name == "Other"]
        main_clusters.sort(key=lambda c: c.size, reverse=True)

        return main_clusters + other_cluster

    # ------------------------------------------------------------------
    # Executive summary generation
    # ------------------------------------------------------------------

    async def _generate_exec_summary(
        self,
        summarized_items: list["SummarizedItem"],
        clusters: list["TopicCluster"],
    ) -> str:
        """Use AI to generate a briefing-level executive summary."""
        prompt = self._build_exec_prompt(summarized_items, clusters)

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
                summarized_items, date.today().isoformat(), clusters
            )

    def _build_exec_prompt(
        self,
        summarized_items: list["SummarizedItem"],
        clusters: list["TopicCluster"],
    ) -> str:
        """Build the executive summary prompt with cluster context."""
        # Story list (top 10)
        lines = []
        for i, si in enumerate(summarized_items[:10], 1):
            lines.append(
                f"  {i}. [{si.scored.ai_score}/10] {si.ai_headline or si.title}"
                f" -- {si.scored.item.source_name}"
            )
        story_list = "\n".join(lines)

        # Cluster summary lines
        cluster_lines = "\n".join(
            f"  - {c.short_description()}" for c in clusters
        ) or "  (no topic clusters)"

        return _EXEC_SUMMARY_PROMPT.format(
            interests=self.settings.user_interests,
            item_count=len(summarized_items),
            cluster_count=len(clusters),
            cluster_lines=cluster_lines,
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
        summarized_items: list["SummarizedItem"],
        date_str: str,
        clusters: list["TopicCluster"] | None = None,
    ) -> str:
        """
        Template-based fallback when AI exec summary fails or is disabled.

        Day 22 improvement: includes per-cluster story counts for richer context.
        """
        if not summarized_items:
            return f"No significant stories found for {date_str}."

        top = summarized_items[0]
        headline = top.ai_headline or top.title
        sources = list({si.scored.item.source_name for si in summarized_items})
        sources_str = ", ".join(sources[:3])

        intro = (
            f"Today's briefing for {date_str} features {len(summarized_items)} "
            f"curated stories from {sources_str} and other sources."
        )

        # Add cluster breakdown if available
        cluster_str = ""
        if clusters:
            cluster_parts = [
                f"{c.name} ({c.size})"
                for c in clusters
                if c.size > 0 and c.name != "Other"
            ]
            if cluster_parts:
                cluster_str = "\n\nToday's themes: " + ", ".join(cluster_parts) + "."

        return (
            f"{intro}{cluster_str}\n\n"
            f"Top story: {headline}\n\n"
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
            for topic in si.scored.ai_topics:
                if topic:
                    counter[topic.lower().strip()] += 1

        # Return topics sorted by frequency, then alphabetically for ties
        return [topic for topic, _ in counter.most_common(top_n)]
