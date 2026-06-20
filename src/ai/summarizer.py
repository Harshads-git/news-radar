"""
src/ai/summarizer.py
====================
AI-powered news summarizer — converts ScoredItems into SummarizedItems.

The summarizer adds two things on top of scoring:
  1. A 3-paragraph AI-written summary:
       Para 1 — What happened (facts)
       Para 2 — Why it matters (context and implications)
       Para 3 — What to watch next (forward-looking insight)
  2. An AI-generated one-sentence headline (more engaging than the original)

Design:
  - Concurrency: asyncio.gather() with semaphore (same pattern as scorer)
  - The summarizer reuses the same BaseAIProvider so the caller only
    needs to initialize one provider for the whole pipeline.
  - Falls back to the original title + summary on AI failure
    (never blocks the pipeline).

Usage:
    from src.ai.summarizer import NewsSummarizer
    from src.ai import AIProviderFactory

    provider = AIProviderFactory.from_model("gpt-4o-mini")
    summarizer = NewsSummarizer(provider, settings)
    summarized_items = await summarizer.summarize_all(scored_items)
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.ai.base import BaseAIProvider
    from src.config import Settings
    from src.models import ScoredItem, SummarizedItem

log = get_logger(__name__)

# Maximum concurrent summarization calls
_DEFAULT_CONCURRENCY = 3  # Lower than scorer — summaries are longer prompts

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT_TEMPLATE = """\
You are an expert tech journalist writing a daily news briefing.
Write a concise, insightful summary of the following news story for a reader
interested in: {interests}

Story:
  Title: {title}
  Source: {source} (score: {platform_score}, comments: {comments})
  Original summary: {original_summary}
  Web context: {web_context}
  AI relevance score: {ai_score}/10
  Key topics: {topics}

Respond ONLY with a JSON object in exactly this format (no markdown, no preamble):
{{
  "headline": "<an engaging, specific one-sentence headline (max 15 words)>",
  "summary": "<paragraph 1: what happened — 2-3 sentences of facts>\\n\\n<paragraph 2: why it matters — 2-3 sentences of context and implications>\\n\\n<paragraph 3: what to watch — 1-2 sentences forward-looking insight>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"]
}}

Guidelines:
- Be factual and specific. No vague phrases like "this is interesting" or "experts say".
- Paragraph 1: State what actually happened. Use numbers and names.
- Paragraph 2: Explain the broader significance. Who is affected? What changes?
- Paragraph 3: What should the reader watch for next?
- key_points: 3 bullet-point takeaways (short, actionable sentences).
"""


class NewsSummarizer:
    """
    Batch AI summarizer for a list of ScoredItems.

    Takes ScoredItems (already filtered by the scorer) and generates
    rich summaries for each, returning SummarizedItems.
    """

    def __init__(
        self,
        provider: "BaseAIProvider",
        settings: "Settings",
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    async def summarize_all(
        self,
        scored_items: list["ScoredItem"],
        web_contexts: dict[str, str] | None = None,
    ) -> list["SummarizedItem"]:
        """
        Summarize all scored items concurrently.

        Parameters
        ----------
        scored_items:
            Items that passed the score threshold (from NewsScorer).
        web_contexts:
            Optional dict mapping item URL → web context string.
            If None, no web context is injected.

        Returns
        -------
        list[SummarizedItem]
            One SummarizedItem per ScoredItem, in the same order.
            Items that fail summarization get a fallback summary.
        """
        if not scored_items:
            return []

        log.section("Phase 3: AI Summarization")
        log.info("Summarizing %d scored items", len(scored_items))

        contexts = web_contexts or {}

        results = await asyncio.gather(
            *[
                self._summarize_one(item, contexts.get(item.item.url, ""))
                for item in scored_items
            ]
        )

        summarized = list(results)
        log.success("Summarized %d items", len(summarized))
        return summarized

    async def summarize_single(
        self,
        scored_item: "ScoredItem",
        web_context: str = "",
    ) -> "SummarizedItem":
        """
        Summarize a single ScoredItem (useful for testing and debugging).
        """
        return await self._summarize_one(scored_item, web_context)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _summarize_one(
        self,
        scored_item: "ScoredItem",
        web_context: str,
    ) -> "SummarizedItem":
        """Summarize one item under the concurrency semaphore."""
        async with self._semaphore:
            log.debug("Summarizing: %s", scored_item.item.title[:60])
            try:
                prompt = self._build_prompt(scored_item, web_context)
                raw = await self.provider.complete(
                    prompt,
                    max_tokens=600,
                    temperature=0.4,
                )
                headline, summary, key_points = self._parse_response(raw)
            except Exception as e:
                log.warning(
                    "Summarization failed for '%s': %s",
                    scored_item.item.title[:50],
                    e,
                )
                headline, summary, key_points = self._make_fallback(scored_item)

            return self._build_summarized_item(scored_item, headline, summary, key_points)

    def _build_prompt(self, scored_item: "ScoredItem", web_context: str) -> str:
        """Construct the summarization prompt from a scored item."""
        item = scored_item.item
        platform_score = str(item.score) if item.score is not None else "n/a"
        comments = str(item.comment_count) if item.comment_count is not None else "n/a"
        original_summary = (item.summary or "")[:400] or "Not available"
        context = web_context[:400] if web_context else "Not available"
        topics = ", ".join(scored_item.ai_topics) if scored_item.ai_topics else "general tech"

        return _SUMMARIZE_PROMPT_TEMPLATE.format(
            interests=self.settings.user_interests,
            title=item.title,
            source=item.source_name,
            platform_score=platform_score,
            comments=comments,
            original_summary=original_summary,
            web_context=context,
            ai_score=scored_item.ai_score,
            topics=topics,
        )

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str, list[str]]:
        """
        Parse the AI's JSON summarization response.

        Returns (headline, summary, key_points).
        Raises ValueError on parse failure (caller handles fallback).
        """
        # Strip markdown fences
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in response: {text[:100]!r}")

        data = json.loads(match.group())

        headline = str(data.get("headline", "")).strip()
        summary = str(data.get("summary", "")).strip()
        key_points = [str(p).strip() for p in data.get("key_points", []) if p]

        if not headline:
            raise ValueError("Missing 'headline' in response")
        if not summary:
            raise ValueError("Missing 'summary' in response")

        return headline, summary, key_points

    @staticmethod
    def _make_fallback(scored_item: "ScoredItem") -> tuple[str, str, list[str]]:
        """Generate a fallback summary when AI summarization fails."""
        item = scored_item.item
        headline = item.title
        summary = item.summary or f"Story from {item.source_name}. Read the original for details."
        key_points = [f"Source: {item.source_name}", f"Score: {scored_item.ai_score}/10"]
        return headline, summary, key_points

    @staticmethod
    def _build_summarized_item(
        scored_item: "ScoredItem",
        headline: str,
        summary: str,
        key_points: list[str],
    ) -> "SummarizedItem":
        """Assemble the final SummarizedItem from parsed AI output."""
        from src.models import SummarizedItem

        return SummarizedItem(
            scored=scored_item,
            ai_headline=headline,
            ai_summary=summary,
            key_points=key_points,
            model_used=scored_item.model_used,
        )
