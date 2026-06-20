"""
src/models.py
=============
Core Pydantic v2 data models for the News Radar pipeline.

Data flows through the pipeline in this order:
  SourceConfig  ← loaded from data/sources.json
      ↓
  NewsItem      ← output of every scraper
      ↓
  ScoredItem    ← NewsItem + AI score + reason
      ↓
  SummarizedItem← ScoredItem + AI summary + context
      ↓
  Briefing      ← daily collection of SummarizedItems

Keeping models in one file makes the data contract explicit and easy
to audit. Anyone reading this file understands exactly what flows
through the entire system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Source Configuration Models  (loaded from data/sources.json)
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """
    Represents a single configured news source from sources.json.

    Examples
    --------
    RSS source:
        {"id": "hn-rss", "type": "rss", "name": "HN", "url": "https://..."}

    Hacker News API:
        {"id": "hn-api", "type": "hackernews", "name": "HN Top"}

    Reddit:
        {"id": "r-prog", "type": "reddit", "subreddit": "programming"}
    """

    id: str = Field(description="Unique identifier for this source.")
    type: str = Field(description="Source type: 'rss', 'hackernews', or 'reddit'.")
    name: str = Field(description="Human-readable display name.")
    enabled: bool = Field(default=True, description="If False, scraper skips this source.")
    limit: int = Field(default=30, ge=1, le=200, description="Max items to fetch.")
    tags: list[str] = Field(default_factory=list, description="Topic tags for this source.")

    # RSS-specific
    url: str | None = Field(default=None, description="Feed URL (RSS/Atom sources only).")

    # Reddit-specific
    subreddit: str | None = Field(default=None, description="Subreddit name without r/.")
    sort: str = Field(default="hot", description="Reddit sort: hot, new, top, rising.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"rss", "hackernews", "reddit", "github"}
        if v not in allowed:
            raise ValueError(f"Source type '{v}' not supported. Choose from: {allowed}")
        return v

    @model_validator(mode="after")
    def validate_required_fields(self) -> SourceConfig:
        if self.type == "rss" and not self.url:
            raise ValueError(f"RSS source '{self.id}' requires a 'url' field.")
        if self.type == "reddit" and not self.subreddit:
            raise ValueError(f"Reddit source '{self.id}' requires a 'subreddit' field.")
        return self


class SourcesConfig(BaseModel):
    """Root wrapper for the entire sources.json file."""

    sources: list[SourceConfig] = Field(default_factory=list)

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        """Return only sources that are enabled."""
        return [s for s in self.sources if s.enabled]


# ---------------------------------------------------------------------------
# NewsItem  (raw output from any scraper)
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """
    A single news story fetched from any source.

    Created by scrapers; consumed by the deduplication and scoring stages.
    All fields are optional except title and url, since different sources
    provide different metadata.
    """

    # Identity
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID, auto-generated if not provided by the source.",
    )
    url: str = Field(description="Canonical URL for the story.")
    title: str = Field(description="Story headline or title.")

    # Content
    summary: str | None = Field(
        default=None,
        description="Original summary/description from the source (not AI-generated).",
    )
    author: str | None = Field(default=None, description="Author or username.")

    # Source metadata
    source_id: str = Field(description="ID of the SourceConfig that produced this item.")
    source_name: str = Field(description="Human-readable source name.")
    source_type: str = Field(description="Source type: rss, hackernews, reddit, etc.")

    # Engagement signals (from the platform, not AI)
    score: int | None = Field(
        default=None,
        description="Platform score (HN points, Reddit upvotes, etc.).",
    )
    comment_count: int | None = Field(
        default=None,
        description="Number of comments on the platform.",
    )
    comments_url: str | None = Field(
        default=None,
        description="URL to the discussion thread (e.g. HN comments page).",
    )

    # Timing
    published_at: datetime | None = Field(
        default=None,
        description="Publication timestamp from the source.",
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this item was fetched by our scraper.",
    )

    # Extras
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags inherited from the source config.",
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Original raw data from the source API (for debugging).",
        exclude=True,  # not serialized to JSON output
    )

    @field_validator("url", mode="before")
    @classmethod
    def clean_url(cls, v: str) -> str:
        """Strip whitespace and trailing slashes for consistent dedup."""
        return str(v).strip().rstrip("/")

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Collapse multiple spaces and strip leading/trailing whitespace."""
        import re

        return re.sub(r"\s+", " ", str(v).strip())

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NewsItem):
            return NotImplemented
        return self.url == other.url


# ---------------------------------------------------------------------------
# ScoredItem  (NewsItem + AI scoring output)
# ---------------------------------------------------------------------------


class ScoredItem(BaseModel):
    """
    A NewsItem that has been evaluated by the AI scoring engine.

    The ai_score field (0-10) is the primary ranking signal for the briefing.
    Items below settings.score_threshold are dropped before summarization.
    """

    # Original item (embedded, not referenced)
    item: NewsItem = Field(description="The original news item.")

    # AI scoring
    ai_score: int = Field(
        ge=0,
        le=10,
        description="AI relevance/quality score from 0 (ignore) to 10 (must-read).",
    )
    ai_score_reason: str = Field(
        default="",
        description="One-sentence explanation of why this score was assigned.",
    )
    # Alias for convenience (used by scorer and tests)
    ai_reason: str = Field(
        default="",
        description="Alias for ai_score_reason — one-sentence explanation.",
    )
    ai_topics: list[str] = Field(
        default_factory=list,
        description="AI-extracted topic labels for this story (e.g. 'AI', 'Python').",
    )

    # Metadata
    scored_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this item was scored.",
    )
    model_used: str = Field(
        default="",
        description="AI model identifier used for scoring (e.g. 'gpt-4o-mini').",
    )
    from_cache: bool = Field(
        default=False,
        description="True if this score was loaded from cache instead of the AI.",
    )

    # Convenience pass-throughs
    @property
    def title(self) -> str:
        return self.item.title

    @property
    def url(self) -> str:
        return self.item.url

    @property
    def source_name(self) -> str:
        return self.item.source_name


# ---------------------------------------------------------------------------
# SummarizedItem  (ScoredItem + AI summary + web context)
# ---------------------------------------------------------------------------


class SummarizedItem(BaseModel):
    """
    A ScoredItem enriched with an AI-generated summary and background context.

    This is the final per-story unit that appears in the Briefing.
    """

    scored: ScoredItem = Field(description="The scored item this summary is based on.")

    # AI enrichment
    ai_summary: str = Field(
        default="",
        description="AI-generated 3-paragraph summary of the story.",
    )
    ai_headline: str = Field(
        default="",
        description="AI-generated engaging headline (may differ from original title).",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="AI-extracted bullet-point key takeaways.",
    )
    web_context: str = Field(
        default="",
        description="Background context fetched from the web to aid summarization.",
    )

    # Metadata
    summarized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    model_used: str = Field(default="")
    from_cache: bool = Field(default=False)

    # Convenience pass-throughs
    @property
    def title(self) -> str:
        return self.scored.item.title

    @property
    def url(self) -> str:
        return self.scored.item.url

    @property
    def ai_score(self) -> int:
        return self.scored.ai_score

    @property
    def source_name(self) -> str:
        return self.scored.item.source_name


# ---------------------------------------------------------------------------
# Briefing  (daily collection — top-level output of the pipeline)
# ---------------------------------------------------------------------------


class BriefingMetadata(BaseModel):
    """Runtime statistics attached to every Briefing for observability."""

    total_fetched: int = Field(default=0, description="Total items fetched across all sources.")
    total_after_dedup: int = Field(default=0, description="Items remaining after deduplication.")
    total_scored: int = Field(default=0, description="Items that were sent to the AI scorer.")
    total_in_briefing: int = Field(default=0, description="Items that made the score threshold.")
    sources_used: list[str] = Field(default_factory=list, description="Source IDs that ran.")
    run_duration_seconds: float = Field(default=0.0, description="Total pipeline run time.")


class Briefing(BaseModel):
    """
    The final daily output of the News Radar pipeline.

    Serialized to:
      - data/briefings/YYYY-MM-DD.json  (persistent store)
      - docs/YYYY-MM-DD.md              (GitHub Pages)
      - Email / webhook payload

    Items are always sorted by ai_score descending (rank #1 = highest score).
    """

    date: str = Field(
        description="Briefing date in ISO format YYYY-MM-DD.",
        examples=["2026-06-13"],
    )
    language: str = Field(
        default="English",
        description="Output language used for AI-generated content.",
    )
    items: list[SummarizedItem] = Field(
        default_factory=list,
        description="Ranked list of stories in this briefing.",
    )
    executive_summary: str = Field(
        default="",
        description="AI-written 2-3 paragraph overview of the day's news.",
    )
    top_topics: list[str] = Field(
        default_factory=list,
        description="Most frequent AI topics across all items.",
    )
    # Convenience flat stats (mirrors BriefingMetadata for quick access)
    total_fetched: int = Field(default=0, description="Total items fetched across all sources.")
    total_scored: int = Field(default=0, description="Items that passed the score threshold.")
    metadata: BriefingMetadata = Field(
        default_factory=BriefingMetadata,
        description="Detailed pipeline run statistics.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this briefing was generated.",
    )

    @model_validator(mode="after")
    def sort_by_score(self) -> Briefing:
        """Always keep items sorted by AI score descending."""
        self.items = sorted(self.items, key=lambda x: x.ai_score, reverse=True)
        return self

    @property
    def top_items(self) -> list[SummarizedItem]:
        """Shortcut: returns the top 5 items (for webhook digests)."""
        return self.items[:5]

    @property
    def item_count(self) -> int:
        return len(self.items)
