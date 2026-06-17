"""
src/scrapers/__init__.py
========================
Scraper package exports and ScraperFactory.

The ScraperFactory is the ONLY place that maps SourceConfig.type strings
to concrete scraper classes. This means:
  - Adding a new source type = add one line here + create the scraper file
  - The Orchestrator (Day 16) never imports scrapers directly
  - Tests can verify all registered types work

Usage:
    from src.scrapers import ScraperFactory

    scraper = ScraperFactory.create("rss")
    items = await scraper.fetch(source_config)

    # Or using a SourceConfig directly:
    scraper = ScraperFactory.for_source(source_config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.scrapers.base import BaseScraper
from src.scrapers.hackernews import HackerNewsScraper
from src.scrapers.reddit import RedditScraper
from src.scrapers.rss import RssScraper

if TYPE_CHECKING:
    from src.models import SourceConfig

# ---------------------------------------------------------------------------
# Registry: maps source type strings → scraper classes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseScraper]] = {
    "rss": RssScraper,
    "hackernews": HackerNewsScraper,
    "reddit": RedditScraper,
}


class ScraperFactory:
    """
    Factory for instantiating the correct BaseScraper subclass.

    All scraper instantiation goes through this class — the orchestrator
    and any other caller never imports concrete scraper classes directly.
    This enforces the Open/Closed principle: adding a new source type
    requires no changes to existing code, only registration here.
    """

    @staticmethod
    def create(source_type: str) -> BaseScraper:
        """
        Create a scraper instance for the given source type string.

        Parameters
        ----------
        source_type:
            One of the registered type keys (e.g. "rss", "hackernews", "reddit").

        Returns
        -------
        BaseScraper
            A fresh instance of the appropriate scraper subclass.

        Raises
        ------
        ValueError
            If the source_type is not registered in the factory.

        Examples
        --------
        >>> scraper = ScraperFactory.create("rss")
        >>> isinstance(scraper, RssScraper)
        True
        """
        cls = _REGISTRY.get(source_type)
        if cls is None:
            registered = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(
                f"No scraper registered for source type '{source_type}'. "
                f"Available types: {registered}"
            )
        return cls()

    @staticmethod
    def for_source(source: "SourceConfig") -> BaseScraper:
        """
        Create the appropriate scraper for the given SourceConfig.

        Convenience wrapper around ``create()`` that reads the type
        from the SourceConfig object.

        Parameters
        ----------
        source:
            A validated SourceConfig from sources.json.
        """
        return ScraperFactory.create(source.type)

    @staticmethod
    def registered_types() -> list[str]:
        """Return a sorted list of all registered source type strings."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def is_registered(source_type: str) -> bool:
        """Return True if a scraper is registered for the given type."""
        return source_type in _REGISTRY


# ---------------------------------------------------------------------------
# Convenience re-exports for clean imports
# ---------------------------------------------------------------------------

__all__ = [
    "BaseScraper",
    "RssScraper",
    "HackerNewsScraper",
    "RedditScraper",
    "ScraperFactory",
]
