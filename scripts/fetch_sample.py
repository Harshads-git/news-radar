"""
scripts/fetch_sample.py
=======================
One-shot script to fetch live RSS feeds and save a sample JSON snapshot
to data/sample_rss.json. Used to verify the scraper against real feeds
and to capture representative test fixtures.

Run with:
    uv run python scripts/fetch_sample.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


async def main() -> None:
    from src.logger import configure_logging, get_logger
    from src.models import SourceConfig
    from src.scrapers.rss import RssScraper

    configure_logging("INFO")
    log = get_logger("scripts.fetch_sample")

    sources = [
        SourceConfig(
            id="hn-rss",
            type="rss",
            name="Hacker News Front Page",
            url="https://hnrss.org/frontpage",
            limit=10,
            tags=["tech", "programming"],
        ),
        SourceConfig(
            id="python-blog",
            type="rss",
            name="Python Insider Blog",
            url="https://feeds.feedburner.com/PythonInsider",
            limit=5,
            tags=["python", "releases"],
        ),
    ]

    scraper = RssScraper()
    all_items = []

    for source in sources:
        log.info("Fetching: %s", source.name)
        items = await scraper.fetch_safe(source)
        log.success("Got %d items from %s", len(items), source.name)
        for item in items:
            all_items.append({
                "source": item.source_name,
                "title": item.title,
                "url": item.url,
                "author": item.author,
                "published_at": item.published_at.isoformat() if item.published_at else None,
                "summary_preview": (item.summary or "")[:120],
                "tags": item.tags,
            })

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_items": len(all_items),
        "items": all_items,
    }

    out_path = Path("data/sample_rss.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.success("Saved %d items to %s", len(all_items), out_path)


if __name__ == "__main__":
    asyncio.run(main())
