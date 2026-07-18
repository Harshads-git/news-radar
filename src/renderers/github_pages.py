"""
src/renderers/github_pages.py
==============================
Writes rendered briefings to the docs/ directory for GitHub Pages hosting.

GitHub Pages setup:
  - Source: `docs/` folder on the `main` branch
  - Each daily briefing gets its own HTML file: docs/YYYY-MM-DD.html
  - The latest briefing is also written to docs/index.html (the landing page)
  - An archive index (docs/archive.html) lists all past briefings
  - A topic index (docs/topics.html) lists all detected topics with counts
  - Per-topic pages (docs/topic-{slug}.html) list stories for each topic
  - A search index (docs/search.json) powers the client-side search widget

Directory layout:
    docs/
      index.html           ← latest briefing with search bar + topic nav
      archive.html         ← index of all past briefings
      topics.html          ← [NEW] topic index page (Day 28)
      topic-{slug}.html    ← [NEW] per-topic archive pages (Day 28)
      search.json          ← [NEW] lightweight search index (Day 28)
      YYYY-MM-DD.html      ← each daily briefing (permanent URL)
      YYYY-MM-DD.md        ← raw markdown (for repo browsing)

Why docs/?
  GitHub Pages serves from `docs/` on the same branch as your source,
  so both code and rendered output live in the same repo — no separate
  gh-pages branch needed. Simple, easy to audit, easy to roll back.

Usage:
    from src.renderers.github_pages import GitHubPagesWriter
    writer = GitHubPagesWriter(docs_dir=Path("docs"))
    writer.write(briefing)
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.exceptions import StorageError
from src.logger import get_logger
from src.renderers.html import render_html
from src.renderers.markdown import render_markdown

if TYPE_CHECKING:
    from src.models import Briefing

log = get_logger(__name__)

_DOCS_DIR_NAME = "docs"


class GitHubPagesWriter:
    """
    Writes Briefing objects as HTML/Markdown to the docs/ directory.

    Maintains:
      - Per-date HTML files (permanent URLs)
      - Per-date Markdown files (for GitHub browsing)
      - index.html (always the latest briefing)
      - archive.html (index of all stored briefings)
    """

    def __init__(self, docs_dir: Path | str) -> None:
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        log.debug("GitHubPagesWriter initialized at %s", self.docs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, briefing: "Briefing") -> dict[str, Path]:
        """
        Write a briefing to all output files.

        Parameters
        ----------
        briefing:
            The Briefing to write.

        Returns
        -------
        dict[str, Path]
            Mapping of output name → absolute Path for each file written:
            {"daily_html", "daily_md", "index_html", "archive_html",
             "topics_html", "search_json"} + one entry per topic page.
        """
        log.section("Phase 5: GitHub Pages Output")

        outputs: dict[str, Path] = {}

        # 1. Per-date HTML
        html_path = self._write_daily_html(briefing)
        outputs["daily_html"] = html_path

        # 2. Per-date Markdown
        md_path = self._write_daily_md(briefing)
        outputs["daily_md"] = md_path

        # 3. index.html (always latest, now includes search + topic nav)
        index_path = self._write_index(briefing)
        outputs["index_html"] = index_path

        # 4. archive.html (list of all briefings)
        archive_path = self._write_archive()
        outputs["archive_html"] = archive_path

        # 5. [Day 28] Search index JSON
        search_path = self._write_search_index(briefing)
        outputs["search_json"] = search_path

        # 6. [Day 28] Topics index + per-topic pages
        topic_pages = self._write_topic_pages(briefing)
        outputs.update(topic_pages)
        if topic_pages:
            topics_index = self._write_topics_index()
            outputs["topics_html"] = topics_index

        log.success(
            "GitHub Pages output written: %d files",
            len(outputs),
        )
        for name, path in outputs.items():
            log.debug("  %s -> %s", name, path.name)

        return outputs

    def list_briefing_dates(self) -> list[str]:
        """
        Return sorted list of all briefing dates with HTML output.

        Returns date strings (YYYY-MM-DD), oldest first.
        """
        dates = []
        for f in self.docs_dir.glob("????-??-??.html"):
            dates.append(f.stem)
        return sorted(dates)

    def collect_all_topics(self) -> dict[str, list[dict]]:
        """
        Scan all per-date JSON briefings in docs/ for topic information.

        Returns a dict mapping topic_slug → list of story dicts:
          {"title", "url", "date", "ai_score", "source_name"}

        Only topics with at least 2 appearances are returned.
        Loads story info from the existing .html filenames and a companion
        search.json if present (avoids re-parsing full briefing JSON).
        """
        # Load search.json if it exists (fastest route)
        search_file = self.docs_dir / "search.json"
        if not search_file.exists():
            return {}

        try:
            entries: list[dict] = json.loads(search_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        topic_map: dict[str, list[dict]] = {}
        for entry in entries:
            for topic in entry.get("topics", []):
                slug = _topic_slug(topic)
                if slug not in topic_map:
                    topic_map[slug] = []
                topic_map[slug].append({
                    "title": entry.get("title", ""),
                    "url": entry.get("url", ""),
                    "date": entry.get("date", ""),
                    "ai_score": entry.get("ai_score", 0),
                    "source_name": entry.get("source_name", ""),
                    "topic_label": topic,
                })

        # Keep only topics with ≥2 stories
        return {slug: stories for slug, stories in topic_map.items() if len(stories) >= 2}

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write_daily_html(self, briefing: "Briefing") -> Path:
        """Write YYYY-MM-DD.html for this briefing."""
        html_str = render_html(briefing)
        path = self.docs_dir / f"{briefing.date}.html"
        self._write_file(path, html_str)
        return path

    def _write_daily_md(self, briefing: "Briefing") -> Path:
        """Write YYYY-MM-DD.md for this briefing."""
        md_str = render_markdown(briefing)
        path = self.docs_dir / f"{briefing.date}.md"
        self._write_file(path, md_str)
        return path

    def _write_index(self, briefing: "Briefing") -> Path:
        """Write index.html with the latest briefing + search bar + topic nav."""
        html_str = render_html(briefing)
        # Inject search bar + topic nav snippet right after <body>
        snippet = _build_search_nav_snippet(briefing.top_topics)
        html_str = html_str.replace("<body>", "<body>" + snippet, 1)
        path = self.docs_dir / "index.html"
        self._write_file(path, html_str)
        return path

    def _write_archive(self) -> Path:
        """Write archive.html listing all past briefings by date."""
        dates = self.list_briefing_dates()
        html_str = _build_archive_html(dates)
        path = self.docs_dir / "archive.html"
        self._write_file(path, html_str)
        return path

    def _write_search_index(self, briefing: "Briefing") -> Path:
        """
        Write / update docs/search.json with entries from this briefing.

        search.json is a flat array of story objects:
          [{"date", "title", "url", "source_name", "ai_score",
            "topics", "summary", "headline"}]

        Existing entries (from previous runs) are preserved. Today's date
        entries are replaced (idempotent re-run).
        """
        search_file = self.docs_dir / "search.json"
        existing: list[dict] = []
        if search_file.exists():
            try:
                existing = json.loads(search_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = []

        # Remove any entries from today (will be replaced)
        today = briefing.date
        existing = [e for e in existing if e.get("date") != today]

        # Add this briefing's items
        new_entries = []
        for item in briefing.items:
            new_entries.append({
                "date": today,
                "title": item.title,
                "url": item.url,
                "source_name": item.source_name,
                "ai_score": item.ai_score,
                "topics": list(item.scored.ai_topics or []),
                "summary": (item.ai_summary or "")[:200],
                "headline": item.ai_headline or item.title,
            })

        all_entries = existing + new_entries
        # Keep last 90 days → cap at 90 * 30 items = 2700 max
        all_entries = all_entries[-2700:]

        path = search_file
        self._write_file(
            path,
            json.dumps(all_entries, ensure_ascii=False, indent=None),
        )
        return path

    def _write_topic_pages(self, briefing: "Briefing") -> dict[str, Path]:
        """
        Write docs/topic-{slug}.html for each topic in the current briefing.

        Returns a dict mapping "topic_{slug}" → Path.
        """
        outputs: dict[str, Path] = {}
        # Collect topics from this briefing
        topic_items: dict[str, list[dict]] = {}
        for item in briefing.items:
            for topic in (item.scored.ai_topics or []):
                slug = _topic_slug(topic)
                if not slug:
                    continue
                if slug not in topic_items:
                    topic_items[slug] = []
                topic_items[slug].append({
                    "title": item.title,
                    "url": item.url,
                    "date": briefing.date,
                    "ai_score": item.ai_score,
                    "source_name": item.source_name,
                    "topic_label": topic,
                })

        for slug, stories in topic_items.items():
            label = stories[0]["topic_label"]
            # Merge with existing search.json history for this topic
            all_topic_stories = self._get_historical_topic_stories(slug)
            # Replace/merge today's stories
            today = briefing.date
            all_topic_stories = [s for s in all_topic_stories if s["date"] != today]
            all_topic_stories.extend(stories)
            # Sort newest first
            all_topic_stories.sort(key=lambda s: s["date"], reverse=True)

            page_html = _build_topic_page_html(label, slug, all_topic_stories)
            path = self.docs_dir / f"topic-{slug}.html"
            self._write_file(path, page_html)
            outputs[f"topic_{slug}"] = path

        return outputs

    def _write_topics_index(self) -> Path:
        """Write docs/topics.html — a grid of all known topics with story counts."""
        all_topics = self.collect_all_topics()
        html_str = _build_topics_index_html(all_topics)
        path = self.docs_dir / "topics.html"
        self._write_file(path, html_str)
        return path

    def _get_historical_topic_stories(self, slug: str) -> list[dict]:
        """Load historical stories for a topic from search.json."""
        search_file = self.docs_dir / "search.json"
        if not search_file.exists():
            return []
        try:
            entries = json.loads(search_file.read_text(encoding="utf-8"))
            result = []
            for entry in entries:
                topics = entry.get("topics", [])
                if any(_topic_slug(t) == slug for t in topics):
                    topic_label = next(
                        (t for t in topics if _topic_slug(t) == slug), slug
                    )
                    result.append({
                        "title": entry.get("title", ""),
                        "url": entry.get("url", ""),
                        "date": entry.get("date", ""),
                        "ai_score": entry.get("ai_score", 0),
                        "source_name": entry.get("source_name", ""),
                        "topic_label": topic_label,
                    })
            return result
        except (OSError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Atomic file write
    # ------------------------------------------------------------------

    def _write_file(self, path: Path, content: str) -> None:
        """Atomically write content to path (tmp → rename)."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise StorageError(
                f"Failed to write {path.name}: {e}",
                path=str(path),
                operation="write",
            ) from e


# ---------------------------------------------------------------------------
# Archive page builder
# ---------------------------------------------------------------------------


def _build_archive_html(dates: list[str]) -> str:
    """Build a styled archive index page listing all briefings."""
    rows = ""
    for date_str in reversed(dates):  # newest first
        label = date_str
        rows += (
            f'  <li class="archive-item">'
            f'<a href="{html.escape(date_str)}.html">{html.escape(label)}</a>'
            f' &mdash; <a href="{html.escape(date_str)}.md" class="md-link">markdown</a>'
            f"</li>\n"
        )

    count = len(dates)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>News Radar — Archive</title>
  <style>
    :root {{
      --bg: #0f1117; --surface: #1a1d27; --border: #2d3147;
      --text: #e2e4ef; --text-muted: #7b82a4; --accent: #6c8fff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
            font-size: 16px; line-height: 1.7; padding: 2rem 1rem; }}
    .container {{ max-width: 640px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;
          background: linear-gradient(135deg, #6c8fff, #a78bfa);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text; }}
    .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 2rem; }}
    .archive-list {{ list-style: none; }}
    .archive-item {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border);
                     display: flex; gap: 0.5rem; align-items: center; }}
    .archive-item a {{ color: #6c8fff; text-decoration: none; }}
    .archive-item a:hover {{ text-decoration: underline; }}
    .md-link {{ color: var(--text-muted) !important; font-size: 0.8rem; }}
    footer {{ margin-top: 2rem; color: var(--text-muted); font-size: 0.8rem; }}
  </style>
</head>
<body>
<div class="container">
  <h1>News Radar</h1>
  <p class="meta">Archive of {count} daily briefings &mdash; {html.escape(generated)}</p>
  <ul class="archive-list">
{rows}  </ul>
  <footer>
    <p><a href="index.html" style="color:#6c8fff">&#8592; Latest Briefing</a>
       &nbsp;&middot;&nbsp;
       <a href="topics.html" style="color:#6c8fff">Browse Topics</a></p>
  </footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Day 28 helpers: slug, search nav, topic pages, topics index
# ---------------------------------------------------------------------------


def _topic_slug(topic: str) -> str:
    """Convert a topic label to a URL-safe slug.

    Example: "Machine Learning" → "machine-learning"
    """
    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "misc"


def _build_search_nav_snippet(top_topics: list[str]) -> str:
    """
    Build an inline HTML snippet (search bar + topic nav pills) to inject
    into index.html right after the opening <body> tag.

    The search bar uses vanilla JS to filter visible story cards by title,
    source, and topic text. No external libraries required.
    """
    topic_pills = ""
    for topic in (top_topics or [])[:12]:
        slug = _topic_slug(topic)
        label = html.escape(topic)
        topic_pills += (
            f'<a href="topic-{html.escape(slug)}.html" '
            f'class="topic-pill">{label}</a>\n      '
        )

    return f"""
<style>
  .nr-nav {{
    background: #1a1d27;
    border-bottom: 1px solid #2d3147;
    padding: 0.75rem 1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  #nr-search {{
    flex: 1 1 200px;
    background: #0f1117;
    border: 1px solid #2d3147;
    color: #e2e4ef;
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    font-size: 0.9rem;
    outline: none;
    min-width: 180px;
  }}
  #nr-search:focus {{ border-color: #6c8fff; }}
  #nr-search::placeholder {{ color: #7b82a4; }}
  .topic-pill {{
    background: #12163a;
    border: 1px solid #2d3147;
    color: #a78bfa;
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem;
    text-decoration: none;
    white-space: nowrap;
    transition: background 0.15s;
  }}
  .topic-pill:hover {{ background: #1e2250; }}
  .nr-nav-links {{ display: flex; gap: 0.5rem; font-size: 0.82rem; }}
  .nr-nav-links a {{ color: #7b82a4; text-decoration: none; }}
  .nr-nav-links a:hover {{ color: #6c8fff; }}
  .nr-no-results {{ display: none; padding: 2rem; color: #7b82a4; text-align: center; }}
</style>
<nav class="nr-nav" aria-label="Navigation">
  <input id="nr-search" type="search" placeholder="&#x1F50D; Search stories..." autocomplete="off" />
  <div style="display:flex;flex-wrap:wrap;gap:0.4rem;flex:1 1 auto;">
      {topic_pills}</div>
  <div class="nr-nav-links">
    <a href="archive.html">Archive</a>
    <a href="topics.html">Topics</a>
  </div>
</nav>
<p class="nr-no-results" id="nr-no-results">No matching stories found.</p>
<script>
(function() {{
  var inp = document.getElementById('nr-search');
  var noRes = document.getElementById('nr-no-results');
  if (!inp) return;
  inp.addEventListener('input', function() {{
    var q = inp.value.toLowerCase().trim();
    var cards = document.querySelectorAll('article, .story-card, [data-score]');
    if (!cards.length) cards = document.querySelectorAll('section > *');
    var visible = 0;
    cards.forEach(function(card) {{
      var text = card.textContent.toLowerCase();
      var show = !q || text.indexOf(q) !== -1;
      card.style.display = show ? '' : 'none';
      if (show) visible++;
    }});
    if (noRes) noRes.style.display = (!q || visible > 0) ? 'none' : 'block';
  }});
}})();
</script>
"""


def _build_topic_page_html(topic_label: str, slug: str, stories: list[dict]) -> str:
    """Build a per-topic archive page listing all stories with that topic."""
    rows = ""
    for story in stories:
        title = html.escape(str(story.get("title", "")))
        url = html.escape(str(story.get("url", "")))
        date = html.escape(str(story.get("date", "")))
        source = html.escape(str(story.get("source_name", "")))
        score = int(story.get("ai_score", 0))
        hue = int((score - 1) / 9 * 120) if score > 0 else 0
        score_color = f"hsl({hue}, 80%, 45%)"

        rows += f"""  <li class="story-item">
    <div class="story-meta">
      <span class="story-date">{date}</span>
      <span class="story-score" style="color:{score_color}" title="AI score">{score}/10</span>
      <span class="story-source">{source}</span>
    </div>
    <a href="{url}" class="story-title" target="_blank" rel="noopener">{title}</a>
  </li>
"""

    total = len(stories)
    label_esc = html.escape(topic_label)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>News Radar — {label_esc}</title>
  <meta name="description" content="All stories tagged with {label_esc} from News Radar.">
  <style>
    :root {{
      --bg: #0f1117; --surface: #1a1d27; --border: #2d3147;
      --text: #e2e4ef; --text-muted: #7b82a4; --accent: #6c8fff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 16px; line-height: 1.7; padding: 2rem 1rem; }}
    .container {{ max-width: 760px; margin: 0 auto; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem;
          background: linear-gradient(135deg, #6c8fff, #a78bfa);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text; }}
    .topic-badge {{ display: inline-block; background: #12163a;
                    border: 1px solid #2d3147; color: #a78bfa;
                    border-radius: 20px; padding: 0.2rem 0.65rem;
                    font-size: 0.85rem; margin-bottom: 1rem; }}
    .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 2rem; }}
    .story-list {{ list-style: none; }}
    .story-item {{ padding: 0.9rem 0; border-bottom: 1px solid var(--border); }}
    .story-meta {{ display: flex; gap: 0.75rem; font-size: 0.8rem;
                   color: var(--text-muted); margin-bottom: 0.2rem; }}
    .story-score {{ font-weight: 600; }}
    .story-title {{ color: var(--accent); text-decoration: none; font-size: 1rem; }}
    .story-title:hover {{ text-decoration: underline; }}
    footer {{ margin-top: 2rem; color: var(--text-muted); font-size: 0.8rem; }}
    footer a {{ color: var(--accent); text-decoration: none; }}
  </style>
</head>
<body>
<div class="container">
  <h1>News Radar</h1>
  <span class="topic-badge">#{label_esc}</span>
  <p class="meta">{total} stories &mdash; {generated}</p>
  <ul class="story-list">
{rows}  </ul>
  <footer>
    <p>
      <a href="topics.html">&#8592; All Topics</a>
      &nbsp;&middot;&nbsp;
      <a href="index.html">Latest Briefing</a>
    </p>
  </footer>
</div>
</body>
</html>"""


def _build_topics_index_html(all_topics: dict[str, list[dict]]) -> str:
    """Build the docs/topics.html grid page listing all known topics."""
    cards = ""
    # Sort by story count descending
    for slug, stories in sorted(all_topics.items(), key=lambda kv: -len(kv[1])):
        label = stories[0].get("topic_label", slug) if stories else slug
        count = len(stories)
        label_esc = html.escape(str(label))
        slug_esc = html.escape(slug)
        cards += f"""  <a href="topic-{slug_esc}.html" class="topic-card">
    <span class="topic-name">#{label_esc}</span>
    <span class="topic-count">{count} stories</span>
  </a>
"""

    total_topics = len(all_topics)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not cards:
        cards = '  <p style="color:#7b82a4;grid-column:1/-1">No topic data yet. Run the pipeline to start tracking topics.</p>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>News Radar — Topics</title>
  <meta name="description" content="Browse all news topics tracked by News Radar.">
  <style>
    :root {{
      --bg: #0f1117; --surface: #1a1d27; --border: #2d3147;
      --text: #e2e4ef; --text-muted: #7b82a4; --accent: #6c8fff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 16px; line-height: 1.7; padding: 2rem 1rem; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem;
          background: linear-gradient(135deg, #6c8fff, #a78bfa);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text; }}
    .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 2rem; }}
    .topics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 1rem;
    }}
    .topic-card {{
      display: flex; flex-direction: column; gap: 0.2rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.2rem;
      text-decoration: none;
      transition: border-color 0.2s, background 0.2s;
    }}
    .topic-card:hover {{ border-color: #6c8fff; background: #12163a; }}
    .topic-name {{ color: #a78bfa; font-weight: 600; font-size: 0.95rem; }}
    .topic-count {{ color: var(--text-muted); font-size: 0.8rem; }}
    footer {{ margin-top: 2.5rem; color: var(--text-muted); font-size: 0.8rem; }}
    footer a {{ color: var(--accent); text-decoration: none; }}
  </style>
</head>
<body>
<div class="container">
  <h1>News Radar</h1>
  <p class="meta">{total_topics} topics tracked &mdash; {generated}</p>
  <div class="topics-grid">
{cards}  </div>
  <footer>
    <p>
      <a href="index.html">&#8592; Latest Briefing</a>
      &nbsp;&middot;&nbsp;
      <a href="archive.html">Archive</a>
    </p>
  </footer>
</div>
</body>
</html>"""
