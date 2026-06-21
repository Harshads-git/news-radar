"""
src/renderers/github_pages.py
==============================
Writes rendered briefings to the docs/ directory for GitHub Pages hosting.

GitHub Pages setup:
  - Source: `docs/` folder on the `main` branch
  - Each daily briefing gets its own HTML file: docs/YYYY-MM-DD.html
  - The latest briefing is also written to docs/index.html (the landing page)
  - An archive index (docs/archive.html) lists all past briefings

Directory layout:
    docs/
      index.html           ← latest briefing (auto-updated each run)
      archive.html         ← index of all past briefings
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
            {"daily_html", "daily_md", "index_html", "archive_html"}
        """
        log.section("Phase 5: GitHub Pages Output")

        outputs: dict[str, Path] = {}

        # 1. Per-date HTML
        html_path = self._write_daily_html(briefing)
        outputs["daily_html"] = html_path

        # 2. Per-date Markdown
        md_path = self._write_daily_md(briefing)
        outputs["daily_md"] = md_path

        # 3. index.html (always latest)
        index_path = self._write_index(briefing)
        outputs["index_html"] = index_path

        # 4. archive.html (list of all briefings)
        archive_path = self._write_archive()
        outputs["archive_html"] = archive_path

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
        """Write index.html pointing to (or containing) the latest briefing."""
        html_str = render_html(briefing)
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
    <p><a href="index.html" style="color:#6c8fff">&#8592; Latest Briefing</a></p>
  </footer>
</div>
</body>
</html>"""
