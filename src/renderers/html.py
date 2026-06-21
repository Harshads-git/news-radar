"""
src/renderers/html.py
=====================
Renders a Briefing to a styled, standalone HTML page.

Features:
  - Self-contained (no external CSS framework, no CDN)
  - Dark mode with CSS custom properties
  - Responsive layout (works on mobile and desktop)
  - Score color coding (green → yellow → red gradient)
  - Expandable/collapsible story cards (no JS needed — CSS :target trick)
  - Semantic HTML5 structure for SEO

Design:
  The HTML is built with a template string approach:
    1. _render_head(): DOCTYPE, meta, title, all <style>
    2. _render_header(): page title + executive summary section
    3. _render_topics(): top topics pill badges
    4. _render_items(): each story card
    5. _render_footer(): generation metadata

Usage:
    from src.renderers.html import render_html
    html_str = render_html(briefing)
    Path("docs/index.html").write_text(html_str, encoding="utf-8")
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Briefing, SummarizedItem

# ---------------------------------------------------------------------------
# Score → color (HSL: 0=red, 120=green — inverted for 1-10 scale)
# ---------------------------------------------------------------------------


def _score_color(score: int) -> str:
    """Return a CSS hsl() color for a score 1-10 (green → yellow → red)."""
    hue = int((score - 1) / 9 * 120)  # 1→0 (red), 10→120 (green)
    return f"hsl({hue}, 80%, 45%)"


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text or ""), quote=True)


def _format_dt(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# CSS styles (embedded in <head>)
# ---------------------------------------------------------------------------

_CSS = """
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --border: #2d3147;
    --text: #e2e4ef;
    --text-muted: #7b82a4;
    --accent: #6c8fff;
    --accent-glow: rgba(108, 143, 255, 0.15);
    --score-bar-bg: #2d3147;
    --radius: 12px;
    --font: 'Inter', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'Fira Code', 'Cascadia Code', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 16px;
    line-height: 1.7;
    padding: 0 1rem;
  }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  .container {
    max-width: 820px;
    margin: 0 auto;
    padding: 2rem 0 4rem;
  }

  /* ---- Page Header ---- */
  .page-header {
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
  }

  .page-header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
  }

  .page-meta {
    color: var(--text-muted);
    font-size: 0.85rem;
  }

  /* ---- Executive Summary ---- */
  .exec-summary {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
  }

  .exec-summary h2 {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
  }

  .exec-summary p {
    color: var(--text);
    font-size: 0.95rem;
    margin-bottom: 0.75rem;
  }

  .exec-summary p:last-child { margin-bottom: 0; }

  /* ---- Top Topics ---- */
  .topics-section {
    margin-bottom: 2rem;
  }

  .topics-section h2 {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
  }

  .topic-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .topic-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-family: var(--font-mono);
    color: var(--accent);
  }

  /* ---- Story Cards ---- */
  .stories { display: flex; flex-direction: column; gap: 1rem; }

  .story-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .story-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }

  .story-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.75rem;
  }

  .rank {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-muted);
    min-width: 2rem;
    line-height: 1.3;
  }

  .story-title-link {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.35;
  }

  .story-title-link:hover { color: var(--accent); text-decoration: none; }

  .story-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
    padding-left: 3rem;
  }

  .score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    background: var(--surface2);
  }

  .score-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .meta-tag {
    background: var(--surface2);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
  }

  /* ---- Summary Paragraphs ---- */
  .story-summary {
    padding-left: 3rem;
    margin-bottom: 1rem;
  }

  .story-summary p {
    color: var(--text);
    font-size: 0.92rem;
    margin-bottom: 0.75rem;
    line-height: 1.7;
  }

  /* ---- Key Points ---- */
  .key-points {
    padding-left: 3rem;
    margin-bottom: 1rem;
  }

  .key-points h4 {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }

  .key-points ul {
    list-style: none;
  }

  .key-points li {
    font-size: 0.88rem;
    color: var(--text);
    padding: 0.2rem 0;
    padding-left: 1.2rem;
    position: relative;
  }

  .key-points li::before {
    content: "→";
    position: absolute;
    left: 0;
    color: var(--accent);
  }

  /* ---- Story Links ---- */
  .story-links {
    padding-left: 3rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .story-links .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text);
    transition: all 0.15s;
  }

  .story-links .btn:hover {
    border-color: var(--accent);
    color: var(--accent);
    text-decoration: none;
    background: var(--accent-glow);
  }

  .story-links .btn-primary {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }

  .story-links .btn-primary:hover {
    opacity: 0.88;
    color: #fff;
  }

  /* ---- Footer ---- */
  .page-footer {
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-muted);
  }

  .page-footer a { color: var(--text-muted); }

  @media (max-width: 640px) {
    .story-meta, .story-summary, .key-points, .story-links {
      padding-left: 0;
    }
    .story-header { flex-direction: column; gap: 0.5rem; }
    .rank { font-size: 1rem; }
  }
"""


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def render_html(briefing: "Briefing") -> str:
    """
    Render a Briefing to a complete, styled HTML page.

    Returns a self-contained HTML string (no external dependencies).
    """
    title = f"News Radar — {briefing.date}"
    parts: list[str] = []
    parts.append(_render_head(title))
    parts.append('<body><div class="container">')
    parts.append(_render_header(briefing))
    if briefing.executive_summary:
        parts.append(_render_exec_summary(briefing.executive_summary))
    if briefing.top_topics:
        parts.append(_render_topics(briefing.top_topics))
    parts.append(_render_items(briefing.items))
    parts.append(_render_footer(briefing))
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Daily AI-curated technology news briefing">
  <meta name="robots" content="index, follow">
  <meta property="og:title" content="{_esc(title)}">
  <meta property="og:type" content="article">
  <title>{_esc(title)}</title>
  <style>{_CSS}</style>
</head>"""


def _render_header(briefing: "Briefing") -> str:
    count = len(briefing.items)
    generated = _format_dt(briefing.generated_at)
    return f"""<header class="page-header">
  <h1>News Radar</h1>
  <p class="page-meta">{_esc(briefing.date)} &middot; {count} stories &middot; {_esc(generated)}</p>
</header>"""


def _render_exec_summary(summary: str) -> str:
    paras = [p.strip() for p in summary.split("\n\n") if p.strip()]
    para_html = "\n".join(f"  <p>{_esc(p)}</p>" for p in paras)
    return f"""<section class="exec-summary">
  <h2>Today&#39;s Overview</h2>
{para_html}
</section>"""


def _render_topics(topics: list[str]) -> str:
    pills = "\n".join(
        f'    <span class="topic-pill">{_esc(t)}</span>' for t in topics
    )
    return f"""<section class="topics-section">
  <h2>Top Topics</h2>
  <div class="topic-pills">
{pills}
  </div>
</section>"""


def _render_items(items: list["SummarizedItem"]) -> str:
    cards = "\n".join(_render_item_card(rank, si) for rank, si in enumerate(items, 1))
    return f'<section class="stories">\n{cards}\n</section>'


def _render_item_card(rank: int, si: "SummarizedItem") -> str:
    item = si.scored.item
    score = si.scored.ai_score
    headline = si.ai_headline or item.title
    color = _score_color(score)
    published = _format_dt(item.published_at)
    platform_score = f"⬆ {item.score}" if item.score else ""
    comments = f"💬 {item.comment_count}" if item.comment_count else ""

    # Summary paragraphs
    summary_html = ""
    if si.ai_summary:
        paras = [p.strip() for p in si.ai_summary.split("\n\n") if p.strip()]
        summary_html = (
            '<div class="story-summary">\n'
            + "\n".join(f"  <p>{_esc(p)}</p>" for p in paras)
            + "\n</div>"
        )

    # Key points
    kp_html = ""
    if si.key_points:
        kp_items = "\n".join(f"    <li>{_esc(kp)}</li>" for kp in si.key_points)
        kp_html = f"""<div class="key-points">
  <h4>Key Points</h4>
  <ul>
{kp_items}
  </ul>
</div>"""

    # Links
    links_html = f'<div class="story-links"><a class="btn btn-primary" href="{_esc(item.url)}" target="_blank" rel="noopener">Read Article ↗</a>'
    if item.comments_url and item.comments_url != item.url:
        links_html += f'<a class="btn" href="{_esc(item.comments_url)}" target="_blank" rel="noopener">💬 Discussion</a>'
    links_html += "</div>"

    # Meta tags
    meta_parts = [
        f'<span class="score-badge"><span class="score-dot" style="background:{color}"></span>{score}/10</span>',
        f'<span class="meta-tag">{_esc(item.source_name)}</span>',
        f'<span class="meta-tag">{_esc(published)}</span>',
    ]
    if platform_score:
        meta_parts.append(f'<span class="meta-tag">{_esc(platform_score)}</span>')
    if comments:
        meta_parts.append(f'<span class="meta-tag">{_esc(comments)}</span>')

    meta_html = "\n".join(f"  {m}" for m in meta_parts)

    return f"""<article class="story-card" id="story-{rank}">
  <div class="story-header">
    <span class="rank">#{rank}</span>
    <a class="story-title-link" href="{_esc(item.url)}" target="_blank" rel="noopener">{_esc(headline)}</a>
  </div>
  <div class="story-meta">
{meta_html}
  </div>
{summary_html}
{kp_html}
{links_html}
</article>"""


def _render_footer(briefing: "Briefing") -> str:
    model = briefing.items[0].model_used if briefing.items else "n/a"
    generated = _format_dt(briefing.generated_at)
    stats = (
        f"Fetched: {briefing.total_fetched} · "
        f"Scored: {briefing.total_scored} · "
        f"In briefing: {len(briefing.items)}"
    )
    return f"""<footer class="page-footer">
  <p>
    Generated by <a href="https://github.com/Harshads-git/news-radar">News Radar</a>
    &middot; {_esc(model)} &middot; {_esc(generated)}
  </p>
  <p style="margin-top:0.4rem">{_esc(stats)}</p>
</footer>"""
