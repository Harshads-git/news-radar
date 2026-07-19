"""
src/delivery/email_template.py
================================
Email-optimized HTML renderer for the daily News Radar briefing.

Why a separate renderer?
  Generic HTML pages use CSS classes and <style> blocks freely.
  Email clients (Gmail, Outlook, Apple Mail) are notoriously hostile
  to modern CSS:
    - Outlook strips <style> blocks entirely
    - Gmail ignores many CSS properties
    - Most clients don't support CSS Grid, Flexbox, or custom properties

  This renderer uses:
    1. INLINE styles on every element (no external/embedded CSS)
    2. TABLE-based layout (not divs) for Outlook compatibility
    3. A limited, email-safe CSS property set (no custom properties)
    4. Max-width 600px (standard email width)
    5. Conditional comments for Outlook 2007-2019 (MSO)

Design palette (dark-mode aware):
  Background  : #0f1117  (near-black)
  Surface     : #1a1d27  (dark navy card)
  Border      : #2d3147
  Text        : #e2e4ef
  Muted       : #7b82a4
  Accent blue : #6c8fff
  Accent purple: #a78bfa
  Score green : hsl(120, 80%, 45%)
  Score yellow: hsl(60, 80%, 45%)
  Score red   : hsl(0, 80%, 45%)

Usage:
    from src.delivery.email_template import render_email_html

    html_str = render_email_html(briefing)
    # → self-contained email HTML, ready for MIMEText("html")
"""

from __future__ import annotations

import html as html_mod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Briefing, SummarizedItem

# ---------------------------------------------------------------------------
# Inline style constants
# ---------------------------------------------------------------------------

_BG = "#0f1117"
_SURFACE = "#1a1d27"
_BORDER = "#2d3147"
_TEXT = "#e2e4ef"
_MUTED = "#7b82a4"
_ACCENT = "#6c8fff"
_ACCENT2 = "#a78bfa"
_FONT = "Arial, 'Helvetica Neue', Helvetica, sans-serif"


def _score_color(score: int) -> str:
    """Return a hex color for score 1-10 (green→yellow→red)."""
    if score >= 8:
        return "#4ade80"   # bright green
    if score >= 6:
        return "#facc15"   # amber
    if score >= 4:
        return "#f97316"   # orange
    return "#f87171"       # red


def _esc(text: str) -> str:
    return html_mod.escape(str(text or ""), quote=True)


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_email_html(briefing: "Briefing") -> str:
    """
    Render a Briefing as an email-safe, inline-styled HTML string.

    Parameters
    ----------
    briefing:
        The daily Briefing to render.

    Returns
    -------
    str
        A complete HTML document safe for use as an email body (MIME text/html).
        All styles are inlined; no external resources are referenced.
    """
    head = _render_head(briefing.date)
    header = _render_header(briefing)
    topics_bar = _render_topics_bar(briefing.top_topics)
    items_section = _render_items(briefing.items)
    footer_html = _render_footer(briefing)

    return f"""\
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
{head}
<body style="margin:0;padding:0;background-color:{_BG};font-family:{_FONT};color:{_TEXT};-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%;">

<!--[if mso]>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
<tr><td align="center">
<![endif]-->

<!-- Outer wrapper -->
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
  style="background-color:{_BG};min-height:100vh;">
<tr><td align="center" style="padding:24px 12px;">

  <!-- Inner 600px container -->
  <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0"
    style="max-width:600px;width:100%;background-color:{_SURFACE};border-radius:12px;
           border:1px solid {_BORDER};overflow:hidden;">

    {header}
    {topics_bar}
    {items_section}
    {footer_html}

  </table>
  <!-- /inner container -->

</td></tr>
</table>
<!-- /outer wrapper -->

<!--[if mso]>
</td></tr>
</table>
<![endif]-->

</body>
</html>"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_head(date: str) -> str:
    title = _esc(f"News Radar — {date}")
    return f"""\
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="color-scheme" content="dark" />
  <meta name="supported-color-schemes" content="dark" />
  <title>{title}</title>
  <!--[if mso]>
  <xml><o:OfficeDocumentSettings><o:AllowPNG/><o:PixelsPerInch>96</o:PixelsPerInch>
  </o:OfficeDocumentSettings></xml>
  <![endif]-->
  <style type="text/css">
    /* Minimal reset — kept here as fallback for clients that support <style> */
    body, table, td, p, a {{ margin:0; padding:0; }}
    img {{ border:0; display:block; }}
    a {{ color:{_ACCENT}; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    @media only screen and (max-width: 620px) {{
      .email-container {{ width:100% !important; }}
      .story-card {{ padding:16px !important; }}
    }}
  </style>
</head>"""


def _render_header(briefing: "Briefing") -> str:
    date_str = _esc(briefing.date)
    count = len(briefing.items)
    summary_html = ""
    if briefing.executive_summary:
        summary_text = _esc(briefing.executive_summary[:500])
        summary_html = f"""\
    <tr><td style="padding:0 28px 20px 28px;">
      <p style="margin:0;font-size:14px;line-height:1.7;color:{_MUTED};
                font-family:{_FONT};">{summary_text}</p>
    </td></tr>"""

    return f"""\
    <!-- Header -->
    <tr>
      <td style="padding:28px 28px 20px 28px;
                 background:linear-gradient(135deg, #1a1d27 0%, #12163a 100%);
                 border-bottom:1px solid {_BORDER};">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td>
              <p style="margin:0 0 4px 0;font-size:11px;font-weight:600;letter-spacing:0.1em;
                         text-transform:uppercase;color:{_MUTED};font-family:{_FONT};">Daily Briefing</p>
              <h1 style="margin:0;font-size:26px;font-weight:700;line-height:1.2;
                          color:{_TEXT};font-family:{_FONT};">
                &#x1F4E1; News Radar
              </h1>
              <p style="margin:6px 0 0 0;font-size:14px;color:{_MUTED};font-family:{_FONT};">
                {date_str} &nbsp;&middot;&nbsp; {count} stories
              </p>
            </td>
            <td align="right" valign="middle">
              <span style="display:inline-block;background:#12163a;border:1px solid {_BORDER};
                            border-radius:8px;padding:8px 14px;font-size:24px;line-height:1;">
                &#x1F4F0;
              </span>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    {summary_html}"""


def _render_topics_bar(top_topics: list[str]) -> str:
    if not top_topics:
        return ""
    pills = ""
    for topic in top_topics[:10]:
        pills += (
            f'<span style="display:inline-block;background:#12163a;border:1px solid {_BORDER};'
            f'color:{_ACCENT2};border-radius:20px;padding:3px 10px;font-size:12px;'
            f'font-family:{_FONT};margin:2px 3px;">{_esc(topic)}</span>'
        )
    return f"""\
    <!-- Topics bar -->
    <tr><td style="padding:12px 24px;border-bottom:1px solid {_BORDER};">
      <p style="margin:0 0 6px 0;font-size:11px;font-weight:600;letter-spacing:0.08em;
                 text-transform:uppercase;color:{_MUTED};font-family:{_FONT};">Today&apos;s Topics</p>
      <div>{pills}</div>
    </td></tr>"""


def _render_items(items: "list[SummarizedItem]") -> str:
    if not items:
        return f"""\
    <tr><td style="padding:32px 28px;text-align:center;">
      <p style="margin:0;color:{_MUTED};font-family:{_FONT};font-size:14px;">
        No stories met the relevance threshold today.
      </p>
    </td></tr>"""

    cards = ""
    for rank, si in enumerate(items, 1):
        cards += _render_item_card(rank, si)
    return cards


def _render_item_card(rank: int, si: "SummarizedItem") -> str:
    item = si.scored.item
    score = si.ai_score
    color = _score_color(score)
    headline = _esc(si.ai_headline or item.title)
    url = _esc(item.url)
    source = _esc(item.source_name)
    summary_para = ""
    if si.ai_summary:
        first_para = si.ai_summary.split("\n\n")[0].strip()[:300]
        summary_para = (
            f'<p style="margin:8px 0 0 0;font-size:13px;line-height:1.65;'
            f'color:{_MUTED};font-family:{_FONT};">{_esc(first_para)}</p>'
        )

    # Key points
    points_html = ""
    if si.key_points:
        pts = "".join(
            f'<li style="margin:0 0 4px 0;font-size:12px;color:{_MUTED};'
            f'font-family:{_FONT};">{_esc(pt)}</li>'
            for pt in si.key_points[:3]
        )
        points_html = (
            f'<ul style="margin:8px 0 0 16px;padding:0;list-style:disc;">{pts}</ul>'
        )

    # Topics
    topics_html = ""
    if si.scored.ai_topics:
        topic_spans = "".join(
            f'<span style="display:inline-block;background:#12163a;'
            f'color:{_ACCENT2};border-radius:12px;padding:2px 8px;'
            f'font-size:11px;font-family:{_FONT};margin:2px 2px 0 0;">'
            f'{_esc(t)}</span>'
            for t in si.scored.ai_topics[:5]
        )
        topics_html = f'<div style="margin-top:8px;">{topic_spans}</div>'

    bg = _SURFACE if rank % 2 == 1 else "#1e2135"
    border_top = f"border-top:1px solid {_BORDER};" if rank > 1 else ""

    return f"""\
    <!-- Story #{rank} -->
    <tr><td class="story-card"
      style="padding:20px 28px;background-color:{bg};{border_top}">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td valign="top" width="40">
            <div style="width:36px;height:36px;border-radius:50%;
                         background:{color}20;border:2px solid {color};
                         text-align:center;line-height:36px;
                         font-size:13px;font-weight:700;color:{color};
                         font-family:{_FONT};">{score}</div>
          </td>
          <td valign="top" style="padding-left:12px;">
            <p style="margin:0 0 2px 0;font-size:10px;color:{_MUTED};
                       font-family:{_FONT};text-transform:uppercase;
                       letter-spacing:0.06em;">#{rank} &nbsp;&middot;&nbsp; {source}</p>
            <h2 style="margin:0;font-size:16px;font-weight:600;line-height:1.35;
                        font-family:{_FONT};">
              <a href="{url}" style="color:{_TEXT};text-decoration:none;"
                 onmouseover="this.style.color='{_ACCENT}'"
                 onmouseout="this.style.color='{_TEXT}'">{headline}</a>
            </h2>
            {summary_para}
            {points_html}
            {topics_html}
            <p style="margin:10px 0 0 0;">
              <a href="{url}"
                 style="display:inline-block;background:#12163a;
                         border:1px solid {_BORDER};border-radius:6px;
                         padding:5px 12px;font-size:12px;color:{_ACCENT};
                         font-family:{_FONT};text-decoration:none;">
                Read article &#x2192;
              </a>
            </p>
          </td>
        </tr>
      </table>
    </td></tr>"""


def _render_footer(briefing: "Briefing") -> str:
    date_str = _esc(briefing.date)
    gen_time = _fmt_dt(briefing.generated_at)
    return f"""\
    <!-- Footer -->
    <tr><td style="padding:20px 28px;background-color:#12163a;
                   border-top:1px solid {_BORDER};text-align:center;">
      <p style="margin:0 0 6px 0;font-size:12px;color:{_MUTED};font-family:{_FONT};">
        &#x1F4E1; <strong style="color:{_TEXT};">News Radar</strong> &nbsp;&middot;&nbsp; {date_str}
      </p>
      <p style="margin:0;font-size:11px;color:#4a5068;font-family:{_FONT};">
        Generated {gen_time} &nbsp;&middot;&nbsp;
        This email was sent by News Radar, your personal AI news curator.
      </p>
    </td></tr>"""
