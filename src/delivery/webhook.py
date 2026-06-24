"""
src/delivery/webhook.py
========================
Webhook delivery for Discord, Slack, and custom HTTP endpoints.

All webhooks are sent as HTTP POST requests using Python's built-in
urllib — no httpx/requests dependency needed for delivery.

Discord:
  Sends a rich embed with color-coded score, top stories, and a
  link to the full briefing on GitHub Pages.

Slack:
  Sends a Block Kit message with a header section and story list.

Custom:
  Sends the full Briefing as JSON, so any endpoint can process it.

Payload limits:
  - Discord embed description: max 4096 characters
  - Discord: max 10 embeds per message
  - Slack text: max 3000 characters per block
  - We cap at top 5 stories in webhook payloads (full briefing on GH Pages)

Usage:
    from src.delivery.webhook import DiscordDelivery, SlackDelivery, CustomWebhookDelivery

    discord = DiscordDelivery(settings)
    await discord.send(briefing)
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from src.exceptions import DeliveryError
from src.logger import get_logger

if TYPE_CHECKING:
    from src.config import Settings
    from src.models import Briefing

log = get_logger(__name__)

# Number of stories to include in webhook payloads
_WEBHOOK_STORY_LIMIT = 5


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, timeout: int = 15) -> None:
    """
    Send a JSON POST request using urllib (no external dependencies).

    Raises
    ------
    DeliveryError
        On HTTP errors, connection failures, or non-2xx responses.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "news-radar/0.1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            if status not in (200, 201, 204):
                raise DeliveryError(
                    f"Webhook returned HTTP {status}",
                    channel="webhook",
                )
    except urllib.error.HTTPError as e:
        raise DeliveryError(
            f"Webhook HTTP error {e.code}: {e.reason}",
            channel="webhook",
        ) from e
    except urllib.error.URLError as e:
        raise DeliveryError(
            f"Webhook connection failed: {e.reason}",
            channel="webhook",
        ) from e
    except OSError as e:
        raise DeliveryError(
            f"Webhook network error: {e}",
            channel="webhook",
        ) from e


async def _post_json_async(url: str, payload: dict) -> None:
    """Async wrapper: runs _post_json in a thread executor."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _post_json, url, payload)


# ---------------------------------------------------------------------------
# Discord Delivery
# ---------------------------------------------------------------------------

# Score → Discord embed color (decimal)
_DISCORD_COLORS = {
    10: 0x27AE60,  # green
    9: 0x2ECC71,   # emerald
    8: 0x3498DB,   # blue
    7: 0x9B59B6,   # purple
    6: 0xF39C12,   # orange
    5: 0xE67E22,   # dark orange
    4: 0xE74C3C,   # red
    3: 0xC0392B,   # dark red
}


def _discord_color(score: int) -> int:
    return _DISCORD_COLORS.get(score, 0x95A5A6)  # grey fallback


class DiscordDelivery:
    """
    Sends a rich Discord embed for the daily briefing.

    Each story becomes a field in one embed. Top 5 stories only,
    with a link to the full briefing.
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    async def send(self, briefing: "Briefing") -> None:
        """Send the briefing to Discord via webhook."""
        if not self.settings.has_discord:
            raise DeliveryError(
                "Discord delivery not configured. Set DISCORD_WEBHOOK_URL in .env",
                channel="discord",
            )

        payload = self._build_payload(briefing)
        log.info("Sending Discord webhook for %s", briefing.date)
        await _post_json_async(self.settings.discord_webhook_url, payload)
        log.success("Discord webhook delivered for %s", briefing.date)

    def _build_payload(self, briefing: "Briefing") -> dict:
        """Build the Discord webhook JSON payload."""
        top_items = briefing.items[:_WEBHOOK_STORY_LIMIT]
        top_score = top_items[0].scored.ai_score if top_items else 8

        fields = []
        for rank, si in enumerate(top_items, 1):
            item = si.scored.item
            headline = si.ai_headline or item.title
            score_str = f"`{si.scored.ai_score}/10`"
            # First paragraph of summary only
            summary_preview = ""
            if si.ai_summary:
                first_para = si.ai_summary.split("\n\n")[0].strip()
                summary_preview = f"\n{first_para[:200]}"

            fields.append({
                "name": f"{rank}. {headline[:150]}",
                "value": f"{score_str} · [{item.source_name}]({item.url}){summary_preview}",
                "inline": False,
            })

        # Executive summary truncated for embed description
        description = ""
        if briefing.executive_summary:
            description = briefing.executive_summary.split("\n\n")[0][:500]

        embed = {
            "title": f"\U0001f4e1 News Radar — {briefing.date}",
            "description": description,
            "color": _discord_color(top_score),
            "fields": fields,
            "footer": {
                "text": (
                    f"{len(briefing.items)} stories · "
                    f"Fetched: {briefing.total_fetched} · "
                    f"Threshold: {self.settings.score_threshold}/10"
                )
            },
        }

        # Add GitHub Pages link if available
        if self.settings.github_pages_enabled:
            repo = "https://harshads-git.github.io/news-radar"
            embed["url"] = repo

        return {"embeds": [embed]}


# ---------------------------------------------------------------------------
# Slack Delivery
# ---------------------------------------------------------------------------


class SlackDelivery:
    """
    Sends a Slack Block Kit message for the daily briefing.

    Uses Slack's Block Kit format for rich layout.
    Top 5 stories in a bullet list with scores.
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    async def send(self, briefing: "Briefing") -> None:
        """Send the briefing to Slack via webhook."""
        if not self.settings.has_slack:
            raise DeliveryError(
                "Slack delivery not configured. Set SLACK_WEBHOOK_URL in .env",
                channel="slack",
            )

        payload = self._build_payload(briefing)
        log.info("Sending Slack webhook for %s", briefing.date)
        await _post_json_async(self.settings.slack_webhook_url, payload)
        log.success("Slack webhook delivered for %s", briefing.date)

    def _build_payload(self, briefing: "Briefing") -> dict:
        """Build the Slack Block Kit JSON payload."""
        top_items = briefing.items[:_WEBHOOK_STORY_LIMIT]

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"\U0001f4e1 News Radar — {briefing.date}",
                    "emoji": True,
                },
            },
        ]

        # Executive summary
        if briefing.executive_summary:
            first_para = briefing.executive_summary.split("\n\n")[0][:500]
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": first_para},
            })

        blocks.append({"type": "divider"})

        # Story list
        story_lines = []
        for rank, si in enumerate(top_items, 1):
            item = si.scored.item
            headline = si.ai_headline or item.title
            story_lines.append(
                f"*{rank}.* <{item.url}|{headline[:100]}> "
                f"_{si.scored.ai_score}/10 · {item.source_name}_"
            )

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(story_lines),
            },
        })

        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (
                    f"Fetched {briefing.total_fetched} stories · "
                    f"{briefing.total_scored} passed threshold · "
                    f"Briefing: {len(briefing.items)} stories"
                ),
            }],
        })

        return {"blocks": blocks}


# ---------------------------------------------------------------------------
# Custom JSON Webhook
# ---------------------------------------------------------------------------


class CustomWebhookDelivery:
    """
    Sends the full Briefing as JSON to any custom HTTP endpoint.

    Useful for piping into n8n, Make.com, Zapier, or custom services.
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    async def send(self, briefing: "Briefing") -> None:
        """POST the briefing JSON to the custom webhook URL."""
        if not self.settings.custom_webhook_url:
            raise DeliveryError(
                "Custom webhook not configured. Set CUSTOM_WEBHOOK_URL in .env",
                channel="custom_webhook",
            )

        payload = self._build_payload(briefing)
        log.info("Sending custom webhook for %s", briefing.date)
        await _post_json_async(self.settings.custom_webhook_url, payload)
        log.success("Custom webhook delivered for %s", briefing.date)

    @staticmethod
    def _build_payload(briefing: "Briefing") -> dict:
        """Build a compact JSON payload for the custom webhook."""
        return {
            "date": briefing.date,
            "executive_summary": briefing.executive_summary,
            "top_topics": briefing.top_topics,
            "total_fetched": briefing.total_fetched,
            "total_scored": briefing.total_scored,
            "item_count": len(briefing.items),
            "items": [
                {
                    "rank": rank,
                    "headline": si.ai_headline or si.scored.item.title,
                    "url": si.scored.item.url,
                    "source": si.scored.item.source_name,
                    "score": si.scored.ai_score,
                    "summary": si.ai_summary.split("\n\n")[0][:300] if si.ai_summary else "",
                    "key_points": si.key_points,
                    "topics": si.scored.ai_topics,
                }
                for rank, si in enumerate(briefing.items, 1)
            ],
        }
