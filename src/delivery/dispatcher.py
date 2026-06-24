"""
src/delivery/dispatcher.py
===========================
Delivery dispatcher — routes a Briefing to all configured channels.

The dispatcher iterates over all enabled delivery channels and sends
concurrently. One channel failing does NOT block others.

Channels (enabled if their config is present in Settings):
  1. Email    → SMTP HTML email
  2. Discord  → Discord webhook embed
  3. Slack    → Slack Block Kit webhook
  4. Custom   → JSON POST to any URL

Usage:
    from src.delivery.dispatcher import DeliveryDispatcher
    from src.config import settings

    dispatcher = DeliveryDispatcher(settings)
    results = await dispatcher.dispatch(briefing)
    # results = {"email": True, "discord": False, ...}

Design:
  - Uses asyncio.gather(return_exceptions=True) for parallel delivery
  - Returns a dict of channel → success (bool) for visibility
  - Logs success/failure per channel
  - Does nothing if no channels are configured (not an error)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.config import Settings
    from src.models import Briefing

log = get_logger(__name__)


class DeliveryDispatcher:
    """
    Routes the completed Briefing to all configured delivery channels.

    Parameters
    ----------
    settings:
        Application settings (determines which channels are configured).
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    async def dispatch(self, briefing: "Briefing") -> dict[str, bool]:
        """
        Send the briefing to all configured channels concurrently.

        Parameters
        ----------
        briefing:
            The completed Briefing to deliver.

        Returns
        -------
        dict[str, bool]
            Mapping of channel name → success flag.
            Only includes channels that were attempted.
        """
        channels = self._enabled_channels()

        if not channels:
            log.info("No delivery channels configured — skipping delivery")
            return {}

        log.section("Phase 8: Delivery")
        log.info("Delivering to %d channel(s): %s", len(channels), ", ".join(channels.keys()))

        # Run all deliveries concurrently
        names = list(channels.keys())
        tasks = [self._deliver_one(name, fn, briefing) for name, fn in channels.items()]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, bool] = {}
        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, Exception):
                results[name] = False
            else:
                results[name] = bool(outcome)

        success_count = sum(1 for v in results.values() if v)
        log.success(
            "Delivery complete: %d/%d channels succeeded",
            success_count,
            len(results),
        )
        return results

    def _enabled_channels(self) -> dict[str, object]:
        """
        Build a dict of channel name → coroutine-returning callable
        for each configured delivery channel.
        """
        from src.delivery.email import EmailDelivery
        from src.delivery.webhook import CustomWebhookDelivery, DiscordDelivery, SlackDelivery

        channels: dict[str, object] = {}
        s = self.settings

        if s.has_email:
            channels["email"] = EmailDelivery(s).send
        if s.has_discord:
            channels["discord"] = DiscordDelivery(s).send
        if s.has_slack:
            channels["slack"] = SlackDelivery(s).send
        if s.custom_webhook_url:
            channels["custom"] = CustomWebhookDelivery(s).send

        return channels

    async def _deliver_one(self, name: str, send_fn: object, briefing: "Briefing") -> bool:
        """
        Call one delivery function, catching and logging any error.

        Returns True on success, False on failure.
        """
        from collections.abc import Callable

        try:
            await send_fn(briefing)  # type: ignore[operator]
            return True
        except Exception as e:
            log.error("Delivery to %s failed: %s", name, e)
            return False

    # ------------------------------------------------------------------
    # Convenience: check what's configured
    # ------------------------------------------------------------------

    def configured_channels(self) -> list[str]:
        """Return names of all configured delivery channels."""
        return list(self._enabled_channels().keys())

    def has_any_channel(self) -> bool:
        """True if at least one delivery channel is configured."""
        s = self.settings
        return any([s.has_email, s.has_discord, s.has_slack, bool(s.custom_webhook_url)])
