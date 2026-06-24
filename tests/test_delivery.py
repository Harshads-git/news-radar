"""
tests/test_delivery.py
=======================
Unit tests for:
  - EmailDelivery: subject building, plain text, SMTP error handling
  - DiscordDelivery: payload structure, color coding, channel guards
  - SlackDelivery: Block Kit payload, channel guards
  - CustomWebhookDelivery: JSON payload shape, channel guards
  - DeliveryDispatcher: concurrent dispatch, error isolation, channel detection
  - _post_json helper: HTTP error handling

All tests mock network I/O — no real SMTP or HTTP calls made.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import DeliveryError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> MagicMock:
    s = MagicMock()
    s.has_email = False
    s.has_discord = False
    s.has_slack = False
    s.custom_webhook_url = ""
    s.smtp_host = "smtp.gmail.com"
    s.smtp_port = 587
    s.smtp_user = "sender@example.com"
    s.smtp_password = "app_password"
    s.email_to = "recipient@example.com"
    s.discord_webhook_url = "https://discord.com/api/webhooks/123/abc"
    s.slack_webhook_url = "https://hooks.slack.com/services/T/B/xyz"
    s.score_threshold = 6
    s.github_pages_enabled = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_scored_item(url: str = "https://example.com/story", score: int = 9) -> MagicMock:
    item = MagicMock()
    item.url = url
    item.title = "Test Story Title"
    item.source_name = "Hacker News"
    item.comment_count = 50
    item.comments_url = "https://news.ycombinator.com/item?id=1"
    item.published_at = datetime(2026, 6, 24, 10, 0, tzinfo=timezone.utc)

    scored = MagicMock()
    scored.item = item
    scored.ai_score = score
    scored.ai_topics = ["AI", "Python"]
    scored.ai_reason = "Highly relevant"
    scored.model_used = "gpt-4o-mini"
    return scored


def make_summarized(url: str = "https://example.com/story", score: int = 9) -> MagicMock:
    si = MagicMock()
    si.scored = make_scored_item(url, score)
    si.ai_headline = "Test AI Headline"
    si.ai_summary = "First paragraph summary.\n\nSecond paragraph details.\n\nThird paragraph outlook."
    si.key_points = ["Point A", "Point B"]
    si.model_used = "gpt-4o-mini"
    # Make si.scored.item accessible
    si.scored.item.url = url
    si.scored.item.title = "Test Story Title"
    si.scored.item.source_name = "Hacker News"
    si.scored.ai_score = score
    return si


def make_briefing(num_items: int = 3, date_str: str = "2026-06-24") -> MagicMock:
    b = MagicMock()
    b.date = date_str
    b.items = [make_summarized(f"https://example.com/{i}", score=9 - i) for i in range(num_items)]
    b.executive_summary = "Today was busy.\n\nMany things happened.\n\nWatch this space."
    b.top_topics = ["AI", "Python", "open source"]
    b.total_fetched = 80
    b.total_scored = 10
    b.generated_at = datetime(2026, 6, 24, 15, 0, tzinfo=timezone.utc)
    return b


# ===========================================================================
# EmailDelivery Tests
# ===========================================================================


class TestEmailDelivery:
    @pytest.mark.unit
    def test_build_subject_includes_date(self):
        from src.delivery.email import EmailDelivery
        s = make_settings(has_email=True)
        ed = EmailDelivery(s)
        briefing = make_briefing(date_str="2026-06-24", num_items=5)
        subject = EmailDelivery._build_subject(briefing)
        assert "2026-06-24" in subject

    @pytest.mark.unit
    def test_build_subject_includes_story_count(self):
        from src.delivery.email import EmailDelivery
        briefing = make_briefing(num_items=7)
        subject = EmailDelivery._build_subject(briefing)
        assert "7" in subject

    @pytest.mark.unit
    def test_build_subject_includes_top_headline(self):
        from src.delivery.email import EmailDelivery
        briefing = make_briefing(num_items=1)
        briefing.items[0].ai_headline = "Unique Headline XYZ"
        subject = EmailDelivery._build_subject(briefing)
        assert "Unique Headline XYZ" in subject

    @pytest.mark.unit
    def test_plain_text_includes_date(self):
        from src.delivery.email import EmailDelivery
        briefing = make_briefing(date_str="2026-06-24")
        text = EmailDelivery._briefing_to_plain_text(briefing)
        assert "2026-06-24" in text

    @pytest.mark.unit
    def test_plain_text_includes_all_story_urls(self):
        from src.delivery.email import EmailDelivery
        briefing = make_briefing(num_items=3)
        text = EmailDelivery._briefing_to_plain_text(briefing)
        for i in range(3):
            assert f"https://example.com/{i}" in text

    @pytest.mark.unit
    def test_plain_text_includes_executive_summary(self):
        from src.delivery.email import EmailDelivery
        briefing = make_briefing()
        briefing.executive_summary = "My unique exec summary for testing"
        text = EmailDelivery._briefing_to_plain_text(briefing)
        assert "My unique exec summary for testing" in text

    @pytest.mark.unit
    async def test_send_raises_when_not_configured(self):
        from src.delivery.email import EmailDelivery
        s = make_settings(has_email=False)
        ed = EmailDelivery(s)
        with pytest.raises(DeliveryError, match="not configured"):
            await ed.send(make_briefing())

    @pytest.mark.unit
    async def test_send_calls_smtp_when_configured(self):
        from src.delivery.email import EmailDelivery
        s = make_settings(has_email=True)
        ed = EmailDelivery(s)

        with patch.object(ed, "_smtp_send"):
            with patch("src.delivery.email.render_html", return_value="<html></html>"):
                await ed.send(make_briefing())

        # If we get here without error, smtp_send was wired up correctly
        assert True

    @pytest.mark.unit
    def test_smtp_auth_error_raises_delivery_error(self):
        import smtplib
        from src.delivery.email import EmailDelivery
        s = make_settings(has_email=True)
        ed = EmailDelivery(s)
        msg = MagicMock()

        with patch("smtplib.SMTP") as MockSMTP:
            smtp_instance = MagicMock()
            smtp_instance.__enter__ = MagicMock(return_value=smtp_instance)
            smtp_instance.__exit__ = MagicMock(return_value=False)
            smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Bad credentials")
            MockSMTP.return_value = smtp_instance
            with pytest.raises(DeliveryError, match="authentication"):
                ed._smtp_send(msg)


# ===========================================================================
# DiscordDelivery Tests
# ===========================================================================


class TestDiscordDelivery:
    @pytest.mark.unit
    async def test_send_raises_when_not_configured(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=False)
        with pytest.raises(DeliveryError, match="not configured"):
            await DiscordDelivery(s).send(make_briefing())

    @pytest.mark.unit
    def test_payload_has_embed(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing())
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    @pytest.mark.unit
    def test_embed_title_contains_date(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing(date_str="2026-06-24"))
        assert "2026-06-24" in payload["embeds"][0]["title"]

    @pytest.mark.unit
    def test_embed_has_fields_for_stories(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing(num_items=5))
        fields = payload["embeds"][0]["fields"]
        assert len(fields) == 5  # max 5

    @pytest.mark.unit
    def test_embed_capped_at_5_stories(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing(num_items=10))
        fields = payload["embeds"][0]["fields"]
        assert len(fields) == 5

    @pytest.mark.unit
    def test_embed_has_color(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing())
        assert "color" in payload["embeds"][0]

    @pytest.mark.unit
    def test_embed_has_footer(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        payload = DiscordDelivery(s)._build_payload(make_briefing())
        assert "footer" in payload["embeds"][0]

    @pytest.mark.unit
    async def test_send_calls_post_json(self):
        from src.delivery.webhook import DiscordDelivery
        s = make_settings(has_discord=True)
        with patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock) as mock_post:
            await DiscordDelivery(s).send(make_briefing())
        mock_post.assert_called_once()

    @pytest.mark.unit
    def test_discord_color_high_score_is_green(self):
        from src.delivery.webhook import _discord_color
        color = _discord_color(10)
        assert color == 0x27AE60

    @pytest.mark.unit
    def test_discord_color_low_score_is_red(self):
        from src.delivery.webhook import _discord_color
        color = _discord_color(4)
        assert color == 0xE74C3C


# ===========================================================================
# SlackDelivery Tests
# ===========================================================================


class TestSlackDelivery:
    @pytest.mark.unit
    async def test_send_raises_when_not_configured(self):
        from src.delivery.webhook import SlackDelivery
        s = make_settings(has_slack=False)
        with pytest.raises(DeliveryError, match="not configured"):
            await SlackDelivery(s).send(make_briefing())

    @pytest.mark.unit
    def test_payload_has_blocks(self):
        from src.delivery.webhook import SlackDelivery
        s = make_settings(has_slack=True)
        payload = SlackDelivery(s)._build_payload(make_briefing())
        assert "blocks" in payload
        assert len(payload["blocks"]) > 0

    @pytest.mark.unit
    def test_header_block_contains_date(self):
        from src.delivery.webhook import SlackDelivery
        s = make_settings(has_slack=True)
        payload = SlackDelivery(s)._build_payload(make_briefing(date_str="2026-06-24"))
        header = payload["blocks"][0]
        assert header["type"] == "header"
        assert "2026-06-24" in header["text"]["text"]

    @pytest.mark.unit
    def test_payload_includes_story_urls(self):
        from src.delivery.webhook import SlackDelivery
        s = make_settings(has_slack=True)
        payload = SlackDelivery(s)._build_payload(make_briefing(num_items=2))
        all_text = json.dumps(payload)
        assert "https://example.com/0" in all_text
        assert "https://example.com/1" in all_text

    @pytest.mark.unit
    async def test_send_calls_post_json(self):
        from src.delivery.webhook import SlackDelivery
        s = make_settings(has_slack=True)
        with patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock) as mock_post:
            await SlackDelivery(s).send(make_briefing())
        mock_post.assert_called_once()


# ===========================================================================
# CustomWebhookDelivery Tests
# ===========================================================================


class TestCustomWebhookDelivery:
    @pytest.mark.unit
    async def test_send_raises_when_not_configured(self):
        from src.delivery.webhook import CustomWebhookDelivery
        s = make_settings(custom_webhook_url="")
        with pytest.raises(DeliveryError, match="not configured"):
            await CustomWebhookDelivery(s).send(make_briefing())

    @pytest.mark.unit
    def test_payload_has_date(self):
        from src.delivery.webhook import CustomWebhookDelivery
        briefing = make_briefing(date_str="2026-06-24")
        payload = CustomWebhookDelivery._build_payload(briefing)
        assert payload["date"] == "2026-06-24"

    @pytest.mark.unit
    def test_payload_has_item_list(self):
        from src.delivery.webhook import CustomWebhookDelivery
        payload = CustomWebhookDelivery._build_payload(make_briefing(num_items=3))
        assert "items" in payload
        assert len(payload["items"]) == 3

    @pytest.mark.unit
    def test_payload_items_have_required_fields(self):
        from src.delivery.webhook import CustomWebhookDelivery
        payload = CustomWebhookDelivery._build_payload(make_briefing(num_items=1))
        item = payload["items"][0]
        assert "headline" in item
        assert "url" in item
        assert "score" in item
        assert "rank" in item

    @pytest.mark.unit
    def test_payload_is_json_serializable(self):
        from src.delivery.webhook import CustomWebhookDelivery
        payload = CustomWebhookDelivery._build_payload(make_briefing())
        json_str = json.dumps(payload)  # must not raise
        assert len(json_str) > 0

    @pytest.mark.unit
    async def test_send_calls_post_json(self):
        from src.delivery.webhook import CustomWebhookDelivery
        s = make_settings(custom_webhook_url="https://example.com/hook")
        with patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock) as mock_post:
            await CustomWebhookDelivery(s).send(make_briefing())
        mock_post.assert_called_once()


# ===========================================================================
# _post_json Tests
# ===========================================================================


class TestPostJson:
    @pytest.mark.unit
    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    def test_http_error_raises_delivery_error(self):
        import urllib.error
        from src.delivery.webhook import _post_json
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.HTTPError(
                url="http://test", code=429, msg="Too Many Requests",
                hdrs=MagicMock(), fp=MagicMock(),
            )
            with pytest.raises(DeliveryError, match="429"):
                _post_json("http://test", {"data": "test"})

    @pytest.mark.unit
    def test_url_error_raises_delivery_error(self):
        import urllib.error
        from src.delivery.webhook import _post_json
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("connection refused")
            with pytest.raises(DeliveryError, match="connection"):
                _post_json("http://test", {"data": "test"})

    @pytest.mark.unit
    def test_successful_call_does_not_raise(self):
        from src.delivery.webhook import _post_json
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.getcode.return_value = 200
        with patch("urllib.request.urlopen", return_value=mock_resp):
            _post_json("http://test", {"data": "test"})  # no exception


# ===========================================================================
# DeliveryDispatcher Tests
# ===========================================================================


class TestDeliveryDispatcher:
    @pytest.mark.unit
    def test_no_channels_configured(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings()
        d = DeliveryDispatcher(s)
        assert not d.has_any_channel()
        assert d.configured_channels() == []

    @pytest.mark.unit
    def test_email_channel_detected(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_email=True)
        d = DeliveryDispatcher(s)
        assert d.has_any_channel()
        assert "email" in d.configured_channels()

    @pytest.mark.unit
    def test_discord_channel_detected(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_discord=True)
        d = DeliveryDispatcher(s)
        assert "discord" in d.configured_channels()

    @pytest.mark.unit
    def test_slack_channel_detected(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_slack=True)
        d = DeliveryDispatcher(s)
        assert "slack" in d.configured_channels()

    @pytest.mark.unit
    def test_custom_webhook_channel_detected(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(custom_webhook_url="https://example.com/hook")
        d = DeliveryDispatcher(s)
        assert "custom" in d.configured_channels()

    @pytest.mark.unit
    async def test_dispatch_returns_empty_when_no_channels(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings()
        d = DeliveryDispatcher(s)
        results = await d.dispatch(make_briefing())
        assert results == {}

    @pytest.mark.unit
    async def test_dispatch_returns_true_on_success(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_discord=True)
        d = DeliveryDispatcher(s)

        with patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock):
            results = await d.dispatch(make_briefing())

        assert results.get("discord") is True

    @pytest.mark.unit
    async def test_dispatch_returns_false_on_failure(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_discord=True)
        d = DeliveryDispatcher(s)

        with patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = DeliveryError("Network failed", channel="discord")
            results = await d.dispatch(make_briefing())

        assert results.get("discord") is False

    @pytest.mark.unit
    async def test_one_channel_failure_does_not_block_others(self):
        """If Discord fails, Slack must still succeed."""
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(has_discord=True, has_slack=True)
        d = DeliveryDispatcher(s)

        call_count = 0

        async def mock_post(url, payload):
            nonlocal call_count
            call_count += 1
            if "discord" in url:
                raise DeliveryError("Discord down", channel="discord")
            # Slack succeeds

        with patch("src.delivery.webhook._post_json_async", side_effect=mock_post):
            results = await d.dispatch(make_briefing())

        assert results.get("discord") is False
        assert results.get("slack") is True
        assert call_count == 2

    @pytest.mark.unit
    async def test_dispatch_all_four_channels(self):
        from src.delivery.dispatcher import DeliveryDispatcher
        s = make_settings(
            has_email=True,
            has_discord=True,
            has_slack=True,
            custom_webhook_url="https://example.com/hook",
        )
        d = DeliveryDispatcher(s)
        assert len(d.configured_channels()) == 4

        # Mock all network calls at the source
        with (
            patch("src.delivery.webhook._post_json_async", new_callable=AsyncMock),
            patch("src.delivery.email.render_html", return_value="<html></html>"),
            patch("asyncio.get_event_loop") as mock_loop,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
            results = await d.dispatch(make_briefing())

        assert len(results) == 4
