"""
src/exceptions.py
=================
Custom exception hierarchy for the News Radar pipeline.

Design principles:
  1. Every exception inherits from NewsRadarError — catch-all for the pipeline.
  2. Exceptions are named after the pipeline STAGE that raises them, not the
     root cause. This makes error handling at the orchestrator level clear.
  3. Exceptions carry structured context (source_id, model, url) so log messages
     are self-contained without needing to re-read variables.

Hierarchy:
    NewsRadarError              ← base for all pipeline errors
      FetchError                ← raised by scrapers (network, parsing)
        RateLimitError          ← HTTP 429 from any source API
        ParseError              ← malformed feed / unexpected API shape
      AIError                   ← raised by AI provider calls
        TokenLimitError         ← prompt too long for model context window
        AIProviderError         ← API key invalid, quota exceeded, 5xx
      StorageError              ← raised by storage layer (read/write)
      ConfigError               ← raised at startup for bad configuration
      DeliveryError             ← raised by email/webhook delivery

Usage:
    from src.exceptions import FetchError, AIError

    # In a scraper:
    raise FetchError("RSS feed returned 503", source_id="hn-rss", url=feed_url)

    # In the orchestrator (catch and log, then continue):
    try:
        items = await scraper.fetch(source)
    except FetchError as e:
        log.error("Scraper failed for %s: %s", e.source_id, e)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------


class NewsRadarError(Exception):
    """
    Base class for all News Radar exceptions.

    Provides a consistent ``context`` dict that subclasses populate with
    structured metadata (source_id, model, url, etc.) so log messages
    are fully self-contained.
    """

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


# ---------------------------------------------------------------------------
# Fetch Errors  (raised by src/scrapers/)
# ---------------------------------------------------------------------------


class FetchError(NewsRadarError):
    """
    Raised when a scraper fails to retrieve items from a source.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    source_id:
        The ``id`` field of the SourceConfig that triggered the error.
    url:
        The URL that was being fetched (if applicable).
    status_code:
        HTTP status code returned (if applicable).

    Example
    -------
    ::

        raise FetchError(
            "Feed returned HTTP 503",
            source_id="hn-rss",
            url="https://hnrss.org/frontpage",
            status_code=503,
        )
    """

    def __init__(
        self,
        message: str,
        *,
        source_id: str = "",
        url: str = "",
        status_code: int | None = None,
        **extra: object,
    ) -> None:
        super().__init__(message, source_id=source_id, url=url, **extra)
        self.source_id = source_id
        self.url = url
        self.status_code = status_code


class RateLimitError(FetchError):
    """
    Raised when a source API returns HTTP 429 (Too Many Requests).

    Includes the ``retry_after`` seconds hint if the API provides it.
    The orchestrator can use this to implement back-off logic.
    """

    def __init__(
        self,
        message: str,
        *,
        source_id: str = "",
        url: str = "",
        retry_after: int = 60,
        **extra: object,
    ) -> None:
        super().__init__(message, source_id=source_id, url=url, status_code=429, **extra)
        self.retry_after = retry_after


class ParseError(FetchError):
    """
    Raised when a scraper receives data it cannot parse into NewsItem models.

    Common causes:
      - RSS feed with unexpected XML schema
      - Reddit API returning a non-standard shape
      - HN API item missing required fields
    """

    def __init__(
        self,
        message: str,
        *,
        source_id: str = "",
        url: str = "",
        raw_data: str = "",
        **extra: object,
    ) -> None:
        super().__init__(message, source_id=source_id, url=url, **extra)
        self.raw_data = raw_data[:200] if raw_data else ""  # truncate for safety


# ---------------------------------------------------------------------------
# AI Errors  (raised by src/ai/)
# ---------------------------------------------------------------------------


class AIError(NewsRadarError):
    """
    Raised when an AI provider call fails.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    model:
        The model identifier being called (e.g. "gpt-4o-mini").
    item_url:
        URL of the NewsItem being processed (for debugging).
    """

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        item_url: str = "",
        **extra: object,
    ) -> None:
        super().__init__(message, model=model, item_url=item_url, **extra)
        self.model = model
        self.item_url = item_url


class TokenLimitError(AIError):
    """
    Raised when a prompt exceeds the model's context window.

    The orchestrator handles this by either truncating the prompt or
    skipping the item for summarization.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        item_url: str = "",
        token_count: int = 0,
        token_limit: int = 0,
        **extra: object,
    ) -> None:
        super().__init__(message, model=model, item_url=item_url, **extra)
        self.token_count = token_count
        self.token_limit = token_limit


class AIProviderError(AIError):
    """
    Raised for fatal AI API failures: invalid key, quota exceeded, 5xx errors.

    Unlike transient errors (which can be retried), these typically require
    user action (fix the API key, add credits, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        model: str = "",
        status_code: int | None = None,
        **extra: object,
    ) -> None:
        super().__init__(message, model=model, **extra)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Storage Errors  (raised by src/storage/)
# ---------------------------------------------------------------------------


class StorageError(NewsRadarError):
    """
    Raised when the storage layer fails to read or write data.

    Common causes:
      - Disk full
      - Permission denied on data directory
      - Corrupted JSON in a briefing file
    """

    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        operation: str = "",
        **extra: object,
    ) -> None:
        super().__init__(message, path=path, operation=operation, **extra)
        self.path = path
        self.operation = operation  # "read", "write", "delete"


# ---------------------------------------------------------------------------
# Config Errors  (raised by src/config.py at startup)
# ---------------------------------------------------------------------------


class ConfigError(NewsRadarError):
    """
    Raised when required configuration is missing or invalid at startup.

    Typically caught in main.py and shown to the user with a helpful message
    pointing them to .env.example.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str = "",
        expected: str = "",
        **extra: object,
    ) -> None:
        super().__init__(message, field=field, expected=expected, **extra)
        self.field = field
        self.expected = expected


# ---------------------------------------------------------------------------
# Delivery Errors  (raised by src/services/)
# ---------------------------------------------------------------------------


class DeliveryError(NewsRadarError):
    """
    Raised when a delivery channel fails to send the briefing.

    The orchestrator treats delivery errors as non-fatal — the briefing is
    saved to disk even if email or webhook delivery fails.
    """

    def __init__(
        self,
        message: str,
        *,
        channel: str = "",
        **extra: object,
    ) -> None:
        super().__init__(message, channel=channel, **extra)
        self.channel = channel  # "email", "discord", "slack", "webhook"
