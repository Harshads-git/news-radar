"""
src/logger.py
=============
Centralised Rich-powered logger for the News Radar pipeline.

Why Rich over stdlib logging?
  - Colour-coded output with clear log levels
  - Automatic terminal width detection and wrapping
  - Syntax-highlighted tracebacks with local variables shown
  - Progress bars and spinners (used by the orchestrator)
  - Zero-config — works out of the box, no handler setup needed

Usage anywhere in the codebase:
    from src.logger import get_logger
    log = get_logger(__name__)

    log.info("Fetching sources...")
    log.debug("Item: %s", item.title)
    log.warning("Rate limited — sleeping 5s")
    log.error("Scraper failed", exc_info=True)
    log.success("Briefing generated!")   # Rich bonus level
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Custom theme for News Radar branding
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        # Log level colours
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        # Semantic highlight styles used with log.markup()
        "source": "bold cyan",
        "score": "bold magenta",
        "url": "underline blue",
        "count": "bold white",
        "success": "bold green",
        "phase": "bold yellow",
    }
)

# ---------------------------------------------------------------------------
# Module-level Rich Console (shared so progress bars don't break output)
# ---------------------------------------------------------------------------

# Force UTF-8 on Windows so Rich can render box-drawing chars.
# os.environ must be set BEFORE Console() is created.
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

console = Console(
    theme=_THEME,
    highlight=True,
    markup=True,
    stderr=False,      # stdout by default so CI captures it
    force_terminal=True,   # prevents Rich falling back to legacy Win32 renderer
)

# ---------------------------------------------------------------------------
# Internal: singleton root handler
# ---------------------------------------------------------------------------

_handler: RichHandler | None = None
_configured: bool = False


def _build_handler(level: int) -> RichHandler:
    """Build a RichHandler with News Radar settings."""
    handler = RichHandler(
        console=console,
        level=level,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,   # show local vars in tracebacks
        tracebacks_suppress=[          # hide internal library frames
            "pydantic",
            "httpx",
            "asyncio",
        ],
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    return handler


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once at application startup.

    Call this from ``src/main.py`` before any other imports that log.
    Subsequent calls are no-ops (idempotent).

    Parameters
    ----------
    level:
        Logging verbosity: "DEBUG", "INFO", "WARNING", "ERROR".
    """
    global _handler, _configured  # noqa: PLW0603

    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers (avoid duplicates in tests)
    root_logger.handlers.clear()

    _handler = _build_handler(numeric_level)
    root_logger.addHandler(_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> "NewsRadarLogger":
    """
    Return a logger for the given module name.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    NewsRadarLogger
        A thin wrapper around ``logging.Logger`` that adds
        ``success()`` and ``section()`` convenience methods.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("Fetching %d sources", 5)
    >>> log.success("Pipeline complete in %.1fs", 12.3)
    >>> log.section("Phase 2: Deduplication")
    """
    return NewsRadarLogger(logging.getLogger(name))


# ---------------------------------------------------------------------------
# NewsRadarLogger wrapper
# ---------------------------------------------------------------------------


class NewsRadarLogger:
    """
    Thin wrapper around ``logging.Logger`` with convenience methods.

    Adds:
      - ``success()`` — green checkmark message (uses INFO level)
      - ``section()`` — prints a separator with a phase header
      - Full pass-through of all stdlib logger methods
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    # ------------------------------------------------------------------
    # Standard log-level delegates
    # ------------------------------------------------------------------

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log an error with full exception traceback (call inside except block)."""
        self._logger.exception(msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Bonus convenience methods
    # ------------------------------------------------------------------

    def success(self, msg: str, *args: object, **kwargs: object) -> None:
        """
        Log a green success message at INFO level.
        Automatically prepends the [OK] tag.
        """
        if args:
            msg = msg % args
        self._logger.info("[success][OK] %s[/success]", msg, **kwargs)

    def section(self, title: str) -> None:
        """
        Print a visual separator marking the start of a pipeline phase.

        Example output:
            ──────────────────── Phase: Fetching ────────────────────
        """
        console.rule(f"[phase]{title}[/phase]", style="dim")

    def pipeline_start(self, date: str, sources: int) -> None:
        """Log the standard pipeline start banner."""
        console.rule("[bold cyan]>> News Radar Pipeline <<[/bold cyan]", style="cyan")
        self._logger.info(
            "[phase]Starting briefing for[/phase] [count]%s[/count] "
            "with [count]%d[/count] sources",
            date,
            sources,
        )

    def pipeline_end(self, items: int, duration: float) -> None:
        """Log the standard pipeline completion banner."""
        self._logger.info(
            "[success]Pipeline complete[/success] — "
            "[count]%d[/count] items in [count]%.1fs[/count]",
            items,
            duration,
        )
        console.rule(style="cyan")

    def isEnabledFor(self, level: int) -> bool:
        """Pass-through for stdlib compatibility."""
        return self._logger.isEnabledFor(level)


# ---------------------------------------------------------------------------
# Module-level convenience: configure with defaults on import
# ---------------------------------------------------------------------------

# Import-time configuration uses INFO level.
# main.py will call configure_logging(settings.log_level) to override.
configure_logging("INFO")
