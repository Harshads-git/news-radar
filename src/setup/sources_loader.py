"""
src/setup/sources_loader.py
============================
Load and validate sources.json into SourcesConfig / SourceConfig models.

Responsibilities:
  1. Read sources.json from disk (path from Settings.sources_file)
  2. Parse JSON → SourcesConfig (Pydantic validates all fields)
  3. Filter to enabled sources only
  4. Provide diagnostic helpers (count by type, warn on empty)

Design:
  - Returns typed SourcesConfig — callers get IDE autocomplete
  - Raises clear ConfigError on missing/malformed file
  - Warns (not raises) when no sources are enabled — lets the pipeline
    run with zero results rather than crashing

Usage:
    from src.setup.sources_loader import load_sources
    from src.config import settings

    sources_config = load_sources(settings.sources_file)
    enabled = sources_config.enabled_sources
    print(f"Loaded {len(enabled)} enabled sources")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.exceptions import ConfigError
from src.logger import get_logger
from src.models import SourceConfig, SourcesConfig

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def load_sources(
    sources_file: Path | str,
    *,
    require_enabled: bool = False,
) -> SourcesConfig:
    """
    Load and validate sources.json.

    Parameters
    ----------
    sources_file:
        Path to the sources JSON file (typically data/sources.json).
    require_enabled:
        If True, raises ConfigError when no sources are enabled.
        Default False — logs a warning instead.

    Returns
    -------
    SourcesConfig
        Validated configuration containing all defined sources.
        Use .enabled_sources to get only enabled ones.

    Raises
    ------
    ConfigError
        - File not found
        - File is not valid JSON
        - Any source fails Pydantic validation
        - require_enabled=True and no sources are enabled
    """
    path = Path(sources_file)

    # ---- File existence check ----
    if not path.exists():
        raise ConfigError(
            f"Sources file not found: {path}\n"
            "Create data/sources.json or set SOURCES_FILE in your .env",
            field="sources_file",
        )

    # ---- Parse JSON ----
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"sources.json is not valid JSON: {e}",
            field="sources_file",
        ) from e
    except OSError as e:
        raise ConfigError(
            f"Cannot read sources file: {e}",
            field="sources_file",
        ) from e

    # ---- Pydantic validation ----
    try:
        sources_config = SourcesConfig.model_validate(data)
    except Exception as e:
        raise ConfigError(
            f"sources.json validation failed: {e}",
            field="sources_file",
        ) from e

    # ---- Diagnostics ----
    total = len(sources_config.sources)
    enabled = sources_config.enabled_sources
    enabled_count = len(enabled)

    log.info(
        "Loaded sources: %d total, %d enabled (%s)",
        total,
        enabled_count,
        _summarize_types(enabled),
    )

    if enabled_count == 0:
        msg = "No sources are enabled in sources.json"
        if require_enabled:
            raise ConfigError(msg, field="sources_file")
        log.warning("%s — pipeline will return no results", msg)

    return sources_config


def _summarize_types(sources: list[SourceConfig]) -> str:
    """Return a human-readable summary like '2 hackernews, 3 rss, 2 reddit'."""
    from collections import Counter
    counts = Counter(s.type for s in sources)
    return ", ".join(f"{n} {t}" for t, n in counts.most_common())


def get_sources_by_type(
    sources_config: SourcesConfig,
    source_type: str,
) -> list[SourceConfig]:
    """
    Filter enabled sources to those of a specific type.

    Parameters
    ----------
    sources_config:
        Loaded SourcesConfig.
    source_type:
        One of: 'rss', 'hackernews', 'reddit'

    Returns
    -------
    list[SourceConfig]
        Enabled sources matching the requested type.
    """
    return [
        s for s in sources_config.enabled_sources
        if s.type == source_type
    ]


def validate_sources_file(sources_file: Path | str) -> list[str]:
    """
    Validate a sources.json file and return a list of issues found.

    Returns an empty list if the file is valid.
    Designed for use in a --check CLI command without raising.

    Returns
    -------
    list[str]
        Human-readable validation issues. Empty = OK.
    """
    path = Path(sources_file)
    issues: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if "sources" not in data:
        issues.append("Missing top-level 'sources' key")
        return issues

    for i, src in enumerate(data.get("sources", [])):
        src_id = src.get("id", f"[index {i}]")
        try:
            SourceConfig.model_validate(src)
        except Exception as e:
            issues.append(f"Source '{src_id}': {e}")

    enabled_count = sum(
        1 for s in data.get("sources", []) if s.get("enabled", True)
    )
    if enabled_count == 0:
        issues.append("Warning: no sources are enabled")

    return issues
