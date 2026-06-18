"""
src/search.py
=============
DuckDuckGo web search for fetching background context before AI summarization.

Why web context matters:
  A story title like "OpenAI announces Project Strawberry" means nothing
  without context. Before summarizing, we query DuckDuckGo for background
  information, then inject it into the AI prompt. This lets the AI write
  summaries like:
    "Project Strawberry is OpenAI's internal codename for their reasoning
    model, later revealed as o1. This announcement introduces..."

  Rather than just:
    "OpenAI announces a new project."

Design:
  - Uses the DuckDuckGo Instant Answer API (no API key needed)
  - Returns a short plain-text snippet (max 500 chars)
  - Falls back to empty string on any error (never blocks the pipeline)
  - Results are cached in memory per run (avoids duplicate API calls
    for the same query within a single pipeline execution)

Usage:
    from src.search import fetch_web_context
    context = await fetch_web_context("OpenAI Project Strawberry reasoning model")
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache

import httpx

from src.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DDG_URL = "https://api.duckduckgo.com/"
_USER_AGENT = "NewsRadar/0.1 (+https://github.com/Harshads-git/news-radar)"
_REQUEST_TIMEOUT = 8.0
_MAX_CONTEXT_CHARS = 500

# In-memory cache for the current run (dict[query_hash → context_text])
_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_web_context(query: str, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """
    Fetch a short background context snippet from DuckDuckGo for the query.

    Parameters
    ----------
    query:
        Search query, typically the story title + key topic words.
    max_chars:
        Maximum length of the returned context string.

    Returns
    -------
    str
        Plain-text background context (may be empty on failure or no result).

    Notes
    -----
    - Results are cached per query hash for the duration of the process.
    - Never raises exceptions — returns "" on any error.
    """
    if not query or not query.strip():
        return ""

    query = query.strip()
    cache_key = _hash_query(query)

    if cache_key in _cache:
        return _cache[cache_key]

    context = await _fetch_ddg_context(query, max_chars)
    _cache[cache_key] = context
    return context


def clear_context_cache() -> None:
    """Clear the in-memory context cache. Useful between test runs."""
    _cache.clear()


# ---------------------------------------------------------------------------
# DuckDuckGo Instant Answer API
# ---------------------------------------------------------------------------


async def _fetch_ddg_context(query: str, max_chars: int) -> str:
    """
    Query DuckDuckGo's Instant Answer API and extract the best snippet.

    The DDG API response contains:
      - AbstractText: Wikipedia-sourced summary (best quality)
      - Abstract: shorter abstract (fallback)
      - Answer: direct answer for simple queries
      - RelatedTopics[].Text: secondary snippets

    We prefer AbstractText > Abstract > Answer > RelatedTopics[0].
    """
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "t": "NewsRadar",
    }

    try:
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(_DDG_URL, params=params)

        if response.status_code != 200:
            log.debug("DDG returned HTTP %d for query: %s", response.status_code, query[:60])
            return ""

        data = response.json()

    except (httpx.TransportError, httpx.TimeoutException) as e:
        log.debug("DDG request failed: %s", e)
        return ""
    except Exception as e:
        log.debug("DDG unexpected error: %s", e)
        return ""

    # Extract best available text
    context = _extract_best_text(data, max_chars)
    if context:
        log.debug("DDG context (%d chars) for: %s", len(context), query[:50])
    return context


def _extract_best_text(data: dict, max_chars: int) -> str:
    """
    Extract the best available text snippet from a DDG API response dict.

    Priority order:
      1. AbstractText (Wikipedia abstract — richest context)
      2. Abstract (shorter abstract)
      3. Answer (direct answer for factual queries)
      4. First RelatedTopics text (secondary result)
    """
    # Priority 1: AbstractText
    if text := (data.get("AbstractText") or "").strip():
        return text[:max_chars]

    # Priority 2: Abstract
    if text := (data.get("Abstract") or "").strip():
        return text[:max_chars]

    # Priority 3: Answer
    if text := (data.get("Answer") or "").strip():
        return text[:max_chars]

    # Priority 4: First related topic
    topics = data.get("RelatedTopics") or []
    for topic in topics:
        if isinstance(topic, dict) and (text := (topic.get("Text") or "").strip()):
            return text[:max_chars]

    return ""


def _hash_query(query: str) -> str:
    """Return a short hash of the query string for use as a cache key."""
    return hashlib.md5(query.lower().encode(), usedforsecurity=False).hexdigest()[:16]
