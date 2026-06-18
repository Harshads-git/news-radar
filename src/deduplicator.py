"""
src/deduplicator.py
===================
Cross-source deduplication engine for News Radar.

Problem: The same story often appears across multiple sources.
  - HN frontpage + HN RSS feed both carry the same 30 top stories
  - Reddit r/programming + r/tech both link to the same TechCrunch article
  - An article is posted, then re-posted 3 days later with a slight title change

Two-stage deduplication:
  Stage 1 — URL Normalization + Equality
    Strip tracking parameters, fragments, and trailing slashes, then
    compare canonical URLs. This catches the most common case: the exact
    same article linked from two different sources.

  Stage 2 — Jaccard Title Similarity
    Compare tokenized titles using the Jaccard coefficient. If two stories
    have a similarity score above TITLE_SIMILARITY_THRESHOLD, the lower-
    scored item is considered a duplicate. This catches:
      - "Python 4.0 Released" vs "Python Version 4.0 is Out"
      - "OpenAI Releases GPT-5" vs "OpenAI Launches GPT-5 Model"

  Stage 1 runs first (O(n) with hash set), Stage 2 only runs on the
  remaining items (still O(n²) in the worst case but with a much smaller n).

Scoring: when two items are considered duplicates, we keep the one with:
  1. Higher platform score (HN points / Reddit upvotes)
  2. If equal score, the one fetched earlier (first-seen wins)

Usage:
    from src.deduplicator import Deduplicator
    deduped = Deduplicator().deduplicate(all_items)
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.logger import get_logger
from src.models import NewsItem

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Jaccard similarity threshold: 0.0 = never match, 1.0 = exact match only
# 0.6 means "60% of title tokens must be shared to be considered a duplicate"
TITLE_SIMILARITY_THRESHOLD = 0.6

# URL query parameters that are tracking/analytics only (not part of content)
_TRACKING_PARAMS = frozenset(
    {
        # UTM parameters (Google Analytics)
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_source_platform", "utm_creative_format",
        # Facebook
        "fbclid", "fb_ref", "fb_source",
        # Twitter / X
        "twclid", "t", "s",
        # LinkedIn
        "lipi",
        # HubSpot
        "hsCtaTracking", "hstc", "hssc", "hsfp",
        # Misc tracking
        "ref", "source", "mc_cid", "mc_eid",
        "_ga", "_gl", "gclid", "dclid",
    }
)

# Canonical domain mappings: redirect domains → their actual domain
# Handles shortened URLs that redirect to well-known domains
_CANONICAL_DOMAINS: dict[str, str] = {
    "amp.reddit.com": "www.reddit.com",
    "old.reddit.com": "www.reddit.com",
    "np.reddit.com": "www.reddit.com",
    "mobile.twitter.com": "twitter.com",
    "mobile.x.com": "x.com",
}

# Title tokens to ignore when computing Jaccard similarity
# (stop words that add noise to the comparison)
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "this", "that", "it", "its",
        "i", "you", "he", "she", "we", "they", "what", "which",
        "who", "how", "why", "when", "where",
    }
)


# ---------------------------------------------------------------------------
# URL Normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication comparison.

    Transformations applied (in order):
      1. Strip leading/trailing whitespace
      2. Lowercase the scheme and hostname
      3. Remove URL fragment (#section)
      4. Remove known tracking query parameters (UTM, fbclid, etc.)
      5. Sort remaining query parameters (so ?a=1&b=2 == ?b=2&a=1)
      6. Strip trailing slash from the path
      7. Apply canonical domain mappings (e.g. old.reddit.com → www.reddit.com)
      8. Remove redundant default ports (:80 for http, :443 for https)

    Returns
    -------
    str
        Normalized URL suitable for equality comparison.
        Returns the original URL if parsing fails.

    Examples
    --------
    >>> normalize_url("https://example.com/article/?utm_source=hn&ref=home")
    'https://example.com/article'
    >>> normalize_url("https://old.reddit.com/r/python/comments/abc123/")
    'https://www.reddit.com/r/python/comments/abc123'
    """
    if not url:
        return url
    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    # 1. Lowercase scheme and host
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    # 2. Apply canonical domain mapping
    # Split off port before looking up domain
    host = netloc.split(":")[0]
    port = netloc.split(":")[1] if ":" in netloc else ""
    host = _CANONICAL_DOMAINS.get(host, host)
    netloc = f"{host}:{port}" if port else host

    # 3. Remove default ports
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        netloc = host

    # 4. Strip tracking query parameters + sort the rest
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        cleaned = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
        # Sort for canonical ordering
        query = urlencode(
            sorted((k, v[0]) for k, v in cleaned.items() if v),
            doseq=False,
        )
    else:
        query = ""

    # 5. Strip trailing slash from path
    path = parsed.path.rstrip("/") or "/"
    # Ensure single leading slash
    if path != "/" and not path.startswith("/"):
        path = "/" + path

    # 6. Drop fragment entirely
    fragment = ""

    normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
    return normalized


# ---------------------------------------------------------------------------
# Jaccard Title Similarity
# ---------------------------------------------------------------------------


def tokenize_title(title: str) -> frozenset[str]:
    """
    Convert a story title into a set of meaningful tokens.

    Process:
      1. Lowercase
      2. Extract only alphanumeric words (strip punctuation)
      3. Remove stop words
      4. Return a frozenset for O(1) set operations

    Examples
    --------
    >>> tokenize_title("Python 4.0 Released with Major Performance Gains")
    frozenset({'python', '40', 'released', 'major', 'performance', 'gains'})
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", title.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """
    Compute the Jaccard similarity coefficient between two token sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns a float in [0.0, 1.0]:
      0.0 = completely different sets
      1.0 = identical sets

    Returns 0.0 if both sets are empty (avoids division by zero).

    Examples
    --------
    >>> jaccard_similarity({"python", "released"}, {"python", "released"})
    1.0
    >>> jaccard_similarity({"python", "released"}, {"rust", "launched"})
    0.0
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def are_similar_titles(title_a: str, title_b: str, threshold: float = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """
    Return True if two story titles are similar enough to be considered duplicates.

    Parameters
    ----------
    title_a, title_b:
        Raw story headline strings.
    threshold:
        Jaccard similarity threshold. Default is TITLE_SIMILARITY_THRESHOLD (0.6).
    """
    tokens_a = tokenize_title(title_a)
    tokens_b = tokenize_title(title_b)
    similarity = jaccard_similarity(tokens_a, tokens_b)
    return similarity >= threshold


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class Deduplicator:
    """
    Two-stage deduplication engine for NewsItem lists.

    Stage 1: URL normalization equality check (O(n) — hash set lookup)
    Stage 2: Jaccard title similarity check (O(n²) — pairwise comparison)

    When two items are considered duplicates, the "winner" is:
      - The item with the higher platform score (HN points / Reddit upvotes)
      - If scores are equal, the first item seen (preserves input order)

    Usage
    -----
    ::

        deduped_items = Deduplicator().deduplicate(all_items_from_all_sources)
    """

    def __init__(
        self,
        title_threshold: float = TITLE_SIMILARITY_THRESHOLD,
        enable_title_dedup: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        title_threshold:
            Jaccard threshold for title similarity deduplication.
            Set to 1.0 to disable title dedup (only URL dedup runs).
        enable_title_dedup:
            If False, only URL-based dedup runs. Useful for testing.
        """
        self.title_threshold = title_threshold
        self.enable_title_dedup = enable_title_dedup

    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Deduplicate a list of NewsItems using two-stage comparison.

        Parameters
        ----------
        items:
            Raw combined list from all scrapers (may have duplicates).

        Returns
        -------
        list[NewsItem]
            Deduplicated list, preserving relative order of first-seen items.
        """
        if len(items) <= 1:
            return list(items)

        before = len(items)

        # Stage 1: URL deduplication
        stage1 = self._dedup_by_url(items)
        url_removed = before - len(stage1)

        # Stage 2: Title similarity deduplication
        if self.enable_title_dedup:
            stage2 = self._dedup_by_title(stage1)
            title_removed = len(stage1) - len(stage2)
        else:
            stage2 = stage1
            title_removed = 0

        total_removed = before - len(stage2)
        if total_removed > 0:
            log.debug(
                "Deduplication: %d → %d items (-%d URL, -%d title)",
                before,
                len(stage2),
                url_removed,
                title_removed,
            )

        return stage2

    # ------------------------------------------------------------------
    # Stage 1: URL deduplication
    # ------------------------------------------------------------------

    def _dedup_by_url(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove exact URL duplicates after normalization.

        When two items share the same normalized URL, keep the one with
        the higher platform score (or the first one if scores are equal).
        """
        # Map normalized_url → best NewsItem seen so far
        seen: dict[str, NewsItem] = {}

        for item in items:
            norm = normalize_url(item.url)
            if norm not in seen:
                seen[norm] = item
            else:
                existing = seen[norm]
                # Keep the one with higher score
                if self._score(item) > self._score(existing):
                    seen[norm] = item

        # Preserve original order using first-seen position
        result_set = set(id(v) for v in seen.values())
        return [item for item in items if id(item) in result_set]

    # ------------------------------------------------------------------
    # Stage 2: Title similarity deduplication
    # ------------------------------------------------------------------

    def _dedup_by_title(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove near-duplicate items based on Jaccard title similarity.

        Algorithm:
          - Tokenize each title once
          - Compare each pair (i, j) where j > i  [upper triangle only]
          - If Jaccard(i, j) ≥ threshold → mark the lower-scored as duplicate
          - Return items that were never marked as duplicates
        """
        n = len(items)
        if n <= 1:
            return list(items)

        # Pre-tokenize all titles
        tokens = [tokenize_title(item.title) for item in items]
        is_duplicate = [False] * n

        for i in range(n):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, n):
                if is_duplicate[j]:
                    continue
                sim = jaccard_similarity(tokens[i], tokens[j])
                if sim >= self.title_threshold:
                    # Mark the lower-scoring item as duplicate
                    if self._score(items[j]) > self._score(items[i]):
                        is_duplicate[i] = True
                        break  # item[i] is gone, no need to compare further
                    else:
                        is_duplicate[j] = True

        return [item for item, dup in zip(items, is_duplicate) if not dup]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(item: NewsItem) -> int:
        """Return the platform score for ranking duplicates. Default 0."""
        return item.score or 0
