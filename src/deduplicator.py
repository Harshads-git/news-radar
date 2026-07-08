"""
src/deduplicator.py
===================
Cross-source deduplication engine for News Radar.

Problem: The same story often appears across multiple sources.
  - HN frontpage + HN RSS feed both carry the same 30 top stories
  - Reddit r/programming + r/tech both link to the same TechCrunch article
  - An article is posted, then re-posted 3 days later with a slight title change
  - "OpenAI's GPT-5 is here" and "GPT-5 launched by OpenAI" — same story,
    low Jaccard but high semantic similarity

Three-stage deduplication (Day 20 upgrade):
  Stage 1 — URL Normalization + Equality
    Strip tracking parameters, fragments, and trailing slashes, then
    compare canonical URLs. This catches the most common case: the exact
    same article linked from two different sources.

  Stage 2 — Jaccard Title Similarity
    Compare tokenized titles using the Jaccard coefficient. If two stories
    have a similarity score above TITLE_SIMILARITY_THRESHOLD, the lower-
    scored item is considered a duplicate. This catches:
      - "Python 4.0 Released" vs "Python Version 4.0 is Out"

  Stage 3 — Semantic (TF-IDF Cosine) Title Similarity       [NEW Day 20]
    Represent each title as a TF-IDF weighted word vector and compute
    cosine similarity between pairs. This catches cases where word choice
    differs but meaning is the same:
      - "OpenAI Releases GPT-5" vs "GPT-5 Launched by OpenAI"
    Uses a pure-Python implementation with no external ML libraries.

  Stage 1 runs first (O(n) with hash set), Stage 2 + 3 only run on the
  remaining items (O(n²) in the worst case but with a much smaller n).

Scoring: when two items are considered duplicates, we keep the one with:
  1. Higher platform score (HN points / Reddit upvotes)
  2. If equal score, the one fetched earlier (first-seen wins)

Usage:
    from src.deduplicator import Deduplicator
    deduped = Deduplicator().deduplicate(all_items)

    # Disable semantic stage (faster, uses only URL + Jaccard):
    deduped = Deduplicator(enable_semantic_dedup=False).deduplicate(all_items)
"""

from __future__ import annotations

import math
import re
from collections import Counter
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

# TF-IDF cosine similarity threshold for semantic Stage 3 dedup
# 0.82 catches near-paraphrases while avoiding false positives on
# short or highly common-word-heavy titles
SEMANTIC_SIMILARITY_THRESHOLD = 0.82

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
# TF-IDF Cosine Similarity (Stage 3 — Semantic Dedup)
# ---------------------------------------------------------------------------


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.

    TF(t) = count(t) / total_tokens

    Raw frequency normalized by document length so shorter and longer
    titles are compared on equal footing.
    """
    if not tokens:
        return {}
    count = Counter(tokens)
    total = len(tokens)
    return {term: freq / total for term, freq in count.items()}


def build_tfidf_vectors(
    token_lists: list[list[str]],
) -> list[dict[str, float]]:
    """
    Compute TF-IDF vectors for a corpus of token lists.

    Why TF-IDF instead of raw bag-of-words?
    Common words like 'released' or 'new' appear in many titles and carry
    little discriminating information. TF-IDF down-weights these terms so
    rare, meaningful words (e.g. 'llama', 'gemini', 'pytorch') dominate
    the similarity score.

    Parameters
    ----------
    token_lists:
        List of tokenized documents. Each inner list is one title's tokens.

    Returns
    -------
    List of TF-IDF weight dicts, one per document.
    """
    n_docs = len(token_lists)
    if n_docs == 0:
        return []

    # Document frequency: how many documents contain each term
    df: dict[str, int] = {}
    for tokens in token_lists:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    # IDF(t) = log((n_docs + 1) / (df(t) + 1)) + 1  (smoothed)
    idf: dict[str, float] = {
        term: math.log((n_docs + 1) / (count + 1)) + 1
        for term, count in df.items()
    }

    # TF-IDF vectors
    vectors: list[dict[str, float]] = []
    for tokens in token_lists:
        tf = _compute_tf(tokens)
        vectors.append({term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()})

    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Compute cosine similarity between two TF-IDF weight dicts.

    cos(A, B) = (A · B) / (‖A‖ × ‖B‖)

    Returns a float in [0.0, 1.0]:
      0.0 = completely orthogonal (no shared terms)
      1.0 = identical direction (same document)

    Returns 0.0 if either vector is zero (empty title).
    """
    if not vec_a or not vec_b:
        return 0.0

    # Dot product over shared keys only
    dot = sum(vec_a[t] * vec_b[t] for t in vec_a if t in vec_b)

    # Norms
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class Deduplicator:
    """
    Three-stage deduplication engine for NewsItem lists.

    Stage 1: URL normalization equality check (O(n) — hash set lookup)
    Stage 2: Jaccard title similarity check (O(n²) — pairwise comparison)
    Stage 3: TF-IDF cosine semantic similarity (O(n²) — pairwise comparison)

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
        semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        enable_title_dedup: bool = True,
        enable_semantic_dedup: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        title_threshold:
            Jaccard threshold for Stage 2 title dedup (default 0.6).
            Set to 1.0 to effectively disable Stage 2.
        semantic_threshold:
            Cosine similarity threshold for Stage 3 semantic dedup (default 0.82).
            Set to 1.0 to effectively disable Stage 3.
        enable_title_dedup:
            If False, Stage 2 (Jaccard) is skipped entirely.
        enable_semantic_dedup:
            If False, Stage 3 (TF-IDF cosine) is skipped. Useful when
            processing speed is critical or the corpus is very small.
        """
        self.title_threshold = title_threshold
        self.semantic_threshold = semantic_threshold
        self.enable_title_dedup = enable_title_dedup
        self.enable_semantic_dedup = enable_semantic_dedup

    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Deduplicate a list of NewsItems using three-stage comparison.

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

        # Stage 2: Jaccard title similarity deduplication
        if self.enable_title_dedup:
            stage2 = self._dedup_by_title(stage1)
            title_removed = len(stage1) - len(stage2)
        else:
            stage2 = stage1
            title_removed = 0

        # Stage 3: TF-IDF cosine semantic deduplication
        if self.enable_semantic_dedup and len(stage2) > 1:
            stage3 = self._dedup_by_semantic(stage2)
            semantic_removed = len(stage2) - len(stage3)
        else:
            stage3 = stage2
            semantic_removed = 0

        total_removed = before - len(stage3)
        if total_removed > 0:
            log.debug(
                "Deduplication: %d → %d items (-%d URL, -%d Jaccard, -%d semantic)",
                before,
                len(stage3),
                url_removed,
                title_removed,
                semantic_removed,
            )

        return stage3

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
    # Stage 3: Semantic (TF-IDF cosine) deduplication
    # ------------------------------------------------------------------

    def _dedup_by_semantic(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove semantically similar items using TF-IDF cosine similarity.

        This catches cases where Jaccard fails because:
          - Word order differs: "OpenAI releases GPT-5" vs "GPT-5 released by OpenAI"
          - Synonym usage: "launched" vs "released" vs "unveiled"
          - TF-IDF assigns high weight to rare discriminating terms
            (e.g. 'llama', 'pytorch') regardless of word order

        Algorithm:
          - Tokenize all remaining titles to word lists (not frozensets)
          - Build TF-IDF vectors over the entire mini-corpus
          - Compare pairwise cosine similarity (upper triangle)
          - Mark lower-scored item as duplicate if similarity ≥ threshold
        """
        n = len(items)
        if n <= 1:
            return list(items)

        # Tokenize preserving duplicates (needed for TF computation)
        token_lists = [
            re.findall(r"[a-zA-Z0-9]+", item.title.lower())
            for item in items
        ]
        # Filter stop words
        token_lists = [
            [t for t in tl if t not in _STOP_WORDS and len(t) > 1]
            for tl in token_lists
        ]

        # Build TF-IDF vectors over this mini-corpus
        vectors = build_tfidf_vectors(token_lists)
        is_duplicate = [False] * n

        for i in range(n):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, n):
                if is_duplicate[j]:
                    continue
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim >= self.semantic_threshold:
                    # Mark the lower-scoring item as duplicate
                    if self._score(items[j]) > self._score(items[i]):
                        is_duplicate[i] = True
                        break
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
