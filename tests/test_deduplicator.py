"""
tests/test_deduplicator.py
==========================
Comprehensive unit tests for the deduplication engine.

Tests cover:
  - URL normalization (UTM stripping, canonical domains, trailing slashes)
  - Jaccard similarity coefficient correctness
  - Title tokenization (stop word removal, punctuation stripping)
  - Full Deduplicator pipeline (URL dedup + title dedup)
  - Score-based winner selection
  - Edge cases (empty lists, single items, all duplicates)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.deduplicator import (
    Deduplicator,
    are_similar_titles,
    jaccard_similarity,
    normalize_url,
    tokenize_title,
)
from src.models import NewsItem


# ---------------------------------------------------------------------------
# Helper: create minimal NewsItems
# ---------------------------------------------------------------------------


def make_item(
    url: str,
    title: str = "Default Title",
    score: int | None = None,
    source_id: str = "test",
) -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        source_id=source_id,
        source_name="Test Source",
        source_type="rss",
        score=score,
    )


# ---------------------------------------------------------------------------
# URL Normalization Tests
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    @pytest.mark.unit
    def test_strips_utm_source(self):
        url = "https://example.com/article?utm_source=newsletter"
        assert normalize_url(url) == "https://example.com/article"

    @pytest.mark.unit
    def test_strips_all_utm_params(self):
        url = "https://example.com/article?utm_source=hn&utm_medium=email&utm_campaign=weekly"
        assert normalize_url(url) == "https://example.com/article"

    @pytest.mark.unit
    def test_strips_fbclid(self):
        url = "https://example.com/article?fbclid=IwAR1234"
        assert normalize_url(url) == "https://example.com/article"

    @pytest.mark.unit
    def test_strips_ref_param(self):
        url = "https://example.com/post?ref=hackernews"
        assert normalize_url(url) == "https://example.com/post"

    @pytest.mark.unit
    def test_preserves_non_tracking_params(self):
        url = "https://example.com/search?q=python&page=2"
        result = normalize_url(url)
        assert "q=python" in result
        assert "page=2" in result

    @pytest.mark.unit
    def test_strips_url_fragment(self):
        url = "https://example.com/article#section-1"
        assert "#" not in normalize_url(url)

    @pytest.mark.unit
    def test_strips_trailing_slash(self):
        url = "https://example.com/article/"
        assert not normalize_url(url).endswith("/")

    @pytest.mark.unit
    def test_lowercases_scheme_and_host(self):
        url = "HTTPS://EXAMPLE.COM/Article"
        result = normalize_url(url)
        assert result.startswith("https://example.com")

    @pytest.mark.unit
    def test_old_reddit_maps_to_www(self):
        url = "https://old.reddit.com/r/python/comments/abc/"
        result = normalize_url(url)
        assert "www.reddit.com" in result
        assert "old.reddit.com" not in result

    @pytest.mark.unit
    def test_amp_reddit_maps_to_www(self):
        url = "https://amp.reddit.com/r/python/"
        result = normalize_url(url)
        assert "www.reddit.com" in result

    @pytest.mark.unit
    def test_same_article_different_tracking_normalizes_equal(self):
        url_a = "https://techcrunch.com/article/?utm_source=hn"
        url_b = "https://techcrunch.com/article/?utm_source=reddit"
        assert normalize_url(url_a) == normalize_url(url_b)

    @pytest.mark.unit
    def test_empty_url_returns_empty(self):
        assert normalize_url("") == ""

    @pytest.mark.unit
    def test_params_sorted_canonically(self):
        url_a = "https://example.com/search?b=2&a=1"
        url_b = "https://example.com/search?a=1&b=2"
        assert normalize_url(url_a) == normalize_url(url_b)


# ---------------------------------------------------------------------------
# Jaccard Similarity Tests
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    @pytest.mark.unit
    def test_identical_sets_return_one(self):
        s = frozenset(["python", "released", "major"])
        assert jaccard_similarity(s, s) == 1.0

    @pytest.mark.unit
    def test_disjoint_sets_return_zero(self):
        a = frozenset(["python", "released"])
        b = frozenset(["rust", "launched"])
        assert jaccard_similarity(a, b) == 0.0

    @pytest.mark.unit
    def test_partial_overlap(self):
        a = frozenset(["python", "released", "major"])
        b = frozenset(["python", "released", "new"])
        # |A ∩ B| = 2, |A ∪ B| = 4 → 0.5
        assert abs(jaccard_similarity(a, b) - 0.5) < 0.01

    @pytest.mark.unit
    def test_subset_returns_fraction(self):
        a = frozenset(["python", "new"])
        b = frozenset(["python", "new", "features", "coming"])
        # |A ∩ B| = 2, |A ∪ B| = 4 → 0.5
        assert jaccard_similarity(a, b) == 0.5

    @pytest.mark.unit
    def test_both_empty_returns_zero(self):
        assert jaccard_similarity(frozenset(), frozenset()) == 0.0

    @pytest.mark.unit
    def test_one_empty_returns_zero(self):
        assert jaccard_similarity(frozenset(["python"]), frozenset()) == 0.0


# ---------------------------------------------------------------------------
# Tokenize Title Tests
# ---------------------------------------------------------------------------


class TestTokenizeTitle:
    @pytest.mark.unit
    def test_lowercases_tokens(self):
        tokens = tokenize_title("Python Released")
        assert "python" in tokens
        assert "Python" not in tokens

    @pytest.mark.unit
    def test_removes_stop_words(self):
        tokens = tokenize_title("The Python language is great for AI")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "for" not in tokens
        assert "python" in tokens
        assert "great" in tokens

    @pytest.mark.unit
    def test_strips_punctuation(self):
        # '4.0' → regex extracts '4' and '0' as separate tokens; dot is stripped
        # '4' and '0' are single chars so filtered out; version becomes absent from set
        tokens = tokenize_title("Python 4.0: A New Era!")
        assert "python" in tokens
        assert "era" in tokens  # non-stop, non-single-char word preserved
        assert "<" not in str(tokens)  # no angle brackets

    @pytest.mark.unit
    def test_returns_frozenset(self):
        assert isinstance(tokenize_title("Python test"), frozenset)

    @pytest.mark.unit
    def test_empty_title_returns_empty_frozenset(self):
        assert tokenize_title("") == frozenset()

    @pytest.mark.unit
    def test_single_char_tokens_excluded(self):
        tokens = tokenize_title("A B Python C")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "python" in tokens


# ---------------------------------------------------------------------------
# Are Similar Titles Tests
# ---------------------------------------------------------------------------


class TestAreSimilarTitles:
    @pytest.mark.unit
    def test_identical_titles_similar(self):
        assert are_similar_titles("Python 4.0 Released", "Python 4.0 Released")

    @pytest.mark.unit
    def test_completely_different_not_similar(self):
        assert not are_similar_titles("Python 4.0 Released", "Rust Overtakes CPP in Systems")

    @pytest.mark.unit
    def test_minor_variation_similar(self):
        # Titles sharing most key nouns/adjectives → high Jaccard
        t1 = "Python Released Major Performance Gains Benchmarks"
        t2 = "Python Released Major Performance Gains Results"
        assert are_similar_titles(t1, t2)

    @pytest.mark.unit
    def test_repost_detected(self):
        t1 = "OpenAI Releases GPT-5 Model with Advanced Reasoning"
        t2 = "OpenAI Launches GPT-5 Advanced Reasoning Model"
        assert are_similar_titles(t1, t2)

    @pytest.mark.unit
    def test_custom_threshold(self):
        # With threshold=1.0 only exact matches pass
        assert not are_similar_titles(
            "Python Released Major Update",
            "Python Released Minor Patch",
            threshold=1.0,
        )


# ---------------------------------------------------------------------------
# Deduplicator — URL Stage Tests
# ---------------------------------------------------------------------------


class TestDeduplicatorUrlStage:
    @pytest.mark.unit
    def test_dedup_removes_exact_url_duplicates(self):
        items = [
            make_item("https://example.com/article"),
            make_item("https://example.com/article"),
            make_item("https://other.com/post"),
        ]
        result = Deduplicator(enable_title_dedup=False, enable_semantic_dedup=False).deduplicate(items)
        assert len(result) == 2

    @pytest.mark.unit
    def test_dedup_removes_utm_duplicates(self):
        items = [
            make_item("https://example.com/article?utm_source=hn"),
            make_item("https://example.com/article?utm_source=reddit"),
        ]
        result = Deduplicator(enable_title_dedup=False, enable_semantic_dedup=False).deduplicate(items)
        assert len(result) == 1

    @pytest.mark.unit
    def test_dedup_removes_trailing_slash_duplicates(self):
        items = [
            make_item("https://example.com/article/"),
            make_item("https://example.com/article"),
        ]
        result = Deduplicator(enable_title_dedup=False, enable_semantic_dedup=False).deduplicate(items)
        assert len(result) == 1

    @pytest.mark.unit
    def test_winner_has_higher_score(self):
        items = [
            make_item("https://example.com/article?utm_source=hn", score=100),
            make_item("https://example.com/article?utm_source=reddit", score=842),
        ]
        result = Deduplicator(enable_title_dedup=False, enable_semantic_dedup=False).deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 842

    @pytest.mark.unit
    def test_unique_urls_preserved(self):
        items = [
            make_item("https://a.com/1", title="Unique Story About Rust"),
            make_item("https://b.com/2", title="Unique Story About Python"),
            make_item("https://c.com/3", title="Unique Story About Go Lang"),
        ]
        result = Deduplicator(enable_title_dedup=False, enable_semantic_dedup=False).deduplicate(items)
        assert len(result) == 3

    @pytest.mark.unit
    def test_empty_list_returns_empty(self):
        assert Deduplicator().deduplicate([]) == []

    @pytest.mark.unit
    def test_single_item_returned_unchanged(self):
        item = make_item("https://example.com/solo")
        result = Deduplicator().deduplicate([item])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Deduplicator — Title Similarity Stage Tests
# ---------------------------------------------------------------------------


class TestDeduplicatorTitleStage:
    @pytest.mark.unit
    def test_dedup_removes_title_duplicates(self):
        items = [
            make_item("https://a.com/1", "Python Released Major Performance Gains Benchmarks"),
            make_item("https://b.com/1", "Python Released Major Performance Gains Results"),
            make_item("https://c.com/2", "Rust Overtakes CPP in Systems Programming Survey"),
        ]
        # threshold=0.5: first two titles share 5/6 tokens → Jaccard ~0.83 → deduped
        result = Deduplicator(title_threshold=0.5).deduplicate(items)
        assert len(result) == 2

    @pytest.mark.unit
    def test_title_winner_has_higher_score(self):
        items = [
            make_item("https://a.com/1", "OpenAI Releases GPT5 Advanced Reasoning", score=50),
            make_item("https://b.com/2", "OpenAI Launches GPT5 Advanced Reasoning Model", score=500),
        ]
        result = Deduplicator(title_threshold=0.5).deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 500

    @pytest.mark.unit
    def test_title_dedup_disabled(self):
        """With enable_title_dedup=False, similar titles are NOT deduped."""
        items = [
            make_item("https://a.com/1", "Python 4.0 Released Major Performance Update"),
            make_item("https://b.com/2", "Python 4.0 Released Huge Performance Improvements"),
        ]
        result = Deduplicator(enable_title_dedup=False).deduplicate(items)
        assert len(result) == 2

    @pytest.mark.unit
    def test_all_unique_titles_preserved(self):
        items = [
            make_item("https://a.com/1", "Python 4.0 Released"),
            make_item("https://b.com/2", "Rust Overtakes CPP Systems"),
            make_item("https://c.com/3", "OpenAI Announces Strawberry Model"),
        ]
        result = Deduplicator().deduplicate(items)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Search Module Tests
# ---------------------------------------------------------------------------


class TestFetchWebContext:
    @pytest.mark.unit
    async def test_empty_query_returns_empty(self):
        from src.search import fetch_web_context
        result = await fetch_web_context("")
        assert result == ""

    @pytest.mark.unit
    async def test_whitespace_query_returns_empty(self):
        from src.search import fetch_web_context
        result = await fetch_web_context("   ")
        assert result == ""

    @pytest.mark.unit
    async def test_result_cached_on_second_call(self):
        from src.search import _memory_cache, clear_context_cache, fetch_web_context
        clear_context_cache()

        with patch("src.search._fetch_ddg_context", new_callable=AsyncMock,
                   return_value="Python is a high-level programming language.") as mock_fetch:
            result1 = await fetch_web_context("Python programming language")
            result2 = await fetch_web_context("Python programming language")

        # Should only have fetched once (second call uses memory cache)
        assert mock_fetch.call_count == 1
        assert result1 == result2

    @pytest.mark.unit
    def test_extract_best_text_prefers_abstract_text(self):
        from src.search import _extract_best_text
        data = {
            "AbstractText": "Best text here",
            "Abstract": "Shorter text",
            "Answer": "Direct answer",
        }
        result = _extract_best_text(data, 500)
        assert result == "Best text here"

    @pytest.mark.unit
    def test_extract_best_text_falls_back_to_abstract(self):
        from src.search import _extract_best_text
        data = {"AbstractText": "", "Abstract": "Abstract text", "Answer": ""}
        result = _extract_best_text(data, 500)
        assert result == "Abstract text"

    @pytest.mark.unit
    def test_extract_best_text_truncates_at_max_chars(self):
        from src.search import _extract_best_text
        data = {"AbstractText": "x" * 1000, "Abstract": "", "Answer": ""}
        result = _extract_best_text(data, 500)
        assert len(result) <= 500

    @pytest.mark.unit
    def test_extract_best_text_empty_response_returns_empty(self):
        from src.search import _extract_best_text
        data = {"AbstractText": "", "Abstract": "", "Answer": "", "RelatedTopics": []}
        assert _extract_best_text(data, 500) == ""

    @pytest.mark.unit
    def test_hash_query_is_deterministic(self):
        from src.search import _hash_query
        assert _hash_query("Python") == _hash_query("Python")

    @pytest.mark.unit
    def test_hash_query_is_case_insensitive(self):
        from src.search import _hash_query
        assert _hash_query("Python") == _hash_query("python")
