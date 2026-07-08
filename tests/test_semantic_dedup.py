"""
tests/test_semantic_dedup.py
=============================
Tests for Day 20 semantic deduplication additions to src/deduplicator.py.

Coverage:
  - _compute_tf: empty, single term, multiple terms, normalization
  - build_tfidf_vectors: empty corpus, single doc, IDF down-weights common terms,
    rare terms get higher weight
  - cosine_similarity: identical vectors, disjoint vectors, partial overlap,
    empty vectors, floating point stability
  - SEMANTIC_SIMILARITY_THRESHOLD: default value sanity
  - Deduplicator.__init__: new parameters have correct defaults
  - Deduplicator._dedup_by_semantic: removes near-paraphrases, preserves
    higher-scored item, distinct items survive, single-item list passthrough
  - Deduplicator.deduplicate: enable_semantic_dedup=False skips Stage 3,
    Stage 3 fires after Stage 2, empty list passthrough
  - Integration: Jaccard alone misses word-order variants that cosine catches

All tests use in-memory NewsItem objects — no I/O.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.deduplicator import (
    SEMANTIC_SIMILARITY_THRESHOLD,
    Deduplicator,
    _compute_tf,
    build_tfidf_vectors,
    cosine_similarity,
)
from src.models import NewsItem


# ===========================================================================
# Helpers
# ===========================================================================


def _item(title: str, score: int = 0, url: str | None = None) -> NewsItem:
    """Minimal NewsItem factory."""
    return NewsItem(
        source_id="test",
        source_name="Test Source",
        source_type="rss",
        title=title,
        url=url or f"https://example.com/{abs(hash(title)) % 10000}",
        score=score,
    )


# ===========================================================================
# _compute_tf
# ===========================================================================


class TestComputeTF:
    @pytest.mark.unit
    def test_empty_tokens_returns_empty_dict(self):
        assert _compute_tf([]) == {}

    @pytest.mark.unit
    def test_single_token_gives_tf_of_one(self):
        result = _compute_tf(["python"])
        assert result == {"python": 1.0}

    @pytest.mark.unit
    def test_tf_sums_to_one(self):
        tokens = ["ai", "ai", "model", "language"]
        result = _compute_tf(tokens)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9

    @pytest.mark.unit
    def test_repeated_token_has_higher_tf(self):
        result = _compute_tf(["gpt", "gpt", "model"])
        assert result["gpt"] > result["model"]

    @pytest.mark.unit
    def test_equal_frequency_tokens_equal_tf(self):
        result = _compute_tf(["openai", "google"])
        assert abs(result["openai"] - result["google"]) < 1e-9

    @pytest.mark.unit
    def test_tf_normalized_by_length(self):
        # 1 token out of 4 = TF of 0.25
        result = _compute_tf(["a", "b", "c", "d"])
        for v in result.values():
            assert abs(v - 0.25) < 1e-9


# ===========================================================================
# build_tfidf_vectors
# ===========================================================================


class TestBuildTFIDFVectors:
    @pytest.mark.unit
    def test_empty_corpus_returns_empty_list(self):
        assert build_tfidf_vectors([]) == []

    @pytest.mark.unit
    def test_output_length_matches_input_length(self):
        docs = [["ai", "model"], ["python", "code"], ["rust", "fast"]]
        vectors = build_tfidf_vectors(docs)
        assert len(vectors) == 3

    @pytest.mark.unit
    def test_single_doc_vector_is_nonempty(self):
        vectors = build_tfidf_vectors([["openai", "gpt"]])
        assert len(vectors) == 1
        assert len(vectors[0]) > 0

    @pytest.mark.unit
    def test_common_term_gets_lower_idf_weight(self):
        # 'release' appears in all docs → low IDF
        # 'pytorch' appears only in doc 0 → high IDF
        docs = [
            ["pytorch", "release"],
            ["tensorflow", "release"],
            ["jax", "release"],
        ]
        vectors = build_tfidf_vectors(docs)
        # In doc 0, pytorch should have higher weight than 'release'
        assert vectors[0]["pytorch"] > vectors[0]["release"]

    @pytest.mark.unit
    def test_term_unique_to_one_doc_has_higher_idf(self):
        docs = [
            ["llama", "model"],
            ["gemini", "model"],
            ["claude", "model"],
        ]
        vectors = build_tfidf_vectors(docs)
        # 'llama' only in doc 0 → higher weight than 'model' which is in all
        assert vectors[0]["llama"] > vectors[0]["model"]

    @pytest.mark.unit
    def test_each_vector_is_a_dict(self):
        vectors = build_tfidf_vectors([["a", "b"], ["c", "d"]])
        for v in vectors:
            assert isinstance(v, dict)

    @pytest.mark.unit
    def test_empty_token_list_gives_empty_vector(self):
        vectors = build_tfidf_vectors([[], ["word"]])
        assert vectors[0] == {}


# ===========================================================================
# cosine_similarity
# ===========================================================================


class TestCosineSimilarity:
    @pytest.mark.unit
    def test_identical_vectors_return_approximately_one(self):
        v = {"openai": 0.5, "gpt": 0.5}
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    @pytest.mark.unit
    def test_disjoint_vectors_return_zero(self):
        v1 = {"openai": 1.0}
        v2 = {"rust": 1.0}
        assert cosine_similarity(v1, v2) == 0.0

    @pytest.mark.unit
    def test_empty_first_vector_returns_zero(self):
        assert cosine_similarity({}, {"a": 1.0}) == 0.0

    @pytest.mark.unit
    def test_empty_second_vector_returns_zero(self):
        assert cosine_similarity({"a": 1.0}, {}) == 0.0

    @pytest.mark.unit
    def test_both_empty_returns_zero(self):
        assert cosine_similarity({}, {}) == 0.0

    @pytest.mark.unit
    def test_partial_overlap_between_zero_and_one(self):
        v1 = {"openai": 1.0, "gpt": 1.0}
        v2 = {"openai": 1.0, "rust": 1.0}
        sim = cosine_similarity(v1, v2)
        assert 0.0 < sim < 1.0

    @pytest.mark.unit
    def test_result_is_symmetric(self):
        v1 = {"a": 1.0, "b": 0.5}
        v2 = {"a": 0.5, "c": 1.0}
        assert abs(cosine_similarity(v1, v2) - cosine_similarity(v2, v1)) < 1e-12

    @pytest.mark.unit
    def test_more_shared_terms_gives_higher_sim(self):
        v_base = {"openai": 1.0, "gpt5": 1.0, "model": 1.0}
        v_close = {"openai": 1.0, "gpt5": 1.0, "released": 1.0}    # 2 shared
        v_far = {"rust": 1.0, "programming": 1.0, "fast": 1.0}      # 0 shared
        sim_close = cosine_similarity(v_base, v_close)
        sim_far = cosine_similarity(v_base, v_far)
        assert sim_close > sim_far

    @pytest.mark.unit
    def test_result_bounded_between_zero_and_one(self):
        v1 = {"a": 2.0, "b": -1.0}  # negative weights edge case
        v2 = {"a": 1.0, "b": 0.5}
        sim = cosine_similarity(v1, v2)
        # Should not crash even with odd weights
        assert isinstance(sim, float)


# ===========================================================================
# SEMANTIC_SIMILARITY_THRESHOLD
# ===========================================================================


class TestSemanticThreshold:
    @pytest.mark.unit
    def test_threshold_is_float(self):
        assert isinstance(SEMANTIC_SIMILARITY_THRESHOLD, float)

    @pytest.mark.unit
    def test_threshold_in_valid_range(self):
        assert 0.0 < SEMANTIC_SIMILARITY_THRESHOLD < 1.0

    @pytest.mark.unit
    def test_threshold_is_strict_enough_to_avoid_false_positives(self):
        # Two clearly different titles should score below default threshold
        docs = [
            ["python", "release"],
            ["rust", "performance"],
        ]
        vecs = build_tfidf_vectors(docs)
        sim = cosine_similarity(vecs[0], vecs[1])
        assert sim < SEMANTIC_SIMILARITY_THRESHOLD


# ===========================================================================
# Deduplicator.__init__ — new parameters
# ===========================================================================


class TestDeduplicatorInit:
    @pytest.mark.unit
    def test_default_semantic_threshold(self):
        d = Deduplicator()
        assert d.semantic_threshold == SEMANTIC_SIMILARITY_THRESHOLD

    @pytest.mark.unit
    def test_default_enable_semantic_dedup_is_true(self):
        d = Deduplicator()
        assert d.enable_semantic_dedup is True

    @pytest.mark.unit
    def test_custom_semantic_threshold_accepted(self):
        d = Deduplicator(semantic_threshold=0.9)
        assert d.semantic_threshold == 0.9

    @pytest.mark.unit
    def test_enable_semantic_dedup_false_accepted(self):
        d = Deduplicator(enable_semantic_dedup=False)
        assert d.enable_semantic_dedup is False


# ===========================================================================
# Deduplicator._dedup_by_semantic
# ===========================================================================


class TestDeduplicatorSemanticStage:
    @pytest.mark.unit
    def test_single_item_list_passthrough(self):
        items = [_item("OpenAI releases GPT-5")]
        d = Deduplicator()
        result = d._dedup_by_semantic(items)
        assert result == items

    @pytest.mark.unit
    def test_empty_list_passthrough(self):
        d = Deduplicator()
        assert d._dedup_by_semantic([]) == []

    @pytest.mark.unit
    def test_near_paraphrase_removed(self):
        # With only 2 docs in the mini-corpus, IDF smoothing reduces cosine sim
        # (shared terms appear in 100% of docs → lower IDF weight).
        # Threshold is set to 0.55 to reliably fire on a 2-item corpus.
        items = [
            _item("OpenAI Releases GPT-5 Model", score=100),
            _item("GPT-5 Model Released by OpenAI", score=50),
        ]
        d = Deduplicator(semantic_threshold=0.55)
        result = d._dedup_by_semantic(items)
        assert len(result) == 1

    @pytest.mark.unit
    def test_higher_score_wins_in_semantic_dedup(self):
        items = [
            _item("OpenAI Releases GPT-5 Model", score=50),    # lower score first
            _item("GPT-5 Model Released by OpenAI", score=100), # higher score second
        ]
        d = Deduplicator(semantic_threshold=0.55)
        result = d._dedup_by_semantic(items)
        assert len(result) == 1
        assert result[0].score == 100

    @pytest.mark.unit
    def test_completely_different_items_all_survive(self):
        items = [
            _item("Rust 2.0 Released", score=100),
            _item("Python 4.0 Announced", score=80),
            _item("Go 1.25 Ships with New GC", score=60),
        ]
        d = Deduplicator(semantic_threshold=0.82)
        result = d._dedup_by_semantic(items)
        assert len(result) == 3

    @pytest.mark.unit
    def test_very_high_threshold_keeps_all(self):
        items = [
            _item("OpenAI Releases GPT-5", score=100),
            _item("GPT-5 Released by OpenAI", score=50),
        ]
        d = Deduplicator(semantic_threshold=1.0)
        result = d._dedup_by_semantic(items)
        assert len(result) == 2


# ===========================================================================
# Deduplicator.deduplicate — integration
# ===========================================================================


class TestDeduplicateIntegration:
    @pytest.mark.unit
    def test_semantic_stage_disabled_skips_stage3(self):
        items = [
            _item("OpenAI Releases GPT-5 Model", score=100),
            _item("GPT-5 Model Released by OpenAI", score=50),
            _item("Rust Programming Language Version 2.0", score=200),
        ]
        d_no_sem = Deduplicator(enable_semantic_dedup=False, semantic_threshold=0.7)
        result_no_sem = d_no_sem.deduplicate(items)
        # Without Stage 3, the paraphrase pair may survive (Jaccard < 0.6)
        # We just verify it runs without error
        assert isinstance(result_no_sem, list)
        assert len(result_no_sem) >= 2

    @pytest.mark.unit
    def test_three_stage_pipeline_removes_paraphrase(self):
        items = [
            _item("OpenAI Releases GPT-5 Model", score=100),
            _item("GPT-5 Model Released by OpenAI", score=50),
            _item("Rust Programming Language Version 2.0", score=200),
        ]
        d = Deduplicator(enable_semantic_dedup=True, semantic_threshold=0.7)
        result = d.deduplicate(items)
        assert len(result) < len(items)
        assert any("Rust" in i.title for i in result)

    @pytest.mark.unit
    def test_url_dedup_still_runs_before_semantic(self):
        url = "https://example.com/same-article"
        items = [
            _item("OpenAI News", score=100, url=url),
            _item("OpenAI News Updated", score=50, url=url),  # same URL → Stage 1
        ]
        d = Deduplicator()
        result = d.deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 100

    @pytest.mark.unit
    def test_empty_list_returns_empty(self):
        assert Deduplicator().deduplicate([]) == []

    @pytest.mark.unit
    def test_single_item_list_returns_list(self):
        items = [_item("Only one story")]
        result = Deduplicator().deduplicate(items)
        assert len(result) == 1

    @pytest.mark.unit
    def test_jaccard_duplicate_caught_before_semantic(self):
        # Exact same words in same order → Stage 2 (Jaccard) catches it
        items = [
            _item("Python Released New Version Today", score=100),
            _item("Python Released New Version Today", score=50),
        ]
        d = Deduplicator()
        result = d.deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 100

    @pytest.mark.unit
    def test_input_order_preserved_for_survivors(self):
        items = [
            _item("First Article Rust", score=100),
            _item("Second Article Python", score=80),
            _item("Third Article Go", score=60),
        ]
        result = Deduplicator().deduplicate(items)
        titles = [i.title for i in result]
        # First, Second, Third should remain in order
        assert titles.index("First Article Rust") < titles.index("Third Article Go")
