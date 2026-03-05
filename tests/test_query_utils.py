"""Tests for query classification and semantic expansion."""

from __future__ import annotations

import pytest

from knowlin_mcp.query_utils import (
    QueryIntent,
    classify_query,
    expand_query,
    get_source_weights,
)


class TestClassifyQuery:
    """Tests for query intent classification."""

    def test_debug_intent(self):
        assert classify_query("error in authentication module") == QueryIntent.DEBUG
        assert classify_query("why does it crash on startup") == QueryIntent.DEBUG
        assert classify_query("traceback from the API handler") == QueryIntent.DEBUG

    def test_howto_intent(self):
        assert classify_query("how to configure the database") == QueryIntent.HOWTO
        assert classify_query("how do I set up logging") == QueryIntent.HOWTO
        assert classify_query("implement caching layer") == QueryIntent.HOWTO

    def test_recall_intent(self):
        assert classify_query("when did we decide on that approach") == QueryIntent.RECALL
        assert classify_query("what was the decision about the API") == QueryIntent.RECALL
        assert classify_query("remember that time we chose Redis") == QueryIntent.RECALL

    def test_explore_intent_default(self):
        assert classify_query("authentication patterns") == QueryIntent.EXPLORE
        assert classify_query("database schema") == QueryIntent.EXPLORE
        assert classify_query("project architecture") == QueryIntent.EXPLORE

    def test_empty_query(self):
        assert classify_query("") == QueryIntent.EXPLORE

    def test_strongest_signal_wins(self):
        # "error" is DEBUG, but multiple HOWTO signals should win
        result = classify_query("how to configure and set up and install error handler")
        assert result == QueryIntent.HOWTO


class TestExpandQuery:
    """Tests for query synonym expansion."""

    def test_expands_known_terms(self):
        result = expand_query("fix the auth bug")
        assert "auth" in result.lower() or "authentication" in result.lower()
        assert len(result) > len("fix the auth bug")

    def test_no_expansion_for_unknown_terms(self):
        query = "something completely unrelated"
        assert expand_query(query) == query

    def test_limits_expansions(self):
        result = expand_query("auth")
        # Should not add ALL synonyms, max 3
        added_terms = result.replace("auth", "").strip().split()
        assert len(added_terms) <= 3

    def test_does_not_duplicate_existing_terms(self):
        result = expand_query("authentication and authorization")
        # "authentication" and "authorization" are synonyms of "auth" but already present
        words = result.lower().split()
        # Count occurrences - should not duplicate
        for word in ["authentication", "authorization"]:
            assert words.count(word) <= 1

    def test_db_expansion(self):
        result = expand_query("db migration issues")
        assert "database" in result.lower() or "sql" in result.lower()


class TestGetSourceWeights:
    """Tests for intent-based source weighting."""

    def test_debug_weights_favor_sessions(self):
        weights = get_source_weights(QueryIntent.DEBUG)
        assert weights["sessions"] > weights["kb"]
        assert weights["sessions"] > weights["docs"]

    def test_howto_weights_favor_docs(self):
        weights = get_source_weights(QueryIntent.HOWTO)
        assert weights["docs"] > weights["sessions"]

    def test_recall_weights_favor_sessions(self):
        weights = get_source_weights(QueryIntent.RECALL)
        assert weights["sessions"] > weights["kb"]
        assert weights["sessions"] > weights["docs"]

    def test_explore_weights_equal(self):
        weights = get_source_weights(QueryIntent.EXPLORE)
        assert weights["kb"] == weights["sessions"] == weights["docs"]

    def test_all_weights_positive(self):
        for intent in QueryIntent:
            weights = get_source_weights(intent)
            assert all(w > 0 for w in weights.values())

    def test_all_sources_present(self):
        for intent in QueryIntent:
            weights = get_source_weights(intent)
            assert "kb" in weights
            assert "sessions" in weights
            assert "docs" in weights
