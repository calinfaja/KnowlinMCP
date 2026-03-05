"""Query classification and semantic expansion for multi-source search.

Classifies queries by intent (debug, howto, recall, explore) and expands
them with synonyms to improve retrieval across different knowledge sources.
"""

from __future__ import annotations

from enum import Enum


class QueryIntent(Enum):
    """Classification of search query intent."""
    DEBUG = "debug"       # Error/bug investigation
    HOWTO = "howto"       # How to accomplish something
    RECALL = "recall"     # Recall a specific past event/decision
    EXPLORE = "explore"   # Open-ended exploration


# Signal words for each intent
_INTENT_SIGNALS = {
    QueryIntent.DEBUG: [
        "error", "bug", "crash", "fail", "broken", "exception", "traceback",
        "stack trace", "segfault", "panic", "undefined", "null", "nan",
        "timeout", "hang", "freeze", "memory leak", "race condition",
    ],
    QueryIntent.HOWTO: [
        "how to", "how do", "how can", "what's the best way", "implement",
        "configure", "setup", "set up", "install", "create", "build",
        "make", "add", "enable", "disable", "use ", "connect",
    ],
    QueryIntent.RECALL: [
        "when did", "where did", "who", "last time", "previously",
        "remember", "that time", "we decided", "we chose", "decision",
        "why did we", "what was the", "history",
    ],
}

# Synonym expansions to improve cross-source retrieval
_SYNONYMS = {
    "auth": ["authentication", "authorization", "login", "oauth", "jwt", "session"],
    "db": ["database", "sql", "query", "schema", "migration", "orm"],
    "api": ["endpoint", "rest", "graphql", "route", "handler", "request"],
    "test": ["testing", "unittest", "pytest", "spec", "assertion", "mock"],
    "deploy": ["deployment", "ci/cd", "pipeline", "release", "publish"],
    "perf": ["performance", "optimization", "latency", "throughput", "benchmark"],
    "config": ["configuration", "settings", "environment", "env", "dotenv"],
    "cache": ["caching", "redis", "memcached", "memoize", "lru"],
    "error": ["exception", "failure", "crash", "bug", "issue"],
    "async": ["asynchronous", "await", "promise", "concurrent", "parallel"],
}


def classify_query(query: str) -> QueryIntent:
    """Classify a search query by intent.

    Returns the most likely intent based on signal words.
    Defaults to EXPLORE for ambiguous queries.
    """
    text = query.lower()

    # Check each intent's signals
    scores = {}
    for intent, signals in _INTENT_SIGNALS.items():
        score = sum(1 for signal in signals if signal in text)
        if score > 0:
            scores[intent] = score

    if scores:
        return max(scores, key=scores.get)

    return QueryIntent.EXPLORE


def expand_query(query: str) -> str:
    """Expand query with synonyms to improve retrieval.

    Appends relevant synonym terms to the original query so that
    the embedding captures related concepts.
    """
    text = query.lower()
    expansions = []

    for term, synonyms in _SYNONYMS.items():
        if len(expansions) >= 3:
            break
        if term in text:
            # Add synonyms that aren't already in the query
            for syn in synonyms:
                if syn not in text:
                    expansions.append(syn)
                    if len(expansions) >= 3:
                        break

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def get_source_weights(intent: QueryIntent) -> dict[str, float]:
    """Get source weights based on query intent.

    Returns weights for each source (kb, sessions, docs) that
    adjust RRF scoring in multi-source search.
    """
    weights = {
        QueryIntent.DEBUG: {"kb": 1.5, "sessions": 2.0, "docs": 0.5},
        QueryIntent.HOWTO: {"kb": 1.5, "sessions": 0.8, "docs": 2.0},
        QueryIntent.RECALL: {"kb": 1.0, "sessions": 2.5, "docs": 0.3},
        QueryIntent.EXPLORE: {"kb": 1.0, "sessions": 1.0, "docs": 1.0},
    }
    return weights.get(intent, {"kb": 1.0, "sessions": 1.0, "docs": 1.0})
