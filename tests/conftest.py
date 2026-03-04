"""Shared fixtures for kln-knowledge-system tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_kb_dir(tmp_path):
    """Create temporary knowledge DB directory with project markers."""
    kb_dir = tmp_path / "test-project" / ".knowledge-db"
    kb_dir.mkdir(parents=True)
    (tmp_path / "test-project" / ".git").mkdir()
    return kb_dir


@pytest.fixture
def sample_entries():
    """Sample V3 entries for testing."""
    return [
        {
            "id": "entry-1",
            "title": "BLE Power Optimization",
            "summary": "Nordic nRF52 power optimization techniques",
            "tags": ["ble", "power", "embedded"],
            "found_date": "2024-12-01T00:00:00",
        },
        {
            "id": "entry-2",
            "title": "OAuth2 Implementation",
            "summary": "OAuth2 security patterns and best practices",
            "tags": ["oauth", "security", "auth"],
            "found_date": "2024-12-02T00:00:00",
        },
        {
            "id": "entry-3",
            "title": "Python Type Hints",
            "summary": "Using type hints for better code quality",
            "tags": ["python", "typing"],
            "found_date": "2024-12-03T00:00:00",
        },
    ]


@pytest.fixture
def kb_with_entries(temp_kb_dir, sample_entries):
    """Create KB directory with sample entries in JSONL."""
    jsonl_path = temp_kb_dir / "entries.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in sample_entries:
            f.write(json.dumps(entry) + "\n")
    return temp_kb_dir


@pytest.fixture
def project_root(temp_kb_dir):
    """Return the project root (parent of .knowledge-db)."""
    return temp_kb_dir.parent
