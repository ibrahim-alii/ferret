"""Pytest fixtures for the eval suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def paper_id() -> str:
    return "2401.00001"


@pytest.fixture
def qrels_fixture() -> dict:
    """Pre-generated qrels for testing without LLM calls."""
    return {
        "What is the main contribution?": {"chunk_001": 1, "chunk_002": 1},
        "What dataset is used?": {"chunk_003": 1},
        "What are the results?": {"chunk_004": 1, "chunk_005": 1},
    }
