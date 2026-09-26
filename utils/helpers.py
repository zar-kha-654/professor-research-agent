"""Small shared helper functions used across tools and agents."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, TypeVar

from utils.config import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE_SECONDS,
)

T = TypeVar("T")


def with_retry(fn: Callable[[], T], max_retries: int = MAX_RETRIES,
                on_retry: Callable[[int, Exception], None] | None = None) -> T:
    """Call fn() with exponential backoff retry.

    Raises the last exception if all attempts fail so the caller can
    decide how to surface it to the UI.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to catch broadly here
            last_exc = exc
            if on_retry:
                on_retry(attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def extract_json_block(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from an LLM response.

    Handles the common case where the model wraps JSON in markdown fences
    despite being told not to, and falls back gracefully.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first {...} balanced block.
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                return json.loads(candidate)
    raise ValueError(f"Unbalanced JSON in model output: {text[:200]}")


def confidence_bucket(score: float) -> str:
    """Convert a numeric confidence (0-1) into a High/Medium/Low label."""
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def slugify_field_name(field: str) -> str:
    """Turn a human field label into a safe spreadsheet column key."""
    return re.sub(r"\s+", " ", field.strip())


def normalize_name(name: str) -> str:
    """Lowercase + strip titles/punctuation for duplicate-name comparison."""
    n = name.lower()
    n = re.sub(r"\b(dr|prof|professor|mr|ms|mrs)\.?\b", "", n)
    n = re.sub(r"[^a-z\s]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker used by the RAG pipeline."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
