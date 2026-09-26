"""Validation helpers: URL checks and structured-output schema checks."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from utils.config import NOT_AVAILABLE

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def is_valid_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:  # noqa: BLE001
        return False


def is_plausible_email(value: str) -> bool:
    if not value or value == NOT_AVAILABLE:
        return True  # N/A is always acceptable; absence isn't an error
    return bool(EMAIL_RE.match(value.strip()))


def validate_extraction_payload(payload: dict, requested_fields: list[str]) -> tuple[bool, list[str]]:
    """Validate the shape of an extractor-agent JSON payload.

    Returns (is_valid, list_of_problems). Never raises.
    """
    problems: list[str] = []

    if "fields" not in payload or not isinstance(payload["fields"], dict):
        problems.append("Missing or malformed 'fields' object.")
        return False, problems

    fields = payload["fields"]
    for requested in requested_fields:
        if requested not in fields:
            problems.append(f"Field '{requested}' missing from extraction output.")
            continue
        entry = fields[requested]
        if not isinstance(entry, dict) or "value" not in entry:
            problems.append(f"Field '{requested}' has malformed structure.")
            continue
        conf = entry.get("confidence", 0)
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            problems.append(f"Field '{requested}' has an invalid confidence value.")

    email_entry = fields.get("Email")
    if email_entry and isinstance(email_entry, dict):
        value = email_entry.get("value", "")
        if not is_plausible_email(value):
            problems.append(f"Email value '{value}' does not look like a valid email.")

    return len(problems) == 0, problems


def sanitize_for_display(value: str, max_len: int = 500) -> str:
    """Trim overly long extracted text before it's shown/stored in a cell."""
    if not isinstance(value, str):
        return str(value)
    value = value.strip()
    if len(value) > max_len:
        return value[:max_len].rsplit(" ", 1)[0] + "…"
    return value
