"""
Central configuration for Professor Research Agent.

Keep every tunable constant here so the rest of the codebase never
hard-codes a model name, limit, or timeout.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
# Single place to change the Groq model used across the whole app.
MODEL_NAME: str = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_BASE: str = "https://api.groq.com/openai/v1"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 2000

# ---------------------------------------------------------------------------
# Crawling / scraping safety limits
# ---------------------------------------------------------------------------
MAX_PAGES_DEFAULT: int = 40
MAX_PAGES_HARD_CAP: int = 100
MAX_CRAWL_DEPTH: int = 2
REQUEST_TIMEOUT_SECONDS: int = 15
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE_SECONDS: float = 1.5
REQUEST_DELAY_SECONDS: float = 0.6  # politeness delay between requests
USER_AGENT: str = (
    "ProfessorResearchAgent/1.0 (+educational research tool; "
    "respects robots.txt; contact: set-your-contact-email)"
)

# ---------------------------------------------------------------------------
# Professor count limits
# ---------------------------------------------------------------------------
MIN_PROFESSORS: int = 1
MAX_PROFESSORS: int = 100
BATCH_SIZE: int = 5  # professors processed per LLM batch

# ---------------------------------------------------------------------------
# RAG / embedding configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 120
RAG_TOP_K: int = 6

# ---------------------------------------------------------------------------
# Standard field catalogue (checkbox list shown in the sidebar/main page)
# ---------------------------------------------------------------------------
STANDARD_FIELDS: list[str] = [
    "Full Name",
    "First Name",
    "Last Name",
    "Academic Title",
    "Department",
    "Faculty/School",
    "Email",
    "Phone",
    "Office Location",
    "Office Hours",
    "Profile URL",
    "Personal Website",
    "Google Scholar",
    "ORCID",
    "Research Interests",
    "Research Areas",
    "Biography",
    "Education",
    "Degrees",
    "Current Position",
    "Previous Positions",
    "Publications",
    "Research Projects",
    "Courses Taught",
    "Supervision Interests",
    "Awards",
    "Grants",
    "LinkedIn",
]

DEFAULT_SELECTED_FIELDS: list[str] = [
    "Full Name",
    "Email",
    "Department",
    "Research Interests",
    "Profile URL",
]

# ---------------------------------------------------------------------------
# Confidence bucketing
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH_THRESHOLD: float = 0.85
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.55

# Value used whenever information cannot be verified on the source page.
NOT_AVAILABLE: str = "N/A"


def get_groq_api_key() -> str | None:
    """Fetch the Groq API key from Streamlit secrets, falling back to env var.

    Returns None (never raises) so callers can show a friendly UI message
    instead of crashing when no key has been configured yet.
    """
    try:
        import streamlit as st  # local import: keeps this module importable
        # outside a Streamlit runtime (e.g. in unit tests).
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")
