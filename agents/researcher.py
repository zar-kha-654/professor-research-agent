"""
Agent 1 — Academic Website Research Specialist.

Responsible for discovering faculty listing pages and professor profile
URLs on the supplied institutional website, within safe crawl limits.
The actual crawling is deterministic Python (tools/website_discovery.py);
this agent's CrewAI definition exists for orchestration/logging purposes
and for the optional LLM-assisted link classification step.
"""
from __future__ import annotations

from crewai import Agent

from tools.website_discovery import DiscoveryResult, discover_faculty_pages
from utils.config import MODEL_NAME

RESEARCHER_ROLE = "Academic Website Research Specialist"
RESEARCHER_GOAL = (
    "Discover faculty listing pages and individual professor profile pages "
    "on the supplied institutional website, respecting robots.txt and "
    "reasonable crawling limits, without wandering to unrelated sites."
)
RESEARCHER_BACKSTORY = (
    "You specialize in navigating university websites to locate official "
    "faculty directories and professor profile pages. You never guess at "
    "URLs and never crawl outside the institution's own domain unless "
    "explicitly permitted."
)


def build_researcher_agent() -> Agent:
    return Agent(
        role=RESEARCHER_ROLE,
        goal=RESEARCHER_GOAL,
        backstory=RESEARCHER_BACKSTORY,
        allow_delegation=False,
        verbose=False,
        llm=MODEL_NAME,
    )


def run_discovery(base_url: str, max_pages: int, progress_callback=None) -> DiscoveryResult:
    """Deterministic discovery step (Python, not an LLM call).

    Kept as a plain function — not every step needs to go through the LLM,
    per the architecture rule that Python should own crawling/parsing.
    """
    return discover_faculty_pages(base_url, max_pages=max_pages, progress_callback=progress_callback)
