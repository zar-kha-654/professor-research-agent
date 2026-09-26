"""
Discover faculty listing pages and individual professor profile pages
starting from a supplied institutional URL, within safe crawl limits.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from utils.config import (
    MAX_CRAWL_DEPTH,
    MAX_PAGES_HARD_CAP,
    REQUEST_DELAY_SECONDS,
)
from tools.web_scraper import fetch_page, same_institutional_domain

# Keywords that suggest a link points to a faculty/people listing page.
LISTING_KEYWORDS = [
    "faculty", "people", "staff", "directory", "professors",
    "department", "team", "members", "academics",
]

# Keywords that suggest a link is an individual profile page.
PROFILE_KEYWORDS = [
    "profile", "faculty/", "people/", "staff/", "bio", "~",
]

# Patterns that are almost never faculty content and waste crawl budget.
EXCLUDE_PATTERNS = [
    "login", "admissions", "donate", "giving", "apply", "calendar",
    "news", "events", "privacy", "sitemap", ".pdf", ".jpg", ".png",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com/company",
    "youtube.com",
]


@dataclass
class DiscoveryResult:
    base_url: str
    reachable: bool
    title: str = ""
    pages_discovered: int = 0
    faculty_listing_pages: list[str] = field(default_factory=list)
    profile_pages: list[str] = field(default_factory=list)
    all_links_seen: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None


def _looks_like(url: str, text: str, keywords: list[str]) -> bool:
    haystack = f"{url.lower()} {text.lower()}"
    return any(k in haystack for k in keywords)


def _is_excluded(url: str) -> bool:
    lower = url.lower()
    return any(p in lower for p in EXCLUDE_PATTERNS)


def discover_faculty_pages(
    base_url: str,
    max_pages: int = 40,
    max_depth: int = MAX_CRAWL_DEPTH,
    progress_callback=None,
) -> DiscoveryResult:
    """Breadth-first crawl of the institutional site, limited by max_pages/depth.

    Prioritizes: (1) the supplied site, (2) faculty listing pages,
    (3) profile pages linked from those pages, (4) other official subpages.
    """
    max_pages = min(max_pages, MAX_PAGES_HARD_CAP)

    root_result = fetch_page(base_url)

if not root_result.success:
    return DiscoveryResult(
        base_url=base_url,
        reachable=False,
        error=(
            f"{root_result.error} "
            "This site does not allow automated access from the research agent. "
            "Please provide a directly accessible faculty directory or profile URL."
        ),
    )

    result = DiscoveryResult(
        base_url=base_url, reachable=True, title=root_result.title, pages_discovered=1
    )
    result.all_links_seen.extend(root_result.links)

    visited: set[str] = {base_url}
    # Queue holds (url, depth). Seed with links found on the root page,
    # prioritizing anything that already looks like a listing page.
    candidates = [
        (u, 1) for _, u in root_result.links
        if same_institutional_domain(base_url, u) and not _is_excluded(u)
    ]

    def _priority(item: tuple[str, int]) -> int:
        u, _ = item
        if _looks_like(u, "", LISTING_KEYWORDS):
            return 0
        if _looks_like(u, "", PROFILE_KEYWORDS):
            return 1
        return 2

    candidates.sort(key=_priority)
    queue = candidates

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        if progress_callback:
            progress_callback(len(visited), max_pages, url)

        page = fetch_page(url)
        result.pages_discovered += 1
        time.sleep(REQUEST_DELAY_SECONDS)

        if not page.success:
            continue

        is_listing = _looks_like(url, page.title, LISTING_KEYWORDS)
        is_profile = _looks_like(url, page.title, PROFILE_KEYWORDS)

        if is_listing:
            result.faculty_listing_pages.append(url)
        elif is_profile:
            result.profile_pages.append(url)

        result.all_links_seen.extend(page.links)

        if depth < max_depth:
            new_links = [
                (u, depth + 1) for _, u in page.links
                if u not in visited and same_institutional_domain(base_url, u)
                and not _is_excluded(u)
            ]
            new_links.sort(key=_priority)
            queue.extend(new_links)
            queue.sort(key=_priority)

    return result


def guess_profile_links_from_listing(listing_url: str) -> list[tuple[str, str]]:
    """Fetch one listing page and return (name_text, profile_url) candidates."""
    page = fetch_page(listing_url)
    if not page.success:
        return []
    candidates = []
    for text, url in page.links:
        if not same_institutional_domain(listing_url, url):
            continue
        if _is_excluded(url):
            continue
        # A plausible "name" link: 2-4 capitalized words, or clearly under
        # a profile-ish path.
        looks_name = bool(re.match(r"^([A-Z][a-zA-Z.'-]+\s*){2,4}$", text.strip()))
        looks_profile_path = _looks_like(url, "", PROFILE_KEYWORDS)
        if looks_name or looks_profile_path:
            candidates.append((text.strip(), url))
    return candidates
