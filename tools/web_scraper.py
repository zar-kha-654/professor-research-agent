"""
Responsible web scraping utilities.

Handles fetching pages with timeouts/retries, robots.txt checks, and
converting raw HTML into clean text suitable for chunking/embedding.
"""
from __future__ import annotations

import urllib.robotparser as robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from utils.helpers import with_retry

_HEADERS = {"User-Agent": USER_AGENT}


@dataclass
class FetchResult:
    url: str
    success: bool
    status_code: int | None = None
    title: str = ""
    clean_text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (text, url)
    error: str | None = None


class RobotsCache:
    """Caches robots.txt parsers per domain so we only fetch them once."""

    def __init__(self) -> None:
        self._parsers: dict[str, robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(domain, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, default to allowing access
                # to the institutional site itself (not a bypass of any
                # access control — just tolerance of a missing file).
                self._parsers[domain] = None  # type: ignore
                return True
            self._parsers[domain] = rp
        parser = self._parsers[domain]
        if parser is None:
            return True
        try:
            return parser.can_fetch(USER_AGENT, url)
        except Exception:
            return True


_robots_cache = RobotsCache()


def is_reachable(url: str) -> tuple[bool, int | None, str | None]:
    """Quick HEAD/GET check used by the 'Analyze Website' button."""
    try:
        resp = with_retry(
            lambda: requests.get(
                url, headers=_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True
            ),
            max_retries=MAX_RETRIES,
        )
        return resp.status_code < 400, resp.status_code, None
    except requests.exceptions.Timeout:
        return False, None, "The request timed out."
    except requests.exceptions.ConnectionError:
        return False, None, "Could not connect to the website."
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def fetch_page(url: str, respect_robots: bool = True) -> FetchResult:
    """Fetch a single page and return clean text + discovered links.

    Never raises: all failure modes are captured in FetchResult.error so
    the calling agent/UI can show a friendly message instead of crashing.
    """
    if respect_robots and not _robots_cache.can_fetch(url):
        return FetchResult(url=url, success=False, error="Blocked by robots.txt")

    try:
        resp = with_retry(
            lambda: requests.get(
                url, headers=_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True
            ),
            max_retries=MAX_RETRIES,
        )
    except requests.exceptions.Timeout:
        return FetchResult(url=url, success=False, error="Request timed out.")
    except requests.exceptions.ConnectionError:
        return FetchResult(url=url, success=False, error="Connection failed.")
    except Exception as exc:  # noqa: BLE001
        return FetchResult(url=url, success=False, error=str(exc))

    if resp.status_code == 403:
        return FetchResult(
            url=url, success=False, status_code=403,
            error="The website blocked automated access (403 Forbidden).",
        )
    if resp.status_code >= 400:
        return FetchResult(
            url=url, success=False, status_code=resp.status_code,
            error=f"The website returned HTTP {resp.status_code}.",
        )

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return FetchResult(
            url=url, success=False, status_code=resp.status_code,
            error=f"Unsupported content type: {content_type or 'unknown'}.",
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip script/style/nav/footer noise before extracting text.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator=" ", strip=True)

    links: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absolute = urljoin(url, href)
        link_text = a.get_text(strip=True)
        links.append((link_text, absolute))

    # Detect a likely JavaScript-heavy page that rendered almost no text —
    # a common failure mode for SPA faculty directories.
    if len(text) < 200 and len(links) < 3:
        return FetchResult(
            url=url, success=False, status_code=resp.status_code, title=title,
            error=(
                "This page appears to require JavaScript to render its content "
                "and could not be read with a static fetch."
            ),
        )

    return FetchResult(
        url=url, success=True, status_code=resp.status_code,
        title=title, clean_text=text, links=links,
    )


def same_institutional_domain(base_url: str, candidate_url: str) -> bool:
    """Keep crawling limited to the institutional domain (and subdomains)."""
    base_domain = urlparse(base_url).netloc.lower().split(":")[0]
    cand_domain = urlparse(candidate_url).netloc.lower().split(":")[0]
    base_root = ".".join(base_domain.split(".")[-2:]) if "." in base_domain else base_domain
    cand_root = ".".join(cand_domain.split(".")[-2:]) if "." in cand_domain else cand_domain
    return base_root == cand_root
