"""
End-to-end pipeline orchestration.

This module is what app.py actually drives. It wires together:
  website discovery -> page fetch -> RAG indexing -> per-professor
  extraction -> verification -> spreadsheet writing

It is deliberately plain Python (not itself an LLM call) so progress can be
reported step by step and errors on one professor never take down the rest
of the batch, per the "never crash the app" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agents.extractor import extract_professor_fields
from agents.verifier import verify_extraction
from tools.rag import SessionVectorStore
from tools.web_scraper import fetch_page
from tools.website_discovery import DiscoveryResult, guess_profile_links_from_listing
from utils.config import NOT_AVAILABLE
from utils.helpers import normalize_name


@dataclass
class ProfessorRecord:
    full_name: str
    source_url: str
    verified_fields: dict = field(default_factory=dict)
    overall_confidence: str = "Low"
    notes: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    professors: list[ProfessorRecord] = field(default_factory=list)
    pages_analyzed: int = 0
    errors: list[str] = field(default_factory=list)


def build_candidate_profile_list(discovery: DiscoveryResult, target_count: int) -> list[tuple[str, str]]:
    """Turn discovery results into an ordered (name_hint, url) candidate list.

    Prefers explicit profile pages; falls back to guessing name links from
    listing pages if not enough profile pages were found directly.
    """
    candidates: list[tuple[str, str]] = [("", u) for u in discovery.profile_pages]

    if len(candidates) < target_count:
        for listing_url in discovery.faculty_listing_pages:
            if len(candidates) >= target_count * 2:  # gather a bit of headroom
                break
            candidates.extend(guess_profile_links_from_listing(listing_url))

    # De-duplicate by URL while preserving order.
    seen = set()
    unique = []
    for name_hint, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique.append((name_hint, url))
    return unique


def index_pages_for_rag(urls: list[str], progress_callback=None) -> tuple[SessionVectorStore, int]:
    """Fetch and index a batch of pages into a fresh session vector store."""
    store = SessionVectorStore()
    pages_ok = 0
    for i, url in enumerate(urls):
        if progress_callback:
            progress_callback(i + 1, len(urls), url)
        page = fetch_page(url)
        if page.success and page.clean_text:
            store.add_page(page.clean_text, source_url=url, source_title=page.title)
            pages_ok += 1
    return store, pages_ok


def run_extraction_for_professor(
    name_hint: str,
    profile_url: str,
    store: SessionVectorStore,
    requested_fields: list[str],
) -> ProfessorRecord:
    """Extract + verify one professor's data. Never raises."""
    errors: list[str] = []
    context_text, best_source = store.context_for(name_hint or "professor", requested_fields)
    source_url = best_source or profile_url

    extraction = extract_professor_fields(
        professor_name=name_hint, source_text=context_text,
        source_url=source_url, requested_fields=requested_fields,
    )
    if extraction.error:
        errors.append(extraction.error)

    verification = verify_extraction(
        professor_name=extraction.full_name or name_hint,
        source_text=context_text, source_url=source_url,
        extracted_fields=extraction.fields,
    )
    if verification.error:
        errors.append(verification.error)

    return ProfessorRecord(
        full_name=extraction.full_name or name_hint or "Unknown",
        source_url=source_url,
        verified_fields=verification.verified_fields or {
            f: {"value": NOT_AVAILABLE, "status": "unverified", "confidence": 0.0}
            for f in requested_fields
        },
        overall_confidence=verification.overall_confidence,
        notes=verification.notes,
        errors=errors,
    )


def run_full_pipeline(
    discovery: DiscoveryResult,
    requested_fields: list[str],
    target_count: int,
    progress_callback=None,
) -> PipelineResult:
    """Run discovery-to-verification for up to target_count professors.

    progress_callback(stage: str, current: int, total: int, detail: str)
    is called throughout so the UI can render a live checklist.
    """
    result = PipelineResult()

    candidates = build_candidate_profile_list(discovery, target_count)[:max(target_count * 2, target_count)]
    if not candidates:
        result.errors.append(
            "No professor profile pages could be discovered. Try a more "
            "specific faculty directory URL."
        )
        return result

    urls_to_index = [u for _, u in candidates]
    if progress_callback:
        progress_callback("indexing", 0, len(urls_to_index), "")

    store, pages_ok = index_pages_for_rag(
        urls_to_index,
        progress_callback=lambda i, n, u: progress_callback and progress_callback("indexing", i, n, u),
    )
    result.pages_analyzed = pages_ok

    seen_names: set[str] = set()
    completed = 0
    for name_hint, url in candidates:
        if completed >= target_count:
            break
        if progress_callback:
            progress_callback("professor", completed + 1, target_count, name_hint or url)

        record = run_extraction_for_professor(name_hint, url, store, requested_fields)

        norm = normalize_name(record.full_name)
        if norm and norm in seen_names:
            continue  # skip obvious duplicate discovered via two different links
        seen_names.add(norm)

        result.professors.append(record)
        completed += 1

    return result
