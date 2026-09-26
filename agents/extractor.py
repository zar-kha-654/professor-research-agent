"""
Agent 2 — Academic Information Extraction Specialist.

Converts raw webpage text (retrieved via the RAG pipeline) into structured
JSON fields for one professor, using the Groq LLM. Never invents data:
returns "N/A" for anything unsupported by the source text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crewai import Agent

from utils.config import MODEL_NAME, NOT_AVAILABLE
from utils.helpers import extract_json_block
from utils.llm_client import chat_completion
from utils.prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_TEMPLATE
from tools.validation import validate_extraction_payload

EXTRACTOR_ROLE = "Academic Information Extraction Specialist"
EXTRACTOR_GOAL = (
    "Convert raw webpage content into structured professor information, "
    "matching it to requested fields and never inventing missing data."
)
EXTRACTOR_BACKSTORY = (
    "You are meticulous about only reporting facts that are explicitly "
    "present in the source text you are given. When a field is not "
    "present, you always return 'N/A' rather than guessing."
)


def build_extractor_agent() -> Agent:
    return Agent(
        role=EXTRACTOR_ROLE,
        goal=EXTRACTOR_GOAL,
        backstory=EXTRACTOR_BACKSTORY,
        allow_delegation=False,
        verbose=False,
        llm=MODEL_NAME,
    )


@dataclass
class ExtractionOutcome:
    success: bool
    full_name: str = ""
    fields: dict = field(default_factory=dict)  # field -> {value, confidence}
    source_url: str = ""
    error: str | None = None
    raw_problems: list[str] = field(default_factory=list)


def extract_professor_fields(
    professor_name: str,
    source_text: str,
    source_url: str,
    requested_fields: list[str],
) -> ExtractionOutcome:
    """Call the LLM to extract requested fields from retrieved context.

    Validates the JSON shape before returning it. On any failure, returns
    an ExtractionOutcome with success=False and every field set to N/A so
    the pipeline can continue without crashing.
    """
    if not source_text.strip():
        return ExtractionOutcome(
            success=False,
            full_name=professor_name,
            fields={f: {"value": NOT_AVAILABLE, "confidence": 0.0} for f in requested_fields},
            source_url=source_url,
            error="No relevant source text was retrieved for this professor.",
        )

    user_prompt = EXTRACTOR_USER_TEMPLATE.format(
        source_url=source_url,
        source_text=source_text[:8000],
        professor_name=professor_name or "(identify from text)",
        field_list="\n".join(f"- {f}" for f in requested_fields),
    )

    try:
        raw = chat_completion(EXTRACTOR_SYSTEM_PROMPT, user_prompt)
        payload = extract_json_block(raw)
    except Exception as exc:  # noqa: BLE001
        return ExtractionOutcome(
            success=False,
            full_name=professor_name,
            fields={f: {"value": NOT_AVAILABLE, "confidence": 0.0} for f in requested_fields},
            source_url=source_url,
            error=f"Extraction call failed: {exc}",
        )

    is_valid, problems = validate_extraction_payload(payload, requested_fields)
    fields_out = payload.get("fields", {})

    # Fill any still-missing requested fields with N/A so downstream code
    # can always rely on every requested key being present.
    for f in requested_fields:
        if f not in fields_out or not isinstance(fields_out.get(f), dict):
            fields_out[f] = {"value": NOT_AVAILABLE, "confidence": 0.0}

    return ExtractionOutcome(
        success=is_valid,
        full_name=payload.get("full_name", professor_name),
        fields=fields_out,
        source_url=payload.get("source_url", source_url),
        error=None if is_valid else "; ".join(problems),
        raw_problems=problems,
    )
