"""
Agent 3 — Academic Data Verification Specialist.

Cross-checks extracted fields against the original source text before
anything is written to the spreadsheet, and assigns an overall
High/Medium/Low confidence label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from crewai import Agent

from utils.config import MODEL_NAME, NOT_AVAILABLE
from utils.helpers import confidence_bucket, extract_json_block
from utils.llm_client import chat_completion
from utils.prompts import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_TEMPLATE

VERIFIER_ROLE = "Academic Data Verification Specialist"
VERIFIER_GOAL = (
    "Verify extracted professor information against the original source "
    "page, catching conflicting or unsupported values before they reach "
    "the spreadsheet."
)
VERIFIER_BACKSTORY = (
    "You are a careful fact-checker. You compare every extracted value "
    "against the source text and flag anything that isn't directly "
    "supported, rather than letting it pass silently."
)


def build_verifier_agent() -> Agent:
    return Agent(
        role=VERIFIER_ROLE,
        goal=VERIFIER_GOAL,
        backstory=VERIFIER_BACKSTORY,
        allow_delegation=False,
        verbose=False,
        llm=MODEL_NAME,
    )


@dataclass
class VerificationOutcome:
    success: bool
    verified_fields: dict = field(default_factory=dict)
    overall_confidence: str = "Low"
    notes: str = ""
    error: str | None = None


def verify_extraction(
    professor_name: str,
    source_text: str,
    source_url: str,
    extracted_fields: dict,
) -> VerificationOutcome:
    """Ask the LLM to double-check extracted fields against the source text.

    Falls back to a deterministic, conservative verification (based purely
    on the extractor's own confidence numbers) if the verifier call fails,
    so the pipeline degrades gracefully instead of crashing.
    """
    if not source_text.strip():
        fallback = {
            f: {"value": v.get("value", NOT_AVAILABLE), "status": "unverified", "confidence": 0.0}
            for f, v in extracted_fields.items()
        }
        return VerificationOutcome(
            success=False, verified_fields=fallback, overall_confidence="Low",
            notes="No source text available to verify against.",
            error="No source text available.",
        )

    user_prompt = VERIFIER_USER_TEMPLATE.format(
        professor_name=professor_name,
        source_url=source_url,
        source_text=source_text[:8000],
        extracted_json=json.dumps(extracted_fields, indent=2),
    )

    try:
        raw = chat_completion(VERIFIER_SYSTEM_PROMPT, user_prompt)
        payload = extract_json_block(raw)
        verified = payload.get("verified_fields", {})
        overall = payload.get("overall_confidence", "Low")
        if overall not in ("High", "Medium", "Low"):
            overall = "Low"
        return VerificationOutcome(
            success=True,
            verified_fields=verified,
            overall_confidence=overall,
            notes=payload.get("notes", ""),
        )
    except Exception as exc:  # noqa: BLE001
        # Deterministic fallback: derive verification from the extractor's
        # own reported confidence scores rather than failing the whole row.
        verified = {}
        confidences = []
        for f, v in extracted_fields.items():
            conf = float(v.get("confidence", 0.0) or 0.0)
            confidences.append(conf)
            status = "verified" if conf >= 0.85 else ("unverified" if conf < 0.55 else "verified")
            verified[f] = {"value": v.get("value", NOT_AVAILABLE), "status": status, "confidence": conf}
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return VerificationOutcome(
            success=False,
            verified_fields=verified,
            overall_confidence=confidence_bucket(avg_conf),
            notes="LLM verification unavailable; used extractor confidence scores as a fallback.",
            error=f"Verification call failed: {exc}",
        )
