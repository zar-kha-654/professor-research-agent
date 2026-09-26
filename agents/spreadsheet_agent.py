"""
Agent 4 — Academic Spreadsheet Data Manager.

Takes verified professor data and writes it into the correct row/column of
the working DataFrame, adding missing columns, avoiding duplicate rows,
and preparing the final downloadable file. This is pure Python (no LLM
calls needed) per the architecture rule that spreadsheet manipulation
belongs to deterministic code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crewai import Agent

from tools.spreadsheet import ensure_columns, find_professor_row, write_professor_data
from utils.config import MODEL_NAME, NOT_AVAILABLE

SPREADSHEET_ROLE = "Academic Spreadsheet Data Manager"
SPREADSHEET_GOAL = (
    "Match verified professor information to spreadsheet columns, insert it "
    "into the correct row, preserve existing data unless overwrite is "
    "explicitly allowed, and keep the spreadsheet free of accidental "
    "duplicates."
)
SPREADSHEET_BACKSTORY = (
    "You are precise about column mapping and never place information in "
    "the wrong field. You preserve the user's existing spreadsheet "
    "structure and formatting wherever possible."
)


def build_spreadsheet_agent() -> Agent:
    return Agent(
        role=SPREADSHEET_ROLE,
        goal=SPREADSHEET_GOAL,
        backstory=SPREADSHEET_BACKSTORY,
        allow_delegation=False,
        verbose=False,
        llm=MODEL_NAME,
    )


@dataclass
class WriteOutcome:
    dataframe: "pd.DataFrame"
    row_index: int
    was_new_row: bool
    fields_written: list[str] = field(default_factory=list)


def write_verified_professor(
    df: "pd.DataFrame",
    full_name: str,
    field_to_column: dict[str, str],
    verified_fields: dict,
    name_column: str | None,
    allow_overwrite: bool,
) -> WriteOutcome:
    """Write one professor's verified fields into the DataFrame."""
    working = ensure_columns(df, list(field_to_column.keys()), field_to_column)

    existing_row = find_professor_row(working, name_column, full_name)
    was_new_row = existing_row is None

    field_values = {
        f: v.get("value", NOT_AVAILABLE) for f, v in verified_fields.items()
    }
    if name_column and name_column in field_to_column.values():
        # Make sure the name itself gets written for brand-new rows.
        for f, col in field_to_column.items():
            if col == name_column and f not in field_values:
                field_values[f] = full_name

    updated_df, row_idx = write_professor_data(
        working, existing_row, field_to_column, field_values, allow_overwrite
    )

    return WriteOutcome(
        dataframe=updated_df,
        row_index=row_idx,
        was_new_row=was_new_row,
        fields_written=list(field_values.keys()),
    )
