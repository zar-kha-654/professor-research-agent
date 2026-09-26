"""
Spreadsheet reading, column mapping, and writing.

Handles the three upload cases described in the spec:
  A) empty template (headers only)
  B) spreadsheet with professor names, other fields blank
  C) partially completed spreadsheet (never overwritten unless the user opts in)
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from utils.config import NOT_AVAILABLE
from utils.helpers import normalize_name


@dataclass
class LoadedSpreadsheet:
    dataframe: "pd.DataFrame"
    original_columns: list[str]
    row_count: int
    file_type: str  # "xlsx" or "csv"
    error: str | None = None


def load_spreadsheet(uploaded_file) -> LoadedSpreadsheet:
    """Read an uploaded .xlsx or .csv file into a DataFrame.

    Never raises for malformed files — returns LoadedSpreadsheet.error instead.
    """
    name = getattr(uploaded_file, "name", "uploaded")
    is_csv = name.lower().endswith(".csv")
    try:
        if is_csv:
            df = pd.read_csv(uploaded_file)
            file_type = "csv"
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            file_type = "xlsx"
    except pd.errors.EmptyDataError:
        return LoadedSpreadsheet(pd.DataFrame(), [], 0, "csv" if is_csv else "xlsx",
                                  error="The uploaded file is empty.")
    except Exception as exc:  # noqa: BLE001
        return LoadedSpreadsheet(pd.DataFrame(), [], 0, "csv" if is_csv else "xlsx",
                                  error=f"Could not read the spreadsheet: {exc}")

    df = df.fillna("")
    return LoadedSpreadsheet(
        dataframe=df,
        original_columns=list(df.columns),
        row_count=len(df),
        file_type=file_type,
    )


def suggest_column_mapping(existing_columns: list[str], requested_fields: list[str]) -> dict[str, str | None]:
    """Cheap heuristic auto-mapping from requested field -> existing column name.

    This runs before any LLM call so we don't spend API budget on something
    a simple keyword match can usually solve. Returns None for fields with
    no confident match (the user maps those manually in the UI, or they get
    added as new columns).
    """
    mapping: dict[str, str | None] = {}
    lowered_existing = {c: c.lower().strip() for c in existing_columns}

    synonyms = {
        "full name": ["name", "professor", "full name", "faculty"],
        "email": ["email", "e-mail", "email address"],
        "department": ["department", "dept"],
        "research interests": ["research interest", "research area", "interests"],
        "profile url": ["profile", "url", "link", "profile url", "webpage"],
        "phone": ["phone", "telephone", "contact"],
        "office location": ["office", "room", "location"],
        "academic title": ["title", "rank", "position"],
    }

    for field_name in requested_fields:
        key = field_name.lower().strip()
        match = None
        for col, low in lowered_existing.items():
            if low == key:
                match = col
                break
        if not match:
            candidates = synonyms.get(key, [key])
            for col, low in lowered_existing.items():
                if any(cand in low for cand in candidates):
                    match = col
                    break
        mapping[field_name] = match
    return mapping


def ensure_columns(df: "pd.DataFrame", fields: list[str], mapping: dict[str, str]) -> "pd.DataFrame":
    """Add any missing spreadsheet columns for requested fields.

    `mapping` is field_name -> target_column_name (already resolved by the
    UI, whether auto-mapped, manually chosen, or newly created).
    """
    result = df.copy()
    for target_col in mapping.values():
        if target_col not in result.columns:
            result[target_col] = ""
    return result


def find_professor_row(df: "pd.DataFrame", name_column: str | None, full_name: str) -> int | None:
    """Find an existing row for a professor by fuzzy name match. Returns index or None."""
    if name_column is None or name_column not in df.columns:
        return None
    target = normalize_name(full_name)
    for idx, val in df[name_column].items():
        if val and normalize_name(str(val)) == target:
            return idx
    return None


def write_professor_data(
    df: "pd.DataFrame",
    row_idx: int | None,
    field_to_column: dict[str, str],
    field_values: dict[str, str],
    allow_overwrite: bool,
) -> tuple["pd.DataFrame", int]:
    """Insert/update one professor's data into the DataFrame.

    Returns (updated_df, row_index_used). If row_idx is None, a new row is
    appended. Existing non-empty cells are preserved unless allow_overwrite.
    """
    result = df.copy()
    if row_idx is None:
        new_row = {col: "" for col in result.columns}
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        row_idx = result.index[-1]

    for field_name, column in field_to_column.items():
        if column not in result.columns:
            result[column] = ""
        value = field_values.get(field_name, NOT_AVAILABLE)
        current = result.at[row_idx, column]
        is_blank = current is None or str(current).strip() == ""
        if is_blank or allow_overwrite:
            result.at[row_idx, column] = value

    return result, row_idx


def detect_duplicates(df: "pd.DataFrame", name_column: str | None,
                       email_column: str | None, url_column: str | None) -> list[list[int]]:
    """Group row indices that appear to describe the same professor."""
    groups: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        keys = []
        if name_column and str(row.get(name_column, "")).strip():
            keys.append(("name", normalize_name(str(row[name_column]))))
        if email_column and str(row.get(email_column, "")).strip():
            keys.append(("email", str(row[email_column]).strip().lower()))
        if url_column and str(row.get(url_column, "")).strip():
            keys.append(("url", str(row[url_column]).strip().lower()))
        for _, key in keys:
            groups.setdefault(key, []).append(idx)
    return [sorted(set(v)) for v in groups.values() if len(v) > 1]


def to_excel_bytes(df: "pd.DataFrame", hyperlink_columns: list[str] | None = None) -> bytes:
    """Serialize a DataFrame to .xlsx bytes, with clickable hyperlinks for
    any column listed in hyperlink_columns (e.g. Source URL, Profile URL)."""
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    if hyperlink_columns:
        wb = load_workbook(buffer)
        ws = wb.active
        header = [cell.value for cell in ws[1]]
        for col_name in hyperlink_columns:
            if col_name not in header:
                continue
            col_idx = header.index(col_name) + 1
            col_letter = get_column_letter(col_idx)
            for row in range(2, ws.max_row + 1):
                cell = ws[f"{col_letter}{row}"]
                url = cell.value
                if url and isinstance(url, str) and url.startswith("http"):
                    cell.hyperlink = url
                    cell.font = Font(color="0563C1", underline="single")
        out = BytesIO()
        wb.save(out)
        return out.getvalue()

    return buffer.getvalue()


def to_csv_bytes(df: "pd.DataFrame") -> bytes:
    return df.to_csv(index=False).encode("utf-8")
