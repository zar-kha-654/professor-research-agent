"""
Professor Research Agent — Streamlit application entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from tools.spreadsheet import (
    detect_duplicates,
    load_spreadsheet,
    suggest_column_mapping,
    to_csv_bytes,
    to_excel_bytes,
)
from tools.validation import is_valid_url
from tools.web_scraper import is_reachable
from tools.website_discovery import discover_faculty_pages
from utils.config import (
    DEFAULT_SELECTED_FIELDS,
    MAX_PAGES_HARD_CAP,
    MAX_PROFESSORS,
    MIN_PROFESSORS,
    MODEL_NAME,
    NOT_AVAILABLE,
    STANDARD_FIELDS,
    get_groq_api_key,
)
from utils.helpers import extract_json_block
from utils.llm_client import GroqNotConfiguredError, chat_completion
from utils.pipeline import run_full_pipeline
from utils.prompts import MATCH_SYSTEM_PROMPT, MATCH_USER_TEMPLATE

st.set_page_config(page_title="Professor Research Agent", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "spreadsheet_df": None,
        "spreadsheet_columns": [],
        "spreadsheet_file_type": "xlsx",
        "discovery_result": None,
        "field_column_mapping": {},
        "pipeline_result": None,
        "results_df": None,
        "research_history": [],
        "stop_requested": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — Research Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Research Settings")

    api_key = get_groq_api_key()
    if api_key:
        st.success("Groq API: configured ✓")
    else:
        st.error("Groq API key not found")
        st.caption(
            "Add `GROQ_API_KEY` under Streamlit **Secrets** (or as an "
            "environment variable) before starting research."
        )
    st.caption(f"Model: `{MODEL_NAME}`")

    st.divider()
    professor_count = st.number_input(
        "How many professors do you want?",
        min_value=MIN_PROFESSORS, max_value=MAX_PROFESSORS, value=10, step=1,
    )
    max_pages = st.slider(
        "Maximum pages to crawl", min_value=10, max_value=MAX_PAGES_HARD_CAP, value=40, step=5,
    )

    st.divider()
    include_sources = st.checkbox("Include Sources", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    allow_overwrite = st.checkbox("Allow AI to overwrite existing data", value=False)

    st.divider()
    if st.session_state["research_history"]:
        st.subheader("Previous Research Sessions")
        for entry in reversed(st.session_state["research_history"]):
            st.caption(f"**{entry['site']}** — {entry['count']} professors — {entry['when']}")

    st.divider()
    st.caption(
        "Uploaded spreadsheets and research results are processed only for "
        "this session. Do not upload confidential or sensitive information."
    )

st.title("🎓 Professor Research Agent")
st.caption(
    "Upload a spreadsheet, point it at an institutional faculty website, and "
    "let a research agent fill in verified, source-grounded professor data."
)

# ---------------------------------------------------------------------------
# Step 1 — Upload Spreadsheet
# ---------------------------------------------------------------------------
st.header("1. Upload Spreadsheet")
uploaded_file = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])

if uploaded_file is not None:
    loaded = load_spreadsheet(uploaded_file)
    if loaded.error:
        st.error(loaded.error)
    else:
        st.session_state["spreadsheet_df"] = loaded.dataframe
        st.session_state["spreadsheet_columns"] = loaded.original_columns
        st.session_state["spreadsheet_file_type"] = loaded.file_type
        col1, col2, col3 = st.columns(3)
        col1.metric("File", uploaded_file.name)
        col2.metric("Rows", loaded.row_count)
        col3.metric("Columns", len(loaded.original_columns))
        st.write("Preview:")
        st.dataframe(loaded.dataframe.head(10), use_container_width=True)

df = st.session_state["spreadsheet_df"]
if df is None:
    st.info("Upload a spreadsheet to continue, or start with a blank template below.")
    if st.button("Start with a blank template"):
        df = pd.DataFrame(columns=["Professor", "Email", "Department", "Research Interests", "Profile URL"])
        st.session_state["spreadsheet_df"] = df
        st.session_state["spreadsheet_columns"] = list(df.columns)
        st.rerun()

# ---------------------------------------------------------------------------
# Step 2 — Institutional Website
# ---------------------------------------------------------------------------
st.header("2. Institutional Website")
site_url = st.text_input("Enter institutional faculty website", placeholder="https://example.edu/faculty")

if st.button("🔎 Analyze Website", disabled=not site_url):
    if not is_valid_url(site_url):
        st.error(
            "That doesn't look like a valid URL. Include the scheme, e.g. https://"
        )
    else:
        reachable, status_code, err = is_reachable(site_url)

        if not reachable and status_code != 403:
            st.error(
                f"The website could not be accessed ({err or status_code}). "
                "Please check the URL or try the faculty directory page directly."
            )
        else:
            if status_code == 403:
                st.warning(
                    "The website returned 403 Forbidden. "
                    "Trying the discovery step anyway..."
                )

            progress_bar = st.progress(
                0,
                text="Crawling institutional site…"
            )
            status_area = st.empty()

            def _cb(current, total, url):
                pct = min(current / max(total, 1), 1.0)
                progress_bar.progress(
                    pct,
                    text=f"Analyzed {current}/{total} pages"
                )
                status_area.caption(url)

            discovery = discover_faculty_pages(
                site_url,
                max_pages=max_pages,
                progress_callback=_cb,
            )

            st.session_state["discovery_result"] = discovery

            progress_bar.empty()
            status_area.empty()
discovery = st.session_state["discovery_result"]
if discovery is not None:
    if not discovery.reachable:
        st.error(f"The website could not be accessed: {discovery.error}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Website title", discovery.title or "—")
        col2.metric("Pages discovered", discovery.pages_discovered)
        col3.metric(
            "Faculty links found",
            len(discovery.faculty_listing_pages) + len(discovery.profile_pages),
        )
        if not discovery.faculty_listing_pages and not discovery.profile_pages:
            st.warning(
                "No faculty listing or profile pages were confidently identified. "
                "Try a more specific URL such as a direct faculty directory page."
            )
        else:
            with st.expander("Discovered pages"):
                st.write("**Faculty listing pages:**")
                for u in discovery.faculty_listing_pages:
                    st.caption(u)
                st.write("**Profile pages:**")
                for u in discovery.profile_pages:
                    st.caption(u)

# ---------------------------------------------------------------------------
# Step 3 — Select Information
# ---------------------------------------------------------------------------
st.header("3. Select Information")
selected_fields = st.multiselect(
    "Choose the fields you want collected",
    options=STANDARD_FIELDS,
    default=DEFAULT_SELECTED_FIELDS,
)

custom_fields_raw = st.text_area(
    "Custom Fields (one per line)",
    placeholder="e.g.\nMachine Learning research areas\nAccepting graduate students?\nLab name",
)
custom_fields = [line.strip() for line in custom_fields_raw.splitlines() if line.strip()]

all_requested_fields = selected_fields + custom_fields

# ---------------------------------------------------------------------------
# Step 3b — Column mapping
# ---------------------------------------------------------------------------
field_to_column: dict[str, str] = {}
if df is not None and all_requested_fields:
    st.subheader("Map Spreadsheet Columns")
    existing_cols = list(df.columns)
    suggested = suggest_column_mapping(existing_cols, all_requested_fields)

    st.caption("Automatic mapping shown below — adjust or add new columns as needed.")
    for f in all_requested_fields:
        options = ["＋ Add new column"] + existing_cols
        default_col = suggested.get(f)
        default_idx = options.index(default_col) if default_col in options else 0
        chosen = st.selectbox(
            f"'{f}' maps to:", options=options, index=default_idx, key=f"map_{f}",
        )
        if chosen == "＋ Add new column":
            field_to_column[f] = f  # new column named after the field itself
        else:
            field_to_column[f] = chosen

# ---------------------------------------------------------------------------
# Step 4 — Number of Professors (already collected in sidebar) + Start
# ---------------------------------------------------------------------------
st.header("4. Number of Professors")
st.write(f"Configured in the sidebar: **{professor_count}** professor(s)")

st.header("5. Start Research")
name_column_guess = None
for candidate in ["Professor", "Full Name", "Name"]:
    if candidate in field_to_column.values():
        name_column_guess = candidate
        break

start_disabled = (
    df is None
    or discovery is None
    or not discovery.reachable
    or not all_requested_fields
    or not api_key
)
if not api_key:
    st.warning("Add your Groq API key before starting research.")

if st.button("🚀 Start Professor Research", disabled=start_disabled, type="primary"):
    st.session_state["stop_requested"] = False
    stage_area = st.container()
    with stage_area:
        st.write("**Website Analysis** ✓")
        st.write("**Faculty Directory** ✓")
        prof_progress = st.progress(0, text="Discovering professors…")
        prof_status = st.empty()

    def _progress_cb(stage, current, total, detail):
        if st.session_state.get("stop_requested"):
            raise KeyboardInterrupt("Research stopped by user")
        if stage == "indexing":
            prof_progress.progress(
                min(current / max(total, 1), 1.0) * 0.4,
                text=f"Analyzing pages: {current}/{total}",
            )
            prof_status.caption(detail)
        elif stage == "professor":
            pct = 0.4 + 0.6 * min(current / max(total, 1), 1.0)
            prof_progress.progress(pct, text=f"Professor {current}/{total}")
            prof_status.caption(detail or "researching…")

    try:
        with st.spinner("Running research pipeline…"):
            pipeline_result = run_full_pipeline(
                discovery, all_requested_fields, professor_count, progress_callback=_progress_cb,
            )
        st.session_state["pipeline_result"] = pipeline_result
        prof_progress.progress(1.0, text="Done")

        # Write results into the spreadsheet.
        working_df = df.copy()
        # Ensure a name column exists to key rows on.
        name_field = "Full Name" if "Full Name" in field_to_column else None
        if name_field is None and selected_fields:
            name_field = selected_fields[0]
        name_col = field_to_column.get(name_field) if name_field else None
        if name_col and name_col not in working_df.columns:
            working_df[name_col] = ""

        from agents.spreadsheet_agent import write_verified_professor

        for record in pipeline_result.professors:
            outcome = write_verified_professor(
                working_df, record.full_name, field_to_column,
                record.verified_fields, name_col, allow_overwrite,
            )
            working_df = outcome.dataframe
            if include_sources:
                if "Source URL" not in working_df.columns:
                    working_df["Source URL"] = ""
                working_df.at[outcome.row_index, "Source URL"] = record.source_url
            if show_confidence:
                if "Confidence" not in working_df.columns:
                    working_df["Confidence"] = ""
                working_df.at[outcome.row_index, "Confidence"] = record.overall_confidence

        st.session_state["results_df"] = working_df

        import datetime
        st.session_state["research_history"].append({
            "site": site_url, "count": len(pipeline_result.professors),
            "when": datetime.datetime.now().strftime("%b %d, %Y"),
        })

        requested = professor_count
        found = len(pipeline_result.professors)
        verified = sum(1 for p in pipeline_result.professors if p.overall_confidence in ("High", "Medium"))
        missing_fields = sum(
            1 for p in pipeline_result.professors for f, v in p.verified_fields.items()
            if v.get("value") == NOT_AVAILABLE
        )
        errors_count = sum(len(p.errors) for p in pipeline_result.professors) + len(pipeline_result.errors)

        st.success("Research Complete ✓")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Requested", requested)
        c2.metric("Found", found)
        c3.metric("Verified", verified)
        c4.metric("Missing fields", missing_fields)
        c5.metric("Errors", errors_count)

        if pipeline_result.errors:
            with st.expander("⚠️ Error report"):
                for e in pipeline_result.errors:
                    st.caption(e)

    except KeyboardInterrupt:
        st.warning("Research stopped. Partial results (if any) are shown below.")
    except GroqNotConfiguredError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"An unexpected error occurred during research: {exc}")

if st.session_state.get("pipeline_result") is not None:
    if st.button("⏹ Stop Research"):
        st.session_state["stop_requested"] = True

# ---------------------------------------------------------------------------
# Step 6 — Duplicate detection
# ---------------------------------------------------------------------------
results_df = st.session_state.get("results_df")
if results_df is not None:
    st.header("6. Preview Results")

    name_col_final = None
    for candidate in field_to_column.values():
        if "name" in candidate.lower() or "professor" in candidate.lower():
            name_col_final = candidate
            break
    email_col_final = field_to_column.get("Email")
    url_col_final = field_to_column.get("Profile URL")

    dup_groups = detect_duplicates(results_df, name_col_final, email_col_final, url_col_final)
    if dup_groups:
        st.warning(f"⚠️ {len(dup_groups)} possible duplicate group(s) found.")
        for group in dup_groups:
            label = results_df.loc[group[0], name_col_final] if name_col_final else f"rows {group}"
            with st.expander(f"Duplicate: {label}"):
                st.dataframe(results_df.loc[group], use_container_width=True)
                action = st.radio(
                    "Action", ["Keep first", "Remove duplicate(s)"],
                    key=f"dup_{group[0]}", horizontal=True,
                )
                if action == "Remove duplicate(s)" and st.button("Apply", key=f"apply_dup_{group[0]}"):
                    results_df = results_df.drop(index=group[1:]).reset_index(drop=True)
                    st.session_state["results_df"] = results_df
                    st.rerun()

    st.subheader("Editable Preview")
    edited_df = st.data_editor(results_df, use_container_width=True, num_rows="dynamic", key="editor")
    if st.button("💾 Save Corrections"):
        st.session_state["results_df"] = edited_df
        st.success("Corrections saved.")

    # -----------------------------------------------------------------
    # Step 7 — Download
    # -----------------------------------------------------------------
    st.header("7. Download")
    hyperlink_cols = [c for c in ["Profile URL", "Source URL", "Personal Website", "Google Scholar"] if c in results_df.columns]
    excel_bytes = to_excel_bytes(st.session_state["results_df"], hyperlink_columns=hyperlink_cols)
    csv_bytes = to_csv_bytes(st.session_state["results_df"])

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇️ Download Excel", data=excel_bytes,
        file_name="professor_research_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    dl2.download_button(
        "⬇️ Download CSV", data=csv_bytes,
        file_name="professor_research_results.csv", mime="text/csv",
    )

    # -----------------------------------------------------------------
    # Step 8 — Research summary + matching
    # -----------------------------------------------------------------
    st.header("8. Research Summary & Matching")

    if st.button("📊 Generate Research Summary"):
        pipeline_result = st.session_state.get("pipeline_result")
        if pipeline_result:
            depts = set()
            all_research = []
            for p in pipeline_result.professors:
                dept_val = p.verified_fields.get("Department", {}).get("value")
                if dept_val and dept_val != NOT_AVAILABLE:
                    depts.add(dept_val)
                ri_val = p.verified_fields.get("Research Interests", {}).get("value")
                if ri_val and ri_val != NOT_AVAILABLE:
                    all_research.append(ri_val)
            st.write(f"**Professors found:** {len(pipeline_result.professors)}")
            st.write(f"**Departments:** {', '.join(sorted(depts)) if depts else '—'}")
            if all_research:
                st.write("**Research areas mentioned:**")
                st.caption(" • ".join(all_research[:20]))

    topic_query = st.text_input("Find professors matching a research topic", placeholder="e.g. Large Language Models")
    if st.button("🔍 Find Matches", disabled=not topic_query):
        pipeline_result = st.session_state.get("pipeline_result")
        if not pipeline_result or not pipeline_result.professors:
            st.info("Run research first to build a professor list to search.")
        else:
            professors_payload = [
                {
                    "full_name": p.full_name,
                    "research_fields": {
                        f: v.get("value") for f, v in p.verified_fields.items()
                        if "research" in f.lower() or "interest" in f.lower() or "area" in f.lower()
                    },
                    "profile_url": p.verified_fields.get("Profile URL", {}).get("value", p.source_url),
                    "source_url": p.source_url,
                }
                for p in pipeline_result.professors
            ]
            try:
                raw = chat_completion(
                    MATCH_SYSTEM_PROMPT,
                    MATCH_USER_TEMPLATE.format(
                        topic=topic_query,
                        professors_json=json.dumps(professors_payload, indent=2),
                    ),
                )
                match_payload = extract_json_block(raw)
                matches = match_payload.get("matches", [])
                if not matches:
                    st.info("No professors in the current results matched that topic.")
                else:
                    st.dataframe(pd.DataFrame(matches), use_container_width=True)
            except GroqNotConfiguredError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Matching failed: {exc}")

# ---------------------------------------------------------------------------
# Start Over
# ---------------------------------------------------------------------------
st.divider()
if st.button("🔄 Start Over"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
