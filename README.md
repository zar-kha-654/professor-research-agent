# 🎓 Professor Research Agent

A Streamlit web application that researches university faculty websites and
automatically fills in a spreadsheet of professor information — with every
extracted fact grounded in a source URL, verified against the original page,
and never hallucinated.

Runs entirely in the browser via **Streamlit Cloud**. No local Python,
Docker, or database installation required.

---

## Features

- **Upload any spreadsheet** (`.xlsx` or `.csv`) — empty template, a list of
  names, or a partially-filled sheet. Existing data is never overwritten
  unless you explicitly allow it.
- **Point at any institutional faculty website** — the agent discovers
  faculty listing pages and individual profile pages within safe, polite
  crawl limits (respects `robots.txt`, capped pages/depth, timeouts).
- **Choose exactly which fields you want** — 25+ standard fields (name,
  email, department, research interests, publications, etc.) plus free-text
  **custom fields** ("Accepting graduate students?", "Lab name", …).
- **Automatic + manual column mapping** so extracted data always lands in
  the right spreadsheet column, with a one-click "add missing column".
- **Source-grounded extraction** — every field is extracted from real page
  text via retrieval-augmented generation (RAG), never invented. Anything
  not found on the page is recorded as `N/A`.
- **Verification pass** — a second agent cross-checks every field against
  the source text and assigns a High/Medium/Low confidence rating.
- **Duplicate detection** by name/email/profile URL, with keep/remove
  actions.
- **Live progress UI**, editable results preview, Excel/CSV download with
  clickable source links, research summaries, and topic-matching search
  ("find professors working on Large Language Models").

---

## Architecture

```
Institutional URL
      │
      ▼
Website discovery (Python: requests + BeautifulSoup, robots.txt aware)
      │
      ▼
Page fetch + clean-text extraction
      │
      ▼
Chunking → embeddings → FAISS vector store (per-session, in-memory only)
      │
      ▼
RAG retrieval (top-k relevant chunks per professor/field)
      │
      ▼
Agent 2 — Information Extractor (Groq LLM → structured JSON)
      │
      ▼
Agent 3 — Verification Agent (checks extracted JSON against source text)
      │
      ▼
Agent 4 — Spreadsheet Agent (maps fields → columns, writes rows, dedups)
      │
      ▼
Completed .xlsx / .csv
```

The LLM is only responsible for **understanding, extraction, normalization,
classification, and verification**. Everything else — downloading pages,
parsing HTML, spreadsheet I/O, validation, retries, and progress tracking —
is deterministic Python. This keeps the app debuggable, cheap to run, and
resistant to hallucination.

### How the agents work

| Agent | Role | What it does |
|---|---|---|
| 1. Researcher | Academic Website Research Specialist | Crawls the institutional site (Python) to find faculty listing and profile pages |
| 2. Extractor | Academic Information Extraction Specialist | Converts retrieved page text into structured JSON fields, using `N/A` for anything not present |
| 3. Verifier | Academic Data Verification Specialist | Cross-checks extracted fields against the source text; assigns High/Medium/Low confidence |
| 4. Spreadsheet Agent | Academic Spreadsheet Data Manager | Maps verified fields to spreadsheet columns and writes rows without clobbering existing data |

Agents are defined with **CrewAI** (`agents/*.py`) for role/goal/backstory
structure and orchestration; the actual step-by-step pipeline that the UI
drives is deterministic Python in `utils/pipeline.py`, calling the Groq LLM
only where genuine language understanding is needed.

### How RAG works

Each research session builds a **fresh, in-memory** FAISS index from the
pages it fetches (see `tools/rag.py`). Page text is cleaned, chunked
(~800 words with overlap), embedded with a small `sentence-transformers`
model, and indexed. When extracting a professor's fields, the pipeline
retrieves the most relevant chunks for that professor + the requested
fields, and only that retrieved text is shown to the extraction LLM — this
is what keeps the model grounded instead of guessing from general
knowledge. Nothing is persisted between sessions.

---

## Tech Stack

- Python 3.11+
- Streamlit (UI)
- CrewAI (agent role/orchestration definitions)
- Groq API (LLM inference)
- sentence-transformers + FAISS (lightweight, session-scoped RAG)
- BeautifulSoup + Requests (web extraction)
- pandas + openpyxl (spreadsheet read/write)

---

## Project Structure

```
professor-research-agent/
│
├── app.py                     # Streamlit UI + workflow
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml            # theme + upload size
│   └── secrets.toml.example   # copy to secrets.toml (not committed)
│
├── agents/
│   ├── researcher.py          # Agent 1 — website research
│   ├── extractor.py           # Agent 2 — information extraction
│   ├── verifier.py            # Agent 3 — verification
│   └── spreadsheet_agent.py   # Agent 4 — spreadsheet writing
│
├── tools/
│   ├── web_scraper.py         # page fetch, robots.txt, clean text
│   ├── website_discovery.py   # crawl + classify listing/profile pages
│   ├── rag.py                 # chunk/embed/FAISS/retrieve
│   ├── spreadsheet.py         # load/map/write/export
│   └── validation.py          # URL + schema validation
│
├── utils/
│   ├── prompts.py             # all LLM prompt templates + anti-hallucination rules
│   ├── config.py              # single source of truth for constants/model name
│   ├── helpers.py             # retry/backoff, JSON parsing, chunking
│   ├── llm_client.py          # Groq API wrapper
│   └── pipeline.py            # end-to-end orchestration used by app.py
│
└── outputs/
    └── .gitkeep
```

---

## Environment Variables / Streamlit Secrets

| Key | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `GROQ_MODEL_NAME` | No | Overrides the default model (`llama-3.3-70b-versatile`) |

Locally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
and fill in your key. **Never commit `secrets.toml`** — it's already in
`.gitignore`.

---

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: Professor Research Agent"
git branch -M main
git remote add origin https://github.com/<your-username>/professor-research-agent.git
git push -u origin main
```

## Streamlit Cloud Deployment

1. Create a GitHub repository and push this project to it.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and branch.
4. Set the main file path to `app.py`.
5. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
6. Click **Deploy**.

The app will build from `requirements.txt` automatically — no Docker or
server setup needed.

---

## Usage

1. **Upload** your spreadsheet, or start from a blank template.
2. **Enter the institutional faculty website URL** and click **Analyze
   Website**.
3. **Select the fields** you want (checkboxes + custom free-text fields).
4. **Map spreadsheet columns** — auto-mapping is suggested; correct or add
   new columns as needed.
5. Set **how many professors** you want in the sidebar.
6. Click **🚀 Start Professor Research** and watch live progress.
7. **Preview**, edit, and **save corrections** in the interactive table.
8. **Download** the completed `.xlsx` or `.csv`.
9. Optionally generate a **research summary** or **search by topic**.

---

## Limitations

- JavaScript-heavy faculty directories (single-page apps that render
  content client-side) cannot be read by the static fetcher used here —
  the app will report this rather than silently returning empty data.
- Sites that require authentication, are behind a CAPTCHA, or actively
  block automated requests will not be crawled — the app never attempts to
  bypass these protections.
- Extraction quality depends on how much information a given professor's
  page actually contains; the app returns `N/A` rather than guessing.
- FAISS/embeddings add noticeable install size; if Streamlit Cloud's free
  tier struggles with the build, consider pinning lighter versions or
  reducing `EMBEDDING_MODEL_NAME` to an even smaller model.

## Responsible Web Scraping

This app:
- Respects `robots.txt` for each domain it visits.
- Stays within the supplied institution's own domain.
- Applies a politeness delay between requests and hard caps on pages/depth.
- Never attempts to bypass logins, CAPTCHAs, paywalls, or other access
  controls — it reports the block to the user instead.
- Does not persist scraped university data beyond the current session
  unless you explicitly export it yourself.

Please use this tool only against sites you have the right to crawl, and
in line with the target site's terms of use.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "GROQ_API_KEY is not set" | Add the key under Streamlit Secrets (or export it as an env var locally) |
| "The website could not be accessed" | Check the URL, or the site may be down/blocking automated requests |
| "blocked automated access (403)" | The site is actively blocking bots — try a different, more specific faculty page |
| "appears to require JavaScript" | The page is a client-rendered SPA; the static fetcher can't read it |
| Many fields come back `N/A` | The professor's page simply doesn't list that information — this is expected, not a bug |
| Slow first run | The embedding model downloads on first use; subsequent runs are faster |
| Streamlit Cloud build fails on FAISS/sentence-transformers | Try pinning exact versions in `requirements.txt` that are known to have prebuilt wheels for the Cloud Python version |

---

## License

Provided as-is for educational/research use. Review and comply with each
target website's terms of service before crawling it.
