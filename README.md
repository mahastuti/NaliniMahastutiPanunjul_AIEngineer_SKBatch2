# AI-Based CV Screening System

Sekolah Karir Batch 2 — AI Engineer mini assignment.

## 1. Project Overview

An end-to-end prototype that reads a candidate CV (PDF), extracts structured candidate information with a **pretrained Gemini model via API** (no model is trained), validates it with Pydantic, verifies every extracted item against the CV text, and compares the result with job requirements to produce an **evidence-based, explainable recommendation for a human reviewer**.

## 2. Problem Statement and Objectives

Manual CV screening is slow and inconsistent; naive LLM screening hallucinates qualifications and hides its reasoning. This prototype shows how to use an LLM for the part it is good at (reading unstructured text into a schema) while keeping judgement **deterministic, explainable, and under human control**:

- extract only what the CV states, with verbatim evidence;
- never treat missing information as a negative qualification;
- never emit an automated hiring decision.

## 3. System Features

- PDF text extraction with explicit rejection of empty / scanned / corrupt / encrypted files (no OCR, no invented text).
- **Two versioned, reusable prompts** (`src/prompts.py`): one for extraction, one for evidence-based explanation. Gemini is used for unstructured reading and explanation only; every decision is deterministic Python.
- **Programmatic verification of both prompts' output** — the narrative LLM cannot change a requirement status, invent evidence, score the candidate, or recommend hiring; attempts are corrected and reported.
- Gemini structured output (`response_schema`) + strict Pydantic validation (`extra="forbid"`, date format checks); one retry with validation feedback, then a structured error for human review.
- **Grounding layer**: every item's evidence quotes are checked against the CV text; unverifiable items are dropped *and reported* — hallucinations cannot silently enter results.
- Deterministic, rule-based requirements matching: `met` / `not_met` / `unknown`, where `not_met` requires proof, not absence.
- Recommendation schema that **cannot represent a final decision** (`requires_human_review: Literal[True]`).
- Prompt-injection defenses: CV is delimited as untrusted data, delimiters stripped, injection-looking content flagged; the deterministic matcher is unaffected by CV wording.
- No protected attributes: the schema has no fields for age/gender/religion/photo etc., the prompt forbids extracting them, and matching never sees them.
- Clearly labelled **mock mode** for demos without an API key.
- Privacy: CVs are processed in memory, not persisted by default; logs contain sizes and codes, never CV text.

## 4. Technology Stack

Python 3.11+, PyMuPDF, Google Gemini API via `google-genai`, Pydantic v2, pytest, python-dotenv, Jupyter Notebook. Dev-only: reportlab (sample PDF generation), tabulate, jupyter.

## 5. Project Structure

```
├── prototype.ipynb            # primary demonstration (10 sections, executed)
├── README.md
├── requirements.txt           # runtime deps
├── requirements-dev.txt       # + test/demo deps
├── .env.example               # GEMINI_API_KEY template
├── src/
│   ├── schemas.py             # strict pydantic models + error base
│   ├── prompts.py             # the two versioned prompts (no business logic)
│   ├── gemini_client.py       # one gemini call path: structured output, retry, errors
│   ├── pdf_parser.py          # pdf -> text, loud failures
│   ├── llm_extractor.py       # extraction prompt + grounding, mock mode
│   ├── recommendation.py      # jd parsing + deterministic matching
│   ├── narrative.py           # recommendation prompt + programmatic verification
│   ├── pipeline.py            # orchestration + cli
│   └── text_utils.py          # shared normalisation
├── data/
│   ├── sample_cvs/            # 3 synthetic cvs (fictional people)
│   ├── job_requirements/      # sample job description
│   └── evaluation/            # hand-labeled ground truth + fixtures
├── outputs/                   # generated results & evaluation report
├── tests/                     # 57 pytest tests
├── scripts/
│   ├── make_sample_cv.py      # regenerate synthetic cvs
│   ├── evaluate.py            # system-level measured evaluation
│   └── evaluate_prompts.py    # prompt-level measured evaluation
└── docs/
    ├── architecture.md
    ├── prompts.md
    └── evaluation.md
```

## 6. System Architecture and Workflow

```
PDF ─► pdf_parser ─► llm_extractor ─► grounding ─► recommendation ─► narrative ─► ScreeningResult
JD  ─► parse_job_description ──────────────────────────┘  (deterministic)  (verified)
```

Seven steps: CV input → PDF text extraction → LLM extraction → Pydantic validation → requirements matching → recommendation with caveats → human review, plus an optional LLM narrative that explains (never alters) the matching result. Details in `docs/architecture.md`; prompt design in `docs/prompts.md`.

## 7. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## 8. Environment Configuration

```bash
cp .env.example .env               # then fill in GEMINI_API_KEY
```

Without a key everything still runs in mock mode (clearly labelled, no API call). Never commit `.env`; `.gitignore` already excludes it.

## 9. How to Run the Notebook

```bash
jupyter notebook prototype.ipynb   # run cells top to bottom
```

The committed notebook already contains executed outputs (mock mode). CLI alternative:

```bash
python -m src.pipeline --cv data/sample_cvs/synthetic_cv_01_clean.pdf \
  --jd data/job_requirements/data_analyst.txt --mode mock [--narrative]
```

## 10. Example Input and Output

Input: `data/sample_cvs/synthetic_cv_01_clean.pdf` + `data/job_requirements/data_analyst.txt`.
Output: `outputs/sample_output_mock.json` — per-requirement assessments with evidence quotes, e.g. R1–R4 `met`, R5 (Docker/ML deployment) `unknown` because the CV contains no evidence either way, signal `partial_support`, `requires_human_review: true`.

## 11. Testing

```bash
pytest
```

97 tests: PDF edge cases (valid/empty/scanned/corrupt/encrypted), schema validation (types, dates, forbidden extra fields, undecidable-decision schema), prompt rendering and injection containment, matching rules (met/not_met/unknown, must-have vs nice-to-have, date arithmetic), narrative verification (status tampering, fabricated evidence, scores, hire/reject language, `unknown` misreported as a gap), pipeline end-to-end, API-failure and invalid-response handling with a fake client, human-review fallback, no-persistence and no-CV-in-logs checks. LLM calls in tests are mocked.

## 12. Evaluation Results

Two measured evaluations; all figures are actual script output from this environment (mock mode).

**System** (`python scripts/evaluate.py`): requirement-level matching accuracy **1.0** on 28 hand-labeled labels (4/4 cases exact); schema-validation decisions **6/6**; hallucination containment **4/4** fabricated items dropped with **9/9** genuine kept; pipeline success **4/4** CVs; structured error paths **3/3**.

**Prompts** (`python scripts/evaluate_prompts.py`): prompt-construction checks **14/14**; extraction schema decisions **6/6** with **4/4** unsupported claims blocked; planted narrative violations detected **11/11** with **0** false alarms; status integrity preserved **6/6**.

Model behaviour against real Gemini is **not measured** (needs an API key): `python scripts/evaluate_prompts.py --mode live`. Full discussion: `docs/evaluation.md`, `docs/prompts.md`.

## 13. Known Limitations

- Live Gemini integration **not verified** in the development environment (no key); live paths are tested against a fake SDK client, which proves error handling and verification, not model compliance.
- Offline prompt metrics measure the templates and this system's enforcement, not how well Gemini follows them.
- Tiny, author-written evaluation sets — regression checks, not a benchmark.
- Narrative prose filtering is pattern-based and deliberately conservative; the status and evidence guarantees do not depend on it.
- Keyword matching misses synonyms; JD parser expects bulleted *Requirements* / *Nice to have* sections.
- Years-of-experience counts all roles without judging relevance (stated in the output).
- No OCR; scanned PDFs are rejected.

## 14. Future Improvements

Verify live extraction and run both evaluations in live mode; A/B comparison across prompt versions (the registry already supports it); synonym/taxonomy-aware matching; reviewer-facing UI for inspecting evidence; OCR for scanned CVs; larger independent evaluation set.
