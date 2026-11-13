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
- **Versioned, reusable prompts** (`src/prompts.py`): two extraction variants and one explanation prompt. Gemini is used for unstructured reading and explanation only; every decision is deterministic Python.
- **Programmatic verification of both prompts' output** — the narrative LLM cannot change a requirement status, invent evidence, score the candidate, or recommend hiring; attempts are corrected and reported.
- Gemini structured output (`response_schema`) + strict Pydantic validation (`extra="forbid"`, date format checks); one retry with validation feedback, then a structured error for human review.
- **Grounding layer**: every item's evidence quotes are checked against the CV text; unverifiable items are dropped *and reported* — hallucinations cannot silently enter results.
- Deterministic, rule-based requirements matching: `met` / `not_met` / `unknown`, where `not_met` requires proof, not absence.
- Recommendation schema that **cannot represent a final decision** (`requires_human_review: Literal[True]`).
- **Execution trace per run**: stage timings, retry counts, error categories and API-reported token usage — with no CV text, credentials or model reasoning in it.
- **Per-run artifacts** under `outputs/runs/<run_id>/`, so a new run never overwrites an older one.
- **Reviewer corrections recorded beside the output**, never written into it, so the AI result stays auditable after a human disagrees.
- Prompt-injection defenses: CV is delimited as untrusted data, delimiters stripped, injection-looking content flagged; the deterministic matcher is unaffected by CV wording.
- No protected attributes: the schema has no fields for age/gender/religion/photo etc., the prompt forbids extracting them, and matching never sees them.
- Clearly labelled **mock mode** for demos without an API key.
- Privacy: CVs are processed in memory and not persisted unless the caller asks for artifacts; logs contain sizes and codes, never CV text.

## 4. Technology Stack

Python 3.11+, PyMuPDF, Google Gemini API via `google-genai`, Pydantic v2, pytest, python-dotenv, Jupyter Notebook. Dev-only: reportlab (sample PDF generation), tabulate, jupyter.

## 5. Project Structure

```
├── prototype.ipynb            # primary demonstration (12 sections, executed)
├── README.md
├── requirements.txt           # runtime deps
├── requirements-dev.txt       # + test/demo deps
├── .env.example               # GEMINI_API_KEY template
├── src/
│   ├── schemas.py             # strict pydantic models + error base
│   ├── prompts.py             # the versioned prompts (no business logic)
│   ├── gemini_client.py       # one gemini call path: structured output, retry, errors
│   ├── pdf_parser.py          # pdf -> text, loud failures
│   ├── llm_extractor.py       # extraction prompt + grounding, mock mode
│   ├── recommendation.py      # jd parsing + deterministic matching
│   ├── narrative.py           # recommendation prompt + programmatic verification
│   ├── tracing.py             # execution trace: stages, retries, tokens
│   ├── artifacts.py           # per-run output files
│   ├── review.py              # human corrections, recorded separately
│   ├── pipeline.py            # orchestration + cli
│   └── text_utils.py          # shared normalisation
├── data/
│   ├── sample_cvs/            # 4 synthetic cvs (fictional people)
│   ├── job_requirements/      # sample job description
│   └── evaluation/            # hand-labeled ground truth + fixtures
├── outputs/                   # generated results, evaluation reports, per-run artifacts
│   └── runs/<run_id>/         # candidate_extraction / requirement_matching / screening_report / execution_trace
├── tests/                     # 162 pytest tests
├── scripts/
│   ├── make_sample_cv.py              # regenerate synthetic cvs
│   ├── evaluate.py                    # system-level measured evaluation (+ csv)
│   ├── evaluate_prompts.py            # prompt-level measured evaluation
│   └── compare_extraction_prompts.py  # a/b comparison of extraction-v1 vs v2
└── docs/
    ├── architecture.md
    ├── prompt_engineering.md
    ├── design_decisions.md
    ├── evaluation.md
    └── assignment_alignment.md
```

## 6. System Architecture and Workflow

```
PDF ─► pdf_parser ─► llm_extractor ─► grounding ─► recommendation ─► narrative ─► ScreeningResult
JD  ─► parse_job_description ──────────────────────────┘  (deterministic)  (verified)   + ExecutionTrace
```

Seven steps: CV input → PDF text extraction → LLM extraction → Pydantic validation → requirements matching → recommendation with caveats → human review, plus an optional LLM narrative that explains (never alters) the matching result. Every stage is timed into an execution trace. Details in `docs/architecture.md`; prompt design in `docs/prompt_engineering.md`; the reasoning behind each choice in `docs/design_decisions.md`.

## 7. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## 8. Environment Configuration

```bash
cp .env.example .env               # then fill in GEMINI_API_KEY
```

Without a key everything still runs in mock mode (clearly labelled, no API call). Never commit `.env`; `.gitignore` already excludes it. `GEMINI_MODEL` overrides the model name, which matters because model names get retired.

## 9. How to Run

```bash
jupyter notebook prototype.ipynb   # run cells top to bottom
```

The notebook selects live mode when `GEMINI_API_KEY` is present and falls back to labelled mock mode otherwise. CLI alternative:

```bash
python -m src.pipeline --cv data/sample_cvs/synthetic_cv_01_clean.pdf \
  --jd data/job_requirements/data_analyst.txt --mode mock [--narrative]
```

Each CLI run writes `outputs/runs/<run_id>/` with four files (`--no-artifacts` to disable). A failed run writes its execution trace alone — no result is fabricated.

Evaluation:

```bash
python scripts/evaluate.py                    # + outputs/evaluation_results.csv
python scripts/evaluate_prompts.py            # add --mode live for model behaviour
python scripts/compare_extraction_prompts.py  # add --mode live for the real a/b
```

## 10. Example Input and Output

Input: `data/sample_cvs/synthetic_cv_01_clean.pdf` + `data/job_requirements/data_analyst.txt`.
Output: `outputs/sample_output_mock.json` — per-requirement assessments with evidence quotes, e.g. R1–R4 `met`, R5 (Docker/ML deployment) `unknown` because the CV contains no evidence either way, signal `partial_support`, `requires_human_review: true`. The same run split by audience lives in `outputs/runs/<run_id>/`.

## 11. Human Review

The final decision is structurally unavailable to the system: `requires_human_review` is typed `Literal[True]` and `is_final_decision` `Literal[False]`. Reviewer disagreement is recorded with `src/review.py`:

```python
from src.review import new_review_record, record_correction, review_view

record = new_review_record(result.trace.run_id)
record_correction(record, result, target="recommendation.assessments[R5]",
                  issue="unverified_requirement",
                  reviewer_note="candidate confirmed Docker experience by phone")
view = review_view(result, record)   # ai output and objection, side by side
```

Corrections are typed (`incorrect_extraction`, `unsupported_qualification`, `unverified_requirement`, `ambiguous_evidence`, `incorrect_matching`), validated against the result when recorded, and stored in a separate file. The AI output is never modified, so it can still be audited after review.

## 12. Testing

```bash
pytest
```

162 tests: PDF edge cases (valid/empty/scanned/corrupt/encrypted), schema validation (types, dates, forbidden extra fields, undecidable-decision schema), prompt rendering, prompt-version selection and injection containment, matching rules (met/not_met/unknown, must-have vs nice-to-have, date arithmetic), narrative verification (status tampering, fabricated evidence, scores, hire/reject language, `unknown` misreported as a gap), execution tracing (stage order, failure recording, retry and token accounting), run artifacts (four files, per-run isolation, no CV text in traces), reviewer corrections (path validation, scope rules, immutability of the AI output), pipeline end-to-end, API-failure and invalid-response handling with a fake client, human-review fallback, no-persistence and no-CV-in-logs checks. LLM calls in tests are mocked.

## 13. Evaluation Results

Three measured evaluations; all figures are actual script output from this environment (mock mode). Per-case rows: `outputs/evaluation_results.csv`.

**System** (`scripts/evaluate.py`): requirement-level matching accuracy **1.0** on 35 hand-labeled labels (5/5 cases exact); schema-validation decisions **6/6**; hallucination containment **4/4** fabricated items dropped with **9/9** genuine kept; structured error paths **3/3**. End-to-end pipeline success is **3/4** CVs at **9.186 s** mean in live mode (the committed report) and 4/4 at 0.004 s in mock mode — the one live failure was an `api_error` on `synthetic_cv_04_injection.pdf`, not a pipeline fault.

**Prompts** (`scripts/evaluate_prompts.py`): prompt-construction checks **14/14**; extraction schema decisions **6/6** with **4/4** unsupported claims blocked; planted narrative violations detected **11/11** with **0** false alarms; status integrity preserved **6/6**.

**Prompt A/B** (`scripts/compare_extraction_prompts.py`): both extraction versions pass every required-rule check and contain **3/3** injection probes under an identical configuration; `extraction-v2` costs 986 more characters of system prompt. No quality claim is made offline.

Extraction accuracy against real Gemini is **still not measured**: `evaluate_prompts.py --mode live` was run and all four CVs returned `api_error` — HTTP 404, a `GEMINI_MODEL` override pointing at a model name Google has retired, so no call reached a model. The measurement has to be repeated with `GEMINI_MODEL` unset. Full discussion: `docs/evaluation.md`, `docs/prompt_engineering.md`.

## 14. Known Limitations

- **Live Gemini extraction accuracy is unmeasured.** Live integration does work — three of four CVs were screened end to end against the real API — and the HTTP 400 schema bug, a retired model name (404) and 503 spikes were all found that way. But integration working is not the same as extraction being accurate, and the accuracy run itself failed on all four CVs with a 404 caused by a stale `GEMINI_MODEL` override, not by the model's behaviour.
- `extraction-v2` is a hypothesis, not an improvement: it has not been compared against v1 on real model output, which is why `extraction-v1` remains the default.
- Offline prompt metrics measure the templates and this system's enforcement, not how well Gemini follows them.
- Tiny, author-written evaluation sets — regression checks, not a benchmark.
- Narrative prose filtering is pattern-based and deliberately conservative; the status and evidence guarantees do not depend on it.
- Keyword matching misses synonyms; the JD parser expects bulleted *Requirements* / *Nice to have* sections.
- Years of experience count all roles without judging relevance (stated in the output, and pinned down by the `irrelevant_experience_candidate` evaluation case).
- Token usage is only recorded when the API reports it; in mock mode there is none, so the API cost of a live screening is not yet measured.
- No OCR; scanned PDFs are rejected.

## 15. Future Improvements

Run both evaluations and the prompt A/B in live mode and report the numbers; feed reviewer corrections back as an independent source of ground truth; synonym/taxonomy-aware matching; a reviewer-facing UI for inspecting evidence; OCR for scanned CVs; a larger evaluation set written by someone other than the author.

## 16. Use of AI Tools

This project was built with AI assistance (Claude), used for implementation, refactoring, test writing and documentation drafting, under my direction and review. The architecture decisions, the evaluation design and the honesty constraints (no invented metrics, `unknown` over `not_met`, no automated decision) are mine, and every number in this repository comes from a script that was actually run in the stated mode — none was written by hand. Where something has not been measured, the documentation says so instead of estimating.
