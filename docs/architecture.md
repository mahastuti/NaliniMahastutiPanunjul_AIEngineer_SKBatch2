# Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │                        src/pipeline.py                        │
                 │           run_screening()  — orchestration, one result        │
                 └──────────────────────────────────────────────────────────────┘
 PDF bytes ─► src/pdf_parser.py ─► src/llm_extractor.py ─► src/recommendation.py ─► src/narrative.py ─► ScreeningResult
                extract_text()       extract_profile()       build_recommendation()   generate_narrative()
                                     + ground_profile()      (deterministic)          + verify_narrative()
 JD text  ──────────────────────────────────────────► parse_job_description()

                 src/prompts.py       — the two versioned prompts (no business logic)
                 src/gemini_client.py — one gemini call path: structured output, retry, errors
                 src/schemas.py       — shared pydantic contracts for every arrow above
                 src/text_utils.py    — shared text normalisation (grounding + matching)
```

Two LLM calls, both through `src/gemini_client.py`, both with structured output and temperature 0:
**extraction** (reads unstructured text) and **narrative** (explains the finished result). Neither
can decide anything: the matching engine sits between them and is pure Python.

The system uses a **pretrained Gemini model through API integration** (`google-genai`). No foundation model is trained or fine-tuned.

## Components and responsibilities

### `src/schemas.py` — contracts
Strict Pydantic models (`extra="forbid"`, whitespace stripping, `YYYY[-MM]` date validation). Three groups:

- **Extraction**: `CandidateProfile` (education, work experience, skills, projects, certifications; every item has an `evidence: list[str]` of verbatim CV quotes). Missing single values are `null`, missing collections are empty — explicitly representable. There are **no fields for protected attributes**; because extras are forbidden, a model response containing `age` or `gender` fails validation.
- **Job side**: `JobDescription` / `JobRequirement` — typed requirement kinds (`min_years`, `education_level`, `skills`, `other`), mandatory vs optional, `all`/`any` term mode, certification-only flag.
- **Recommendation**: per-requirement `RequirementAssessment` (status, `facts`, `interpretation`, `evidence`), `Caveat`, and `Recommendation` whose `requires_human_review: Literal[True]` / `is_final_decision: Literal[False]` make an automated final decision *unrepresentable*, not merely discouraged.

`ScreeningError(code, message)` is the base error; every failure carries a machine-readable code so callers never parse messages.

### `src/pdf_parser.py` — CV input and text extraction
Validates bytes before parsing (empty, >5 MB, missing `%PDF` header), opens with PyMuPDF, then rejects encrypted, page-less, oversized documents. After extraction it distinguishes a **scanned** PDF (no text + images present) from a blank one, rejects mostly-unreadable text, and never invents content — OCR is deliberately out of scope. Logs sizes only, never content.

### `src/prompts.py` — prompt templates (data, not logic)
The two versioned prompts, their input variables, temperatures (0.0 for reproducibility) and
target schemas, plus `PROMPT_REGISTRY` for evaluation metadata and `wrap_untrusted()`, which
wraps untrusted content in a delimiter block after stripping that delimiter from the content so
it cannot close the block early. Full detail: `docs/prompts.md`.

### `src/gemini_client.py` — one call path
Shared by both prompts so the retry and error policy exists once: builds the client from
`GEMINI_API_KEY` (never hardcoded), sends `system_instruction` + user prompt with
`response_mime_type="application/json"` and a cleaned JSON schema (`response_schema_for()`: the
Gemini API rejects the `additionalProperties` that `extra="forbid"` generates, so the keyword is
dropped from the API schema while the Pydantic model still enforces it on our side), splits SDK `APIError`s
into retryable (429/5xx) and fatal (4xx), and retries a schema-invalid answer once with a
summary of the validation errors (locations and types only, so CV content is never echoed back
into a prompt). Persistent failure raises the caller's error class with a code — the
human-review fallback, never a fabricated result.

### `src/llm_extractor.py` — extraction (prompt 1)
Calls `extraction-v1` and validates into `CandidateProfile`. Mock mode returns an explicitly
labelled canned profile mirroring the clean sample CV; no network call.

**`ground_profile()`** is the anti-hallucination layer: every evidence quote is normalised and
searched in the CV text; items with no verifiable quote are dropped and the drop is **reported**
as a grounding issue (surfaced as a warning caveat). This converts "trust the LLM" into "verify,
then trust what survives".

### `src/narrative.py` — explanation (prompt 2), then verification
Calls `recommendation-v1` with the profile, the requirements, and the **deterministic matching
results**, asking only for prose. `verify_narrative()` then re-checks the answer in code, because
prompt instructions are not a guarantee: a restated status that disagrees with the engine is
overwritten, explanations for unknown or duplicated requirement IDs are removed, missing ones are
reported, unverifiable quotes are dropped, `potential_gaps` entries that name a non-`not_met`
requirement are removed (so missing information cannot be presented as a weakness), and sentences
containing scores, rankings or hire/reject language are stripped. Every correction becomes a
`narrative_violation` caveat. Mock mode builds the narrative from a template — no model call.

### `src/recommendation.py` — deterministic evaluation
No LLM here: matching is pure rules, so it is reproducible, testable, and immune to prompt injection.

- `parse_job_description()` turns bulleted *Requirements* / *Nice to have* sections into typed requirements (with a warning when no heading is found and everything defaults to mandatory).
- Assessment per kind:
  - **min_years**: date intervals per role, overlaps merged; year-only dates produce a conservative and a generous bound. `met` if the conservative bound suffices; `not_met` only if even the generous bound falls short *and* no role is undated; otherwise `unknown`.
  - **education_level**: ranked degree levels; completion must be evidenced; anything unprovable is `unknown`.
  - **skills**: whole-token keyword match across skills, certifications, role technologies/responsibilities, projects, education; `any`/`all` per requirement; certification requirements only look at certifications.
- The invariant everywhere: **missing information is `unknown`, never `not_met`**; every `not_met` carries the proof in `facts`, and `interpretation` states what the facts do not show (e.g. relevance not assessed).
- Output includes signal (`strong_support` / `partial_support` / `mandatory_gap` / `insufficient_information`), data gaps, and caveats (unverified extraction, keyword-matching limits, protected attributes excluded, per-gap notes).

### `src/pipeline.py` — orchestration
`run_screening()` wires the steps in order (JD parsed first so bad input fails before any API call), converts extraction warnings/grounding issues into caveats, flags injection-looking CV content, and returns one `ScreeningResult` with **provenance** (`gemini_llm` vs `mock_canned`, `human_verified: False`) so LLM output is never presented as verified fact. In-memory only; nothing persisted unless the caller writes it. Also exposes the CLI (`python -m src.pipeline`).

## Data flow guarantees

1. Every recommendation statement traces back to evidence quotes that verifiably occur in the CV.
2. LLM output is quarantined behind validation + grounding before any judgement logic sees it.
3. Both non-deterministic components sit *outside* the decision logic: extraction runs before the
   rules and cannot see them; the narrative runs after and is verified against them. No requirement
   status in a returned result was ever produced or modified by a model.
4. Provenance records, per result, which part came from a model, which prompt version, and that
   nothing is human-verified yet.
5. The final artifact structurally requires human review.
