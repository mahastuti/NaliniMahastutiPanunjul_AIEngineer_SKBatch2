# Engineering Design Decisions

Eight decisions that shaped this system, with what was given up for each. Where a decision has
not been validated by measurement, that is stated rather than argued around.

---

## 1. PyMuPDF for PDF text extraction

**Problem.** CVs arrive as PDFs. Text must come out reliably, and the system must be able to
tell "this PDF has no text" from "this PDF is a scan" — a wrong guess here silently produces an
empty profile and an unfairly negative screening.

**Approach.** PyMuPDF (`pymupdf`), reading embedded text per page and counting embedded images
to distinguish a scanned document from a blank one.

**Alternatives considered.** `pdfplumber` (better table reconstruction, noticeably slower);
`PyPDF2`/`pypdf` (pure Python, weaker text extraction on complex layouts); OCR via Tesseract.

**Rationale.** CV layout is mostly linear text, so table fidelity matters less than speed and
robustness. PyMuPDF also exposes `needs_pass` and `get_images()`, which is what makes the
encrypted/scanned/blank distinction possible instead of guessed.

**Trade-offs.** AGPL licensing (fine here, would need review for closed-source commercial use).
Multi-column CVs can interleave text; the grounding layer tolerates this because it normalises
whitespace before matching quotes.

**Limitations.** No OCR: scanned PDFs are rejected with `scanned_pdf` rather than processed.
Reading order in heavily designed CVs is not guaranteed.

---

## 2. Gemini API instead of training a model

**Problem.** Turning arbitrary CV prose into a fixed schema. This needs broad language
understanding of formats, abbreviations and phrasings that a small labelled dataset cannot cover.

**Approach.** A pretrained Gemini model through `google-genai`, with structured output. No
foundation model is trained or fine-tuned.

**Alternatives considered.** Fine-tuning a smaller open model on labelled CVs; a pure
regex/NER pipeline (spaCy); a classical resume parser library.

**Rationale.** The assignment is a two-week prototype with no labelled CV corpus and no budget
for one. Fine-tuning needs thousands of annotated CVs to beat a strong general model at exactly
this task, and a regex pipeline breaks on the first unfamiliar layout. Cost per CV is small and
latency is acceptable for human-in-the-loop screening.

**Trade-offs.** A network dependency and a per-call cost; an external vendor sees the CV text;
model names get retired (`gemini-2.5-flash` returned 404 during development, hence the
`GEMINI_MODEL` override); output varies between model versions, so evaluation numbers are tied
to a model identifier, which is recorded in the trace.

**Limitations.** Reproducibility is bounded by the vendor: temperature 0.0 reduces variation but
does not eliminate it, and a silent model update can move results.

---

## 3. Pydantic for structured-output validation

**Problem.** An LLM returning "JSON" can return the wrong shape, the wrong types, invented
fields, or prose wrapped around a JSON block. Downstream logic must never see any of that.

**Approach.** Strict Pydantic v2 models (`extra="forbid"`, whitespace stripping, `YYYY[-MM]`
date validators) used both as the API's `response_schema` and as the validator of what comes
back. Invalid output triggers one retry carrying a summary of the validation errors.

**Alternatives considered.** `json.loads` plus hand-written checks; `jsonschema`; trusting the
API's structured-output mode alone.

**Rationale.** Pydantic makes the contract executable and the failure mode loud. Two properties
matter beyond convenience: `extra="forbid"` turns "do not extract age or gender" from a polite
request into a validation error, and `Literal[True]`/`Literal[False]` on the recommendation make
an automated final decision *unrepresentable* rather than merely discouraged.

**Trade-offs.** Strictness rejects responses that are nearly right, costing a retry and
sometimes a run. That is the intended direction of failure: a rejected response is visible, a
silently coerced one is not. The API also rejects some keywords Pydantic emits
(`additionalProperties`), which forced `response_schema_for()` to clean the schema — strictness
is kept on our side.

**Limitations.** Validation proves shape, not truth. A perfectly typed hallucination passes;
that is the grounding layer's job, not Pydantic's.

---

## 4. Separating extraction and recommendation into two prompts

**Problem.** One prompt doing "read the CV and tell me if they are a good fit" produces output
whose correctness cannot be checked, because reading errors and judgement errors are entangled.

**Approach.** Two prompts with deterministic Python between them. `extraction-v*` reads;
`src/recommendation.py` decides; `recommendation-v1` explains the decision it was handed.

**Alternatives considered.** A single end-to-end prompt; chain-of-thought in one call;
a multi-agent arrangement.

**Rationale.** Each call then has a verifiable contract. Extraction output can be checked
against the CV text; narrative output can be checked against the matching result. A single call
offers nothing to check it against. It also keeps the LLM entirely outside the decision, which
is what makes the matching reproducible and injection-proof.

**Trade-offs.** Two API calls instead of one: roughly double the latency and cost per screening
when the narrative is requested. The narrative is therefore opt-in (`--narrative`).

**Limitations.** The split assumes matching can be expressed as rules. For a requirement needing
genuine semantic judgement ("strong communication skills"), the rules can only return `unknown`.

---

## 5. Deterministic Python for requirement matching

**Problem.** Screening decisions must be reproducible, explainable, and testable, and must not
change because a model was updated or because a CV contained a sentence addressed to the AI.

**Approach.** Pure-Python rules: date intervals with overlap merging for years of experience,
ranked degree levels, whole-token keyword matching with `any`/`all` modes. Three statuses, with
one invariant: **missing information is `unknown`, never `not_met`**; `not_met` requires proof.

**Alternatives considered.** Asking the LLM to score each requirement; embedding similarity
between requirement and CV text; a weighted numeric fit score.

**Rationale.** Rules can be unit-tested against hand-labelled cases (5 cases, 35 labels here),
they produce the same answer every time, and they are immune to prompt injection by
construction. The `unknown` status is the core of the design: treating an absent keyword as a
failed requirement is the standard way automated screening becomes unfair, and the schema makes
that outcome impossible to express by accident.

**Trade-offs.** Keyword matching misses synonyms, so a real match can land in `unknown`. This is
a deliberate direction of error — under-claiming rather than over-claiming — and it is stated in
every output as a caveat. An embedding approach would catch more synonyms while making each
decision unexplainable and unstable.

**Limitations.** Years of experience are counted without judging relevance; the
`irrelevant_experience_candidate` evaluation case exists to pin that behaviour down so the
limitation stays visible. The JD parser expects bulleted *Requirements* / *Nice to have*
sections.

---

## 6. Jupyter Notebook as the demonstration interface

**Problem.** A reviewer must be able to see what each stage does, with intermediate values, and
re-run it. A deployed web app shows the result and hides the pipeline.

**Approach.** `prototype.ipynb` as the primary interface, importing from `src/` with no business
logic duplicated in cells, committed with executed outputs.

**Alternatives considered.** Streamlit (the original assignment description); a FastAPI service;
a CLI only.

**Rationale.** The thing being assessed is engineering reasoning, and a notebook shows the
intermediate state — raw text, raw model output, grounding drops, per-requirement decisions —
which is exactly what a UI hides. It also runs anywhere with no server, and a committed executed
notebook is evidence that the pipeline ran.

**Trade-offs.** Not a product: no concurrency, no auth, no upload UI. Notebook state can go stale
if cells are run out of order, and committed outputs can drift from the code unless the notebook
is re-executed before committing.

**Limitations.** A notebook does not demonstrate operational behaviour under load. The CLI
(`python -m src.pipeline`) exists for scripted and batch use.

---

## 7. Evidence-based recommendations

**Problem.** A recommendation a reviewer cannot verify is worse than no recommendation: it
transfers confidence without transferring the means to check it.

**Approach.** Every extracted item carries verbatim quotes, checked against the CV text; every
requirement assessment carries the facts it used, the quotes behind them, and an
`interpretation` field stating what those facts do *not* show. Provenance records which part came
from a model and which prompt version produced it.

**Alternatives considered.** A numeric fit score; a confidence percentage per field; a ranked
shortlist.

**Rationale.** A reviewer can check a quote in seconds. They cannot check "0.82". Numeric
confidence was deliberately not invented: without a calibration method and data to validate it,
a confidence number is a decoration that makes output look more rigorous than it is — which is
why the assignment's own instruction against arbitrary confidence scores is followed literally.

**Trade-offs.** Output is longer and less immediately scannable than a score. Grounding also
discards genuine items whose quotes were paraphrased, so recall is traded for trustworthiness.

**Limitations.** A quote proves the CV says something, not that it is true. Verification against
reality remains human work.

---

## 8. Keeping a human in the decision loop

**Problem.** Automated rejection at scale is the main harm this kind of system can do, and
"advisory only" written in a README is not a control.

**Approach.** Make it structural. `Recommendation.requires_human_review: Literal[True]` and
`is_final_decision: Literal[False]` cannot hold any other value. The narrative prompt is
forbidden from hire/reject language and the output is filtered in code. Reviewer corrections are
recorded in a separate `ReviewRecord` (`src/review.py`) and never written into the AI output, so
what the system produced stays auditable after a human disagrees with it.

**Alternatives considered.** A documented policy; a confidence threshold for auto-rejection; a
reviewer UI that edits the profile in place.

**Rationale.** Types are enforced; policies are forgotten. Editing in place would be the obvious
convenience and is the one thing that would destroy the audit trail: once a correction is merged
into the profile, nobody can tell what the system actually produced, and every evaluation number
quietly improves each time a human fixes something.

**Trade-offs.** Reviewers must read a side-by-side view instead of a single corrected record,
and downstream consumers have to join two files.

**Limitations.** The code cannot force a human to read anything. It can only ensure the output
never claims to be a decision, and that disagreement is recorded rather than absorbed. Reviewer
corrections are also not yet fed back into evaluation as a second source of ground truth — that
is the obvious next step and is listed under future improvements.
