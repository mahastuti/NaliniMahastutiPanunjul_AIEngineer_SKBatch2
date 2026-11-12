# Prompt Engineering

Both prompts live in `src/prompts.py`, separate from business logic: no module there imports
matching or pipeline code, and no prompt text appears anywhere else in `src/`.

| version | purpose | input variables | structured output | temperature |
|---|---|---|---|---|
| `extraction-v1` | raw CV text → structured candidate information | `cv_text`, `validation_feedback` | `CandidateProfile` | 0.0 |
| `recommendation-v1` | profile + requirements + **deterministic matching results** → reviewer narrative | `candidate_json`, `requirements_json`, `matching_json`, `validation_feedback` | `NarrativeReport` | 0.0 |

`PROMPT_REGISTRY` records this metadata so an evaluation run can state which prompt version
produced it; the version is also written into every pipeline result's `provenance`.
Temperature is 0.0 for both — low randomness is required for reproducible evaluation.

## Division of labour

Gemini is used only where language understanding is needed:

- **unstructured reading** (extraction-v1): turning free-text CVs into a schema;
- **evidence-based explanation** (recommendation-v1): turning an already-computed result into prose.

Everything that decides anything — requirement matching, statuses, durations, degree levels — is
deterministic Python in `src/recommendation.py`. The LLM never computes a status, and the
narrative prompt receives the statuses as *input*, not as something to produce.

## 1. Extraction prompt (`extraction-v1`)

Rules encoded in the system instruction: extract only what the CV states; `null`/`[]` for
anything absent; verbatim evidence quotes for every item (an item whose quotes cannot be found
is discarded — the prompt says so, and the code enforces it); `YYYY[-MM]` dates with no
invention from vague phrases; ambiguity extracted as written rather than resolved by guessing;
no protected attributes; no evaluation or scoring; CV content is untrusted data whose embedded
instructions must be ignored.

Enforcement that does not rely on the prompt:

| prompt rule | programmatic enforcement |
|---|---|
| return valid schema | cleaned JSON schema of `CandidateProfile` as `response_schema`, then Pydantic validation, one retry with feedback |
| no protected attributes | `extra="forbid"` — `age`, `gender`, … make the response invalid |
| valid dates | `field_validator` on every date field |
| verbatim evidence | `ground_profile()` drops items whose quotes are absent from the CV text |
| no invented qualifications | same grounding layer; drops are reported as caveats |
| ignore CV instructions | delimiter block with the closing tag stripped from the content; matching is deterministic anyway |

## 2. Recommendation prompt (`recommendation-v1`)

Inputs: the validated and grounded candidate JSON, the job requirements, and the deterministic
matching results. Output: `candidate_summary`, `confirmed_strengths`, `requirement_explanations`
(status restated per requirement + explanation + evidence quotes), `potential_gaps`,
`uncertainties`, `reviewer_questions`.

Rules: never change, add, or remove a status; quote evidence verbatim from the data given;
describe `unknown` as missing information and never as a weakness; no scores, ratings,
percentages, rankings, or hire/reject recommendations; no personal attributes.

Enforcement in `verify_narrative()` (`src/narrative.py`), applied to every answer:

| prompt rule | programmatic enforcement |
|---|---|
| restate statuses exactly | a differing status is **overwritten** with the engine's and logged as a violation |
| no invented requirements | explanations for unknown IDs are removed; duplicates removed; missing IDs reported |
| verbatim evidence | quotes absent from the CV text are dropped |
| no scores / rankings / decisions | sentences matching banned patterns are stripped from prose and lists |
| `unknown` is not a weakness | `potential_gaps` entries naming a requirement that is not `not_met` are removed |

Violations are surfaced as `narrative_violation` caveats on the result, never silently repaired.
The banned-content patterns are deliberately conservative: the hiring verbs are banned outright
rather than only in modal constructions, because a false positive costs one stripped sentence
while a false negative puts a hiring decision in front of the reviewer. This was not theoretical
— an early version only matched "should be hired" and let "Should we hire this candidate?"
through; a test caught it and the pattern was widened.

## Measured results

From `python scripts/evaluate_prompts.py --mode mock` (offline, this environment):

- **A. Prompt construction: 14/14 checks passed.** Input variables rendered, exactly one data
  block per untrusted input, injection probes contained inside the block, retry feedback
  injected, required rules present in both system prompts.
- **B. Extraction contract: 6/6** schema decisions correct on simulated outputs;
  **4/4** unsupported claims blocked; **9/9** genuine items retained.
- **C. Recommendation contract: 11/11** planted violations detected, **0** false alarms on the
  compliant fixture, **6/6** fixtures preserved status integrity after verification.
- **D. Live model behaviour: not measured** (no API key).

## Limitations and observed errors

0. **Observed error (first live run): HTTP 400 `Unknown name "additional_properties"`.** Passing
   the Pydantic class directly as `response_schema` made the SDK send `additional_properties`
   (generated by `extra="forbid"`), which the Gemini API rejects. Fixed by `response_schema_for()`
   in `src/gemini_client.py`, which sends a cleaned JSON schema; strictness is unchanged because
   the response is still validated by the Pydantic model. Guarded by `tests/test_gemini_client.py`,
   which captures the request body offline. Also observed: a transient `503 UNAVAILABLE` (high
   demand) from the API — rerunning is the remedy, the client retries once. Model names are also
   retired over time (`gemini-2.5-flash` returned 404 for a new account), hence `GEMINI_MODEL`.
1. **Model compliance is unmeasured.** Offline sections measure the templates and the
   enforcement layer. How often Gemini actually obeys the prompts is unknown until
   `evaluate_prompts.py --mode live` is run with a key.
2. **Simulated outputs are hand-written** representative failures, not a sample of real model
   errors; the real distribution may differ.
3. **Injection probes are a small fixed set.** Containment here is not proof of robustness; the
   real protection is that matching is deterministic, so a successful injection could at worst
   corrupt extraction — and grounding then drops anything not in the CV.
4. **Prose filtering is pattern-based** and could miss a novel phrasing of a verdict. The
   status-integrity and evidence guarantees do not depend on it.
5. **Prompts and ground truth share an author**, so live scores will be indicative, not independent.
