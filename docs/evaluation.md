# Evaluation

System-level evaluation. Prompt-level evaluation (construction, injection containment, narrative
rule enforcement, live extraction accuracy) is in `docs/prompts.md` and
`scripts/evaluate_prompts.py`; the two are kept separate so nothing is measured twice.

Run: `python scripts/evaluate.py --mode mock` (default) or `--mode live` (needs `GEMINI_API_KEY`).
Machine-readable results: `outputs/evaluation_report.json`. The evaluation date is fixed (2026-11-12) so duration-based results are reproducible.

## What is measured, and what deliberately is not

| Dimension | How | Where |
|---|---|---|
| Matching quality | 4 hand-labeled candidate/JD cases, 28 requirement labels | `evaluate.py` |
| Validation reliability | 6 simulated LLM outputs with known defects | `evaluate.py` |
| Hallucination containment | fabricated items planted in simulated outputs | `evaluate.py` |
| System reliability | end-to-end runs on 4 synthetic CVs + 3 error paths | `evaluate.py` |
| Prompt construction & injection containment | 14 checks on rendered prompts | `evaluate_prompts.py` |
| Narrative rule enforcement | 6 simulated narratives, 11 planted violations | `evaluate_prompts.py` |
| Extraction quality | field-level comparison vs hand-written ground truth | `evaluate_prompts.py --mode live` |

Extraction quality is **not** measured in mock mode on purpose: the mock profile is a fixture written to match the sample CV, so scoring it against ground truth would be circular and would overstate quality. Both reports say so explicitly instead of printing a misleading number.

## Measured results (mock mode, development environment, 2026-11-12)

All numbers below are actual outputs of `scripts/evaluate.py`, not estimates.

**Matching quality** — requirement-level accuracy **1.000** (28/28 labels, 4/4 cases exact).

| status | precision | recall | support |
|---|---|---|---|
| met | 1.0 | 1.0 | 11 |
| not_met | 1.0 | 1.0 | 2 |
| unknown | 1.0 | 1.0 | 15 |

The cases cover: a fully matching candidate, a partial candidate (proven year/degree shortfalls → `not_met`, absent evidence → `unknown`), an information-poor candidate (everything `unknown`, signal `insufficient_information`), and a candidate whose undated role correctly blocks a `not_met` on years.

**Validation reliability** — **6/6** fixtures decided correctly: faithful and defective-but-well-formed outputs accepted; malformed JSON, smuggled protected attributes (`age`, `gender`), and wrong field types all rejected by the schema.

**Hallucination containment** — **4/4** fabricated items dropped by the grounding layer (a skill with invented evidence, a skill with no evidence, an invented certification, and a real role whose evidence was paraphrased rather than quoted — unverifiable counts as unsupported); **9/9** genuine items kept; every drop reported as a grounding issue. A wrong `full_name` was also dropped.

**System reliability** — end-to-end success **4/4** CVs (mock mode; timings exclude real API latency). On the ambiguous, sparse and injection CVs the mock profile is grounded away, yielding an empty profile, all-`unknown` assessments, and `insufficient_information` — the designed honest behavior, not a crash. Error paths **3/3**: non-PDF bytes, corrupt PDF, and empty upload each produced the expected structured error code.

**Prompt level** (summary; full detail in `docs/prompts.md`) — construction checks **14/14**, planted narrative violations detected **11/11** with **0** false alarms, status integrity preserved **6/6**.

## Limitations of this evaluation

1. **Live Gemini extraction is unverified.** The biggest gap. `evaluate_prompts.py --mode live` computes name accuracy, skill/company/certification precision+recall, degree-level accuracy and injection resistance against `data/evaluation/extraction_ground_truth.json`, but it has not been run (no API key in the development environment). Until then, nothing here says how well Gemini actually extracts or how often it obeys the prompts.
2. **Tiny, author-written dataset.** 4 matching cases, 6 simulated outputs, 3 synthetic CVs, all written by the author of the rules. Perfect scores mean the implementation matches its own specification on covered scenarios (regression tests), not that the system generalises.
3. **Simulated LLM defects ≠ real defect distribution.** The fixtures test this system's defenses against representative failure types; real Gemini errors may differ.
4. **Response time in mock mode excludes the dominant cost** (the API call).
