# Assignment Alignment

Two things are recorded here: how the implementation maps onto the assignment's evaluation
criteria, and which parts of the original assignment document no longer describe this
repository. The assignment document itself has **not** been edited — the deltas are listed so a
reviewer can see they were noticed rather than ignored.

## 1. Descriptions in the assignment document that are now out of date

The original brief describes a **Streamlit** interface. This repository uses a Jupyter notebook
as the primary demonstration interface and a CLI for scripted use; there is no Streamlit app and
no `app.py`. The reasoning is in `docs/design_decisions.md` §6: the notebook shows the
intermediate state of each pipeline stage, which is what a reviewer needs to see and what a UI
hides.

| assignment document says | this repository does | where |
|---|---|---|
| Streamlit app as the prototype interface | `prototype.ipynb`, executed and committed | §12 of the addendum, `docs/design_decisions.md` §6 |
| "upload a CV in the web UI" | `--cv` path on the CLI, or the notebook's input cell | `src/pipeline.py` |
| "display the result in the app" | per-run JSON artifacts in `outputs/runs/<run_id>/` plus notebook output | `src/artifacts.py` |
| a deployment step | none; the prototype runs locally, by design for a screening-assistance tool handling personal data | `docs/design_decisions.md` §2 |

Nothing else in the brief conflicts with the implementation. If the Streamlit wording is a hard
requirement rather than a description, the pipeline is already UI-agnostic: `run_screening()`
takes bytes and a string and returns a typed result, so a thin Streamlit layer would add no
business logic.

## 2. Evaluation criteria → where each is addressed

| criterion | how it is addressed | evidence |
|---|---|---|
| **Problem relevance** | the system targets the part of screening that is automatable (reading unstructured CVs into a schema) and deliberately leaves judgement to a human | `README.md` §2, `docs/design_decisions.md` §4 |
| **Solution design** | LLM confined to reading and explaining; all decisions deterministic; two verified boundaries around the model | `docs/architecture.md`, `docs/design_decisions.md` §4–5 |
| **Functionality** | end-to-end pipeline, CLI, notebook, per-run artifacts, prompt-version selection | `src/pipeline.py`, `prototype.ipynb`, `outputs/runs/` |
| **Output quality** | typed schemas, verbatim evidence on every item, explicit `unknown`, caveats per result | `src/schemas.py`, `outputs/runs/<run_id>/requirement_matching.json` |
| **Efficiency** | one API call for extraction, a second only when a narrative is requested; per-stage timings and token usage recorded | `src/tracing.py`, `execution_trace.json` |
| **Reliability** | structured error codes for every failure mode, retry policy, 154 tests, error paths measured 3/3 | `tests/`, `docs/evaluation.md` |
| **Human oversight** | final decision unrepresentable in the schema; hire/reject language stripped; corrections recorded without overwriting output | `src/schemas.py`, `src/narrative.py`, `src/review.py` |
| **Evaluation** | three measured evaluation scripts, hand-labelled ground truth, per-case CSV, stated limitations | `docs/evaluation.md`, `outputs/evaluation_results.csv` |

## 3. Schema names

The addendum names six schemas. This repository uses names that read better in code; the
mapping is one-to-one:

| assignment name | this repository | file |
|---|---|---|
| `CandidateExtraction` | `CandidateProfile` (+ `ExtractionResult` wrapper) | `src/schemas.py` |
| `EvidenceReference` | `EvidenceRef` | `src/schemas.py` |
| `JobRequirement` | `JobRequirement` | `src/schemas.py` |
| `RequirementMatch` | `RequirementAssessment` | `src/schemas.py` |
| `ScreeningRecommendation` | `Recommendation` (+ `ScreeningResult` wrapper) | `src/schemas.py` |
| `ExecutionTrace` | `ExecutionTrace` | `src/schemas.py` |

## 4. Confidence scores

The addendum asks that confidence not be represented by arbitrary numbers. No numeric confidence
is produced anywhere in this system. Uncertainty is carried by the three-valued status
(`met` / `not_met` / `unknown`), by `data_gaps`, and by typed `Caveat` entries. Inventing a
calibrated-looking number without a calibration method would make the output look more rigorous
than the evidence supports, which is the failure this rule is guarding against.
