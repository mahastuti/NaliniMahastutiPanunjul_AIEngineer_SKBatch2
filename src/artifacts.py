"""write one run's results to disk as separate, purpose-built json files.

every file here is produced from an actual pipeline execution; nothing is hand-written.
each run gets its own directory named after its run id, so a new run never overwrites an
older one and a result can always be traced back to the trace that produced it.

the four artifacts split the result by audience:

| file                        | answers                                               |
|-----------------------------|-------------------------------------------------------|
| candidate_extraction.json   | what was read out of the cv, and what was discarded    |
| requirement_matching.json   | how each requirement was decided, with evidence        |
| screening_report.json       | what the reviewer should look at, with caveats         |
| execution_trace.json        | what the system did, how long it took, what it cost    |
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.schemas import ExecutionTrace, ScreeningResult

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("outputs/runs")

CANDIDATE_FILE = "candidate_extraction.json"
MATCHING_FILE = "requirement_matching.json"
REPORT_FILE = "screening_report.json"
TRACE_FILE = "execution_trace.json"


def _write(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_dir(run_id: str, root: Path | None = None) -> Path:
    return (root or DEFAULT_RUNS_DIR) / run_id


def build_candidate_extraction(result: ScreeningResult) -> dict:
    """validated profile plus the provenance needed to judge how much to trust it."""
    extraction_caveats = [
        c.model_dump()
        for c in result.recommendation.caveats
        if c.code in {"grounding_issue", "llm_extraction_unverified", "mock_mode",
                      "possible_prompt_injection"}
    ]
    return {
        "run_id": result.trace.run_id,
        "mode": result.mode,
        "provenance": result.provenance.model_dump(),
        "profile": result.profile.model_dump(),
        "extraction_caveats": extraction_caveats,
        "notice": (
            "every item above carries verbatim evidence quotes that were found in the cv "
            "text. items whose quotes could not be found were dropped before this file was "
            "written and are listed in extraction_caveats."
        ),
    }


def build_requirement_matching(result: ScreeningResult) -> dict:
    """one row per requirement: id, description, category, status, evidence, explanation."""
    kinds = {r.id: r.kind for r in result.job.requirements}
    narrative_by_id = {
        e.requirement_id: e.explanation for e in (result.narrative.requirement_explanations
                                                  if result.narrative else [])
    }
    rows = []
    for a in result.recommendation.assessments:
        rows.append({
            "requirement_id": a.requirement_id,
            "requirement_text": a.requirement_text,
            "category": kinds.get(a.requirement_id, "other"),
            "mandatory": a.mandatory,
            "status": a.status,
            "supporting_evidence": [e.model_dump() for e in a.evidence],
            "facts": a.facts,
            "explanation": a.interpretation,
            "narrative_explanation": narrative_by_id.get(a.requirement_id),
            "caveat": (
                "status 'unknown' means the cv did not provide enough information; it is not "
                "evidence that the candidate lacks this qualification."
                if a.status == "unknown" else None
            ),
        })
    return {
        "run_id": result.trace.run_id,
        "job_title": result.job.title,
        "matching_source": result.provenance.matching_source,
        "mandatory_counts": result.recommendation.mandatory_counts.model_dump(),
        "optional_counts": result.recommendation.optional_counts.model_dump(),
        "requirements": rows,
        "data_gaps": result.recommendation.data_gaps,
        "notice": (
            "statuses are computed by deterministic python rules, never by a model. the "
            "narrative_explanation field, when present, is model-written prose that was "
            "verified against these statuses and cannot change them."
        ),
    }


def build_screening_report(result: ScreeningResult) -> dict:
    """the reviewer-facing summary: assessment, caveats and human-review indicators."""
    return {
        "run_id": result.trace.run_id,
        "mode": result.mode,
        "provenance": result.provenance.model_dump(),
        "signal": result.recommendation.signal,
        "summary": result.recommendation.summary,
        "recommendation": result.recommendation.model_dump(),
        "narrative": result.narrative.model_dump() if result.narrative else None,
        "human_review": {
            "requires_human_review": result.recommendation.requires_human_review,
            "is_final_decision": result.recommendation.is_final_decision,
            "decision_notice": result.recommendation.decision_notice,
            "open_questions": [
                a.requirement_id
                for a in result.recommendation.assessments
                if a.status == "unknown"
            ],
            "review_record_file": "review_record.json (written by src/review.py once a human reviews)",
        },
        "notices": result.notices,
    }


def write_run_artifacts(result: ScreeningResult, root: Path | None = None) -> Path:
    """write all four artifacts for a successful run. returns the run directory."""
    directory = run_dir(result.trace.run_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / CANDIDATE_FILE, build_candidate_extraction(result))
    _write(directory / MATCHING_FILE, build_requirement_matching(result))
    _write(directory / REPORT_FILE, build_screening_report(result))
    _write(directory / TRACE_FILE, result.trace.model_dump())
    logger.info("artifacts written: %s", directory)
    return directory


def write_failed_run(
    trace: ExecutionTrace, error_code: str, message: str, root: Path | None = None
) -> Path:
    """a run that failed still produces its trace, so failures are as inspectable as
    successes. no result files are written, because there is no result — the pipeline
    never fabricates one."""
    directory = run_dir(trace.run_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    payload = trace.model_dump()
    payload["error"] = {"code": error_code, "message": message}
    _write(directory / TRACE_FILE, payload)
    logger.info("failed run traced: %s (%s)", directory, error_code)
    return directory
