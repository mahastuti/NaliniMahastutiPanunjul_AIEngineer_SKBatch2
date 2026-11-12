"""llm-written explanation of the deterministic matching result, then verified.

the recommendation prompt asks gemini to explain `Recommendation`. prompt instructions
alone are not trusted to keep it honest, so every answer is checked programmatically:

- a status the model restated differently is overwritten with the deterministic one
  and recorded as a violation (the matching engine stays authoritative);
- an evidence quote that is not found in the cv text is dropped;
- scores, rankings and hire/reject language are detected and the offending text removed.

violations are reported, never silently repaired away: they surface as caveats.
"""

import json
import logging
import re
from typing import Any

from src.gemini_client import build_client, generate_structured, resolve_model
from src.prompts import (
    RECOMMENDATION_PROMPT_VERSION,
    RECOMMENDATION_TEMPERATURE,
    render_recommendation_prompt,
)
from src.schemas import (
    CandidateProfile,
    JobDescription,
    Mode,
    NarrativeReport,
    NarrativeResult,
    Recommendation,
    RequirementExplanation,
    ScreeningError,
)
from src.text_utils import normalize_for_match

logger = logging.getLogger(__name__)

MOCK_LABEL = (
    "MOCK MODE: this narrative was generated from a template, not by any model. "
    "it restates the deterministic matching result."
)

# phrases a decision-support narrative must never contain.
# the hiring action itself is out of scope for this text — the narrative describes
# qualifications and evidence — so the decision verbs are banned outright rather than
# only in modal constructions ("should we hire..." must be caught as surely as
# "should be hired"). a false positive costs one stripped sentence and a logged
# violation; a false negative puts a hiring decision in front of the reviewer.
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("hiring_decision", re.compile(
        r"\b(hire|hires|hired|hiring|reject|rejects|rejected|rejecting|"
        r"shortlist\w*|disqualif\w*|accept the candidate|advance the candidate|"
        r"decline the candidate|move forward with)\b", re.I)),
    ("overall_verdict", re.compile(
        r"\b(strong fit|weak fit|good fit|poor fit|not a fit|perfect fit|"
        r"ideal candidate|best candidate|top candidate|highly recommended)\b", re.I)),
    ("numeric_score", re.compile(
        r"\b\d{1,3}\s*(?:%|percent)\b|\b\d+(?:\.\d+)?\s*(?:/|out of)\s*\d+\b|"
        r"\bscore[sd]?\b|\brating\b|\brated\b|\bpoints?\b", re.I)),
    ("ranking", re.compile(
        r"\b(rank(?:s|ed|ing)?|percentile|top \d+|better than (?:other|most))\b", re.I)),
]


class NarrativeError(ScreeningError):
    """codes: missing_api_key, api_error, empty_response, invalid_output."""


# ---------------------------------------------------------------- public api


def generate_narrative(
    profile: CandidateProfile,
    job: JobDescription,
    recommendation: Recommendation,
    cv_text: str,
    *,
    mode: Mode,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> NarrativeResult:
    """produce a verified narrative. `client` allows injection in tests."""
    if mode == "mock":
        raw, model_name = _mock_narrative(recommendation), None
    elif mode == "live":
        model_name = resolve_model(model)
        raw = _generate_live(profile, job, recommendation, model_name, api_key, client)
    else:
        raise NarrativeError("invalid_mode", f"unknown mode: {mode!r}")

    report, violations = verify_narrative(raw, recommendation, cv_text)
    if mode == "mock":
        violations = [MOCK_LABEL, *violations]
    logger.info("narrative generated: mode=%s violations=%d", mode, len(violations))
    return NarrativeResult(
        report=report,
        mode=mode,
        model=model_name,
        prompt_version=RECOMMENDATION_PROMPT_VERSION,
        violations=violations,
    )


def verify_narrative(
    report: NarrativeReport, recommendation: Recommendation, cv_text: str
) -> tuple[NarrativeReport, list[str]]:
    """enforce the prompt's rules in code. returns (corrected_report, violations)."""
    authoritative = {a.requirement_id: a.status for a in recommendation.assessments}
    cv_norm = normalize_for_match(cv_text)
    violations: list[str] = []

    # 1. statuses may not be changed, invented, or dropped
    explanations: list[RequirementExplanation] = []
    seen: set[str] = set()
    for item in report.requirement_explanations:
        rid = item.requirement_id
        if rid not in authoritative:
            violations.append(f"explanation for unknown requirement '{rid}' was removed.")
            continue
        if rid in seen:
            violations.append(f"duplicate explanation for '{rid}' was removed.")
            continue
        seen.add(rid)
        fixed = item
        if item.status_restated != authoritative[rid]:
            violations.append(
                f"status for '{rid}' was restated as '{item.status_restated}' but the matching "
                f"engine says '{authoritative[rid]}'; the deterministic status was restored."
            )
            fixed = fixed.model_copy(update={"status_restated": authoritative[rid]})
        # 2. evidence quotes must occur in the cv
        verified = [q for q in fixed.evidence_quotes if _quote_ok(q, cv_norm)]
        if len(verified) < len(fixed.evidence_quotes):
            violations.append(f"explanation for '{rid}': unverifiable evidence quote(s) removed.")
            fixed = fixed.model_copy(update={"evidence_quotes": verified})
        # 3. no scores or decisions in the prose
        cleaned, found = _strip_banned(fixed.explanation)
        if found:
            violations.append(f"explanation for '{rid}' contained {', '.join(found)}; text removed.")
            fixed = fixed.model_copy(update={"explanation": cleaned})
        explanations.append(fixed)

    missing = [rid for rid in authoritative if rid not in seen]
    if missing:
        violations.append(f"no explanation produced for: {', '.join(missing)}.")

    # 4. gaps may only cite not_met; unknown must not be presented as a weakness
    not_met_ids = {rid for rid, s in authoritative.items() if s == "not_met"}
    gaps, dropped_gaps = _filter_gaps(report.potential_gaps, authoritative, not_met_ids)
    if dropped_gaps:
        violations.append(
            f"{dropped_gaps} item(s) in potential_gaps referred to requirements that are not "
            "'not_met' (missing information treated as a weakness); removed."
        )

    # 5. free-text sections must also be free of scores and decisions
    summary, found = _strip_banned(report.candidate_summary)
    if found:
        violations.append(f"candidate_summary contained {', '.join(found)}; text removed.")
    strengths, n = _clean_list(report.confirmed_strengths)
    if n:
        violations.append(f"{n} item(s) in confirmed_strengths contained banned content; removed.")
    uncertainties, n = _clean_list(report.uncertainties)
    if n:
        violations.append(f"{n} item(s) in uncertainties contained banned content; removed.")
    questions, n = _clean_list(report.reviewer_questions)
    if n:
        violations.append(f"{n} item(s) in reviewer_questions contained banned content; removed.")

    verified_report = NarrativeReport(
        candidate_summary=summary,
        confirmed_strengths=strengths,
        requirement_explanations=explanations,
        potential_gaps=gaps,
        uncertainties=uncertainties,
        reviewer_questions=questions,
    )
    return verified_report, violations


# ---------------------------------------------------------------- helpers


def _quote_ok(quote: str, cv_norm: str) -> bool:
    q = normalize_for_match(quote)
    return len(q.strip()) >= 3 and q in cv_norm


def _strip_banned(text: str) -> tuple[str, list[str]]:
    """remove sentences containing banned content; return (cleaned, kinds_found)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, kinds = [], []
    for sentence in sentences:
        hits = [kind for kind, pattern in _BANNED_PATTERNS if pattern.search(sentence)]
        if hits:
            kinds.extend(hits)
        else:
            kept.append(sentence)
    return " ".join(kept).strip(), sorted(set(kinds))


def _clean_list(items: list[str]) -> tuple[list[str], int]:
    kept = []
    removed = 0
    for item in items:
        cleaned, found = _strip_banned(item)
        if found or not cleaned:
            removed += bool(found)
            if cleaned:
                kept.append(cleaned)
            continue
        kept.append(cleaned)
    return kept, removed


def _filter_gaps(
    gaps: list[str], authoritative: dict[str, str], not_met_ids: set[str]
) -> tuple[list[str], int]:
    """drop gap entries that name a requirement whose status is not 'not_met'."""
    kept, dropped = [], 0
    for gap in gaps:
        mentioned = {rid for rid in authoritative if re.search(rf"\b{re.escape(rid)}\b", gap)}
        if mentioned and not (mentioned & not_met_ids):
            dropped += 1
            continue
        kept.append(gap)
    return kept, dropped


# ---------------------------------------------------------------- live mode


def _generate_live(
    profile: CandidateProfile,
    job: JobDescription,
    recommendation: Recommendation,
    model: str,
    api_key: str | None,
    client: Any | None,
) -> NarrativeReport:
    if client is None:
        client = build_client(api_key, NarrativeError)
    candidate_json = profile.model_dump_json(indent=2)
    requirements_json = json.dumps(
        [{"id": r.id, "text": r.text, "mandatory": r.mandatory} for r in job.requirements], indent=2
    )
    matching_json = json.dumps(
        [
            {
                "requirement_id": a.requirement_id,
                "status": a.status,
                "mandatory": a.mandatory,
                "facts": a.facts,
                "interpretation": a.interpretation,
                "evidence": [e.quote for e in a.evidence],
            }
            for a in recommendation.assessments
        ],
        indent=2,
    )
    return generate_structured(
        client=client,
        model=model,
        render=lambda feedback: render_recommendation_prompt(
            candidate_json, requirements_json, matching_json, feedback
        ),
        schema=NarrativeReport,
        temperature=RECOMMENDATION_TEMPERATURE,
        error_cls=NarrativeError,
    )


# ---------------------------------------------------------------- mock mode


def _mock_narrative(recommendation: Recommendation) -> NarrativeReport:
    """template narrative built from the deterministic result. no model is called."""
    by_status = {
        s: [a for a in recommendation.assessments if a.status == s]
        for s in ("met", "not_met", "unknown")
    }
    return NarrativeReport(
        candidate_summary=(
            "Template summary generated without a model. The candidate's stated background is "
            "described in the extracted profile; no overall judgement is made here."
        ),
        confirmed_strengths=[
            f"{a.requirement_id}: {a.requirement_text} - supported by: {a.facts[0] if a.facts else 'evidence in the cv'}"
            for a in by_status["met"]
        ],
        requirement_explanations=[
            RequirementExplanation(
                requirement_id=a.requirement_id,
                status_restated=a.status,
                explanation=f"{a.facts[0] if a.facts else 'no facts recorded.'} {a.interpretation}",
                evidence_quotes=[e.quote for e in a.evidence[:2]],
            )
            for a in recommendation.assessments
        ],
        potential_gaps=[
            f"{a.requirement_id}: {a.requirement_text} - {a.facts[-1] if a.facts else 'shortfall recorded by the matching engine.'}"
            for a in by_status["not_met"]
        ],
        uncertainties=[
            f"{a.requirement_id}: {a.requirement_text} - the cv does not provide enough information to decide."
            for a in by_status["unknown"]
        ],
        reviewer_questions=[
            f"Ask the candidate to clarify: {a.requirement_text}" for a in by_status["unknown"][:5]
        ],
    )
