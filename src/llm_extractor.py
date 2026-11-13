"""cv -> CandidateProfile via gemini (live) or a clearly labelled canned profile (mock)."""

import logging
from typing import Any

from src.gemini_client import build_client, generate_structured, resolve_model
from src.prompts import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_TEMPERATURE,
    render_extraction_prompt,
)
from src.schemas import (
    CallStats,
    CandidateProfile,
    Certification,
    Education,
    ExtractionResult,
    Mode,
    Project,
    ScreeningError,
    Skill,
    WorkExperience,
)
from src.text_utils import normalize_for_match

logger = logging.getLogger(__name__)

MOCK_LABEL = (
    "MOCK MODE: this profile is a hard-coded demo fixture, not the output of any model. "
    "it only matches the bundled synthetic sample cv."
)


class ExtractionError(ScreeningError):
    """codes: missing_api_key, api_error, empty_response, invalid_output."""


# ---------------------------------------------------------------- public api


def extract_profile(
    cv_text: str,
    *,
    mode: Mode,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> ExtractionResult:
    """extract a profile and ground it in the cv text. `client` allows injection in tests.

    `prompt_version` selects an extraction prompt variant; the chosen version is recorded
    on the result so an evaluation run can always say which prompt produced it.
    """
    stats = CallStats()
    if mode == "mock":
        raw = _mock_profile()
        model_name = None
        warnings = [MOCK_LABEL]
    elif mode == "live":
        model_name = resolve_model(model)
        raw = _extract_live(cv_text, model_name, api_key, client, stats, prompt_version)
        warnings = []
    else:
        raise ExtractionError("invalid_mode", f"unknown mode: {mode!r}")

    profile, issues = ground_profile(raw, cv_text)
    return ExtractionResult(
        profile=profile,
        mode=mode,
        model=model_name,
        prompt_version=prompt_version if mode == "live" else None,
        grounding_issues=issues,
        warnings=warnings,
        call_stats=stats,
    )


def ground_profile(profile: CandidateProfile, cv_text: str) -> tuple[CandidateProfile, list[str]]:
    """keep only items whose evidence quotes really occur in the cv text; report the rest."""
    cv_norm = normalize_for_match(cv_text)
    issues: list[str] = []

    def quote_ok(quote: str) -> bool:
        q = normalize_for_match(quote)
        return len(q.strip()) >= 3 and q in cv_norm

    def keep(section: str, items: list[Any]) -> list[Any]:
        kept = []
        for idx, item in enumerate(items):
            verified = [q for q in item.evidence if quote_ok(q)]
            if not verified:
                issues.append(f"{section}[{idx}] dropped: no evidence quote found in the cv text.")
                continue
            if len(verified) < len(item.evidence):
                issues.append(f"{section}[{idx}]: some evidence quotes were not found and were removed.")
            kept.append(item.model_copy(update={"evidence": verified}))
        return kept

    name = profile.full_name
    if name is not None and normalize_for_match(name) not in cv_norm:
        issues.append("full_name dropped: not found in the cv text.")
        name = None

    grounded = CandidateProfile(
        full_name=name,
        education=keep("education", profile.education),
        work_experience=keep("work_experience", profile.work_experience),
        skills=keep("skills", profile.skills),
        projects=keep("projects", profile.projects),
        certifications=keep("certifications", profile.certifications),
    )
    return grounded, issues


# ---------------------------------------------------------------- live mode


def _extract_live(
    cv_text: str, model: str, api_key: str | None, client: Any | None,
    stats: CallStats | None = None,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> CandidateProfile:
    if client is None:
        client = build_client(api_key, ExtractionError)
    return generate_structured(
        client=client,
        model=model,
        render=lambda feedback: render_extraction_prompt(cv_text, feedback, prompt_version),
        schema=CandidateProfile,
        temperature=EXTRACTION_TEMPERATURE,
        error_cls=ExtractionError,
        stats=stats,
    )


# ---------------------------------------------------------------- mock mode


def _mock_profile() -> CandidateProfile:
    """canned profile for samples/synthetic_cv.pdf. quotes are verbatim lines of that cv."""
    skills_line = "Python, SQL, R, Tableau, Microsoft Excel"
    return CandidateProfile(
        full_name="Dewi Contoh",
        education=[
            Education(
                institution="Universitas Contoh",
                degree="Bachelor of Statistics",
                degree_level="bachelor",
                field_of_study="Statistics",
                start_date="2015",
                end_date="2019",
                in_progress=False,
                evidence=["Bachelor of Statistics - Universitas Contoh | 2015 - 2019"],
            )
        ],
        work_experience=[
            WorkExperience(
                company="PT Contoh Analitika",
                title="Data Analyst",
                start_date="2021-01",
                end_date="2024-03",
                is_current=False,
                responsibilities=[
                    "Built SQL-based dashboards in Tableau for weekly sales reporting.",
                    "Automated weekly reports with Python (pandas).",
                    "Forecasted product demand with ARIMA models.",
                ],
                technologies=["SQL", "Tableau", "Python", "pandas", "ARIMA"],
                evidence=[
                    "Data Analyst - PT Contoh Analitika | Jan 2021 - Mar 2024",
                    "Built SQL-based dashboards in Tableau for weekly sales reporting.",
                    "Automated weekly reports with Python (pandas), saving about 6 hours per week.",
                    "Forecasted product demand with ARIMA models.",
                ],
            ),
            WorkExperience(
                company="Universitas Contoh",
                title="Research Assistant",
                start_date="2019",
                end_date="2020",
                is_current=False,
                responsibilities=["Cleaned survey datasets in R and prepared descriptive statistics."],
                technologies=["R"],
                evidence=[
                    "Research Assistant - Universitas Contoh | 2019 - 2020",
                    "Cleaned survey datasets in R and prepared descriptive statistics.",
                ],
            ),
        ],
        skills=[
            Skill(name=name, evidence=[skills_line])
            for name in ["Python", "SQL", "R", "Tableau", "Microsoft Excel"]
        ],
        projects=[
            Project(
                name="Retail Demand Forecasting",
                description="Time series forecasting project comparing ARIMA and Prophet on public retail data.",
                technologies=["ARIMA", "Prophet"],
                evidence=[
                    "Retail Demand Forecasting - Time series forecasting project comparing ARIMA "
                    "and Prophet on public retail data."
                ],
            )
        ],
        certifications=[
            Certification(
                name="Google Data Analytics Certificate",
                issuer="Google",
                date="2022",
                evidence=["Google Data Analytics Certificate (2022)"],
            )
        ],
    )
