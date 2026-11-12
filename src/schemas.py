"""pydantic schemas shared by every module in the screening pipeline."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["live", "mock"]
DegreeLevel = Literal["high_school", "diploma", "bachelor", "master", "doctorate", "other"]
RequirementStatus = Literal["met", "not_met", "unknown"]

_DATE_RE = re.compile(r"^\d{4}(-(0[1-9]|1[0-2]))?$")


class ScreeningError(Exception):
    """base error with a machine-readable code, so callers never parse messages."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _check_date(value: str | None) -> str | None:
    if value is not None and not _DATE_RE.match(value):
        raise ValueError("date must be YYYY-MM or YYYY")
    return value


# ---------------------------------------------------------------- extraction
# these models are what the LLM must fill. no protected attributes exist here
# on purpose: age, gender, religion, photo, address etc. have no field.


class Education(_Strict):
    institution: str | None
    degree: str | None
    degree_level: DegreeLevel | None
    field_of_study: str | None
    start_date: str | None
    end_date: str | None
    in_progress: bool | None
    evidence: list[str]

    _dates = field_validator("start_date", "end_date")(_check_date)


class WorkExperience(_Strict):
    company: str | None
    title: str | None
    start_date: str | None
    end_date: str | None
    is_current: bool | None
    responsibilities: list[str]
    technologies: list[str]
    evidence: list[str]

    _dates = field_validator("start_date", "end_date")(_check_date)


class Skill(_Strict):
    name: str
    evidence: list[str]


class Project(_Strict):
    name: str | None
    description: str | None
    technologies: list[str]
    evidence: list[str]


class Certification(_Strict):
    name: str
    issuer: str | None
    date: str | None
    evidence: list[str]

    _dates = field_validator("date")(_check_date)


class CandidateProfile(_Strict):
    full_name: str | None
    education: list[Education]
    work_experience: list[WorkExperience]
    skills: list[Skill]
    projects: list[Project]
    certifications: list[Certification]


# ---------------------------------------------------------------- job side


class JobRequirement(_Strict):
    id: str
    text: str
    mandatory: bool
    kind: Literal["min_years", "education_level", "skills", "other"]
    min_years: float | None = None
    education_level: DegreeLevel | None = None
    terms: list[str] = Field(default_factory=list)
    term_mode: Literal["all", "any"] = "all"
    certification_only: bool = False


class JobDescription(_Strict):
    title: str | None
    requirements: list[JobRequirement]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- recommendation


class EvidenceRef(_Strict):
    section: str
    quote: str


class Caveat(_Strict):
    code: str
    severity: Literal["info", "warning"]
    message: str


class RequirementAssessment(_Strict):
    requirement_id: str
    requirement_text: str
    mandatory: bool
    status: RequirementStatus
    facts: list[str]  # things verifiable in the extracted, grounded data
    interpretation: str  # what the facts do and do not show
    evidence: list[EvidenceRef]


class RequirementCounts(_Strict):
    met: int
    not_met: int
    unknown: int


class Recommendation(_Strict):
    signal: Literal[
        "strong_support", "partial_support", "mandatory_gap", "insufficient_information"
    ]
    summary: str
    mandatory_counts: RequirementCounts
    optional_counts: RequirementCounts
    assessments: list[RequirementAssessment]
    data_gaps: list[str]
    caveats: list[Caveat]
    # typed so a final automated decision cannot be represented at all
    requires_human_review: Literal[True] = True
    is_final_decision: Literal[False] = False
    decision_notice: str = (
        "this is decision support only. a human reviewer must verify the extracted "
        "data against the original cv and make the hiring decision."
    )


# ---------------------------------------------------------------- pipeline


class ExtractionResult(_Strict):
    profile: CandidateProfile
    mode: Mode
    model: str | None
    grounding_issues: list[str]
    warnings: list[str]


class Provenance(_Strict):
    profile_source: Literal["gemini_llm", "mock_canned"]
    model: str | None
    grounded_in_cv_text: Literal[True] = True
    human_verified: Literal[False] = False


class ScreeningResult(_Strict):
    mode: Mode
    provenance: Provenance
    job: JobDescription
    profile: CandidateProfile
    recommendation: Recommendation
    notices: list[str]
