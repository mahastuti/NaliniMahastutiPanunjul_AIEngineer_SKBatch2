"""evidence-based requirement matching. deterministic: no llm call, no hiring decision.

core rule: missing information is `unknown`, never `not_met`. `not_met` is only issued when
the extracted data is complete enough to prove the shortfall (years, degree level).
"""

import re
from dataclasses import dataclass
from datetime import date

from src.schemas import (
    CandidateProfile,
    Caveat,
    Education,
    EvidenceRef,
    JobDescription,
    JobRequirement,
    Recommendation,
    RequirementAssessment,
    RequirementCounts,
    WorkExperience,
)
from src.text_utils import contains_term, normalize_for_match

_LEVEL_RANK = {"high_school": 1, "diploma": 2, "bachelor": 3, "master": 4, "doctorate": 5}

# ---------------------------------------------------------------- job description parsing

_MANDATORY_HEADING = re.compile(r"^(requirements?|qualifications?|must[- ]have|required|what you.?ll need)\b", re.I)
_OPTIONAL_HEADING = re.compile(r"^(nice[- ]to[- ]have|preferred|bonus|good to have|plus|optional)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|tahun)\b", re.I)
_EDU_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("doctorate", re.compile(r"\b(ph\.?d|doctorate|doctoral|s3)\b", re.I)),
    ("master", re.compile(r"\b(master'?s?|m\.?sc|s2|magister)\b", re.I)),
    ("bachelor", re.compile(r"\b(bachelor'?s?|b\.?sc|s1|sarjana|undergraduate degree)\b", re.I)),
    ("diploma", re.compile(r"\b(diploma|d3|d4)\b", re.I)),
    ("high_school", re.compile(r"\b(high school|sma|smk)\b", re.I)),
]
_PREFIX = re.compile(
    r"^(?:(?:strong|solid|good|proven|hands-on|working)\s+)*"
    r"(?:(?:experience|proficiency|knowledge|familiarity|expertise)\s+(?:with|in|of|using|on)\s+"
    r"|(?:experience|proficiency|knowledge|familiarity|expertise)\s+)",
    re.I,
)
_SUFFIX = re.compile(
    r"\s+(?:proficiency|experience|skills?|knowledge|expertise|background|certifications?|certificates?)$", re.I
)
_SPLIT = re.compile(r"\s*(?:,|/|;|&|\band\b|\bor\b|\bwith\b)\s*", re.I)


def parse_job_description(text: str) -> JobDescription:
    """turn bullet lists under requirement headings into structured requirements."""
    title: str | None = None
    section: bool | None = None  # True mandatory, False optional, None ignored
    found_heading = False
    bullets: list[tuple[str, bool | None]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(?:job title|title|position)\s*:\s*(.+)$", stripped, re.I)
        if m:
            title = m.group(1).strip()
            continue
        bullet = _BULLET.match(line)
        if bullet:
            bullets.append((bullet.group(1).strip(), section))
            continue
        # short non-bullet lines are treated as headings
        if len(stripped) <= 50 and not stripped.endswith("."):
            heading = stripped.rstrip(":").strip()
            if _MANDATORY_HEADING.match(heading):
                section, found_heading = True, True
            elif _OPTIONAL_HEADING.match(heading):
                section, found_heading = False, True
            else:
                section = None

    warnings: list[str] = []
    if bullets and not found_heading:
        warnings.append("no 'Requirements' heading found; every bullet was treated as mandatory.")
        bullets = [(b, True) for b, _ in bullets]

    requirements = [
        _parse_requirement(f"R{i}", btext, bool(mandatory))
        for i, (btext, mandatory) in enumerate(((b, s) for b, s in bullets if s is not None), start=1)
    ]
    if not requirements:
        warnings.append("no requirement bullets could be parsed from the job description.")
    return JobDescription(title=title, requirements=requirements, warnings=warnings)


def _parse_requirement(rid: str, text: str, mandatory: bool) -> JobRequirement:
    years = _YEARS.search(text)
    if years:
        return JobRequirement(id=rid, text=text, mandatory=mandatory, kind="min_years", min_years=float(years.group(1)))
    for level, pattern in _EDU_PATTERNS:
        if pattern.search(text):
            return JobRequirement(id=rid, text=text, mandatory=mandatory, kind="education_level", education_level=level)  # type: ignore[arg-type]

    certification_only = bool(re.search(r"\bcertif", text, re.I))
    cleaned = text.strip().rstrip(".;")
    cleaned = _PREFIX.sub("", cleaned)
    cleaned = _SUFFIX.sub("", cleaned)
    mode = "any" if re.search(r"\bor\b|/", cleaned, re.I) else "all"
    terms = [t.strip() for t in _SPLIT.split(cleaned) if t.strip()][:8]
    if not terms:
        return JobRequirement(id=rid, text=text, mandatory=mandatory, kind="other")
    return JobRequirement(
        id=rid, text=text, mandatory=mandatory, kind="skills",
        terms=terms, term_mode=mode, certification_only=certification_only,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- matching helpers


@dataclass(frozen=True)
class _Source:
    section: str
    label: str
    text_norm: str
    evidence: list[str]


def _sources(profile: CandidateProfile, certification_only: bool) -> list[_Source]:
    out: list[_Source] = []
    for c in profile.certifications:
        out.append(_Source("certifications", c.name, normalize_for_match(f"{c.name} {c.issuer or ''}"), c.evidence))
    if certification_only:
        return out
    for s in profile.skills:
        out.append(_Source("skills", s.name, normalize_for_match(s.name), s.evidence))
    for w in profile.work_experience:
        blob = " ".join([w.title or "", *w.responsibilities, *w.technologies])
        out.append(_Source("work_experience", f"{w.title or 'role'} at {w.company or 'unknown company'}", normalize_for_match(blob), w.evidence))
    for p in profile.projects:
        blob = " ".join([p.name or "", p.description or "", *p.technologies])
        out.append(_Source("projects", p.name or "project", normalize_for_match(blob), p.evidence))
    for e in profile.education:
        out.append(_Source("education", e.degree or "education", normalize_for_match(f"{e.degree or ''} {e.field_of_study or ''}"), e.evidence))
    return out


def _refs(sources: list[_Source], limit: int = 3) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for s in sources:
        for q in s.evidence[:2]:
            ref = EvidenceRef(section=s.section, quote=q)
            if ref not in refs:
                refs.append(ref)
    return refs[:limit]


def _month_index(value: str, *, end: bool, optimistic: bool) -> int:
    """month index for YYYY or YYYY-MM. year-only dates resolve to the pessimistic or optimistic edge."""
    year, _, month = value.partition("-")
    if month:
        return int(year) * 12 + int(month)
    # year-only: start is latest (dec) / end is earliest (jan) when pessimistic
    if end:
        return int(year) * 12 + (12 if optimistic else 1)
    return int(year) * 12 + (1 if optimistic else 12)


def _role_interval(w: WorkExperience, today: date, optimistic: bool) -> tuple[int, int] | None:
    if w.start_date is None:
        return None
    if w.is_current:
        now = today.year * 12 + today.month
        return _month_index(w.start_date, end=False, optimistic=optimistic), now
    if w.end_date is None:
        return None
    return (
        _month_index(w.start_date, end=False, optimistic=optimistic),
        _month_index(w.end_date, end=True, optimistic=optimistic),
    )


def _union_months(intervals: list[tuple[int, int]]) -> int:
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start  # type: ignore[operator]
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start  # type: ignore[operator]
    return total


def experience_months(profile: CandidateProfile, today: date) -> tuple[int, int, int]:
    """return (lower_bound_months, upper_bound_months, undated_role_count), overlaps merged."""
    lower: list[tuple[int, int]] = []
    upper: list[tuple[int, int]] = []
    undated = 0
    for w in profile.work_experience:
        lo = _role_interval(w, today, optimistic=False)
        hi = _role_interval(w, today, optimistic=True)
        if lo is None or hi is None:
            undated += 1
            continue
        lower.append(lo)
        upper.append(hi)
    return _union_months(lower), _union_months(upper), undated


def _is_completed(e: Education, today: date) -> bool:
    if e.in_progress is True:
        return False
    if e.in_progress is False:
        return True
    if e.end_date is None:
        return False
    return _month_index(e.end_date, end=True, optimistic=False) <= today.year * 12 + today.month


# ---------------------------------------------------------------- per-requirement assessment


def _assess(req: JobRequirement, profile: CandidateProfile, today: date) -> RequirementAssessment:
    if req.kind == "min_years" and req.min_years is not None:
        return _assess_years(req, profile, today)
    if req.kind == "education_level" and req.education_level is not None:
        return _assess_education(req, profile, today)
    if req.kind == "skills":
        return _assess_terms(req, profile)
    return _result(
        req, "unknown", [], "this requirement cannot be assessed automatically; the reviewer must judge it.", []
    )


def _result(
    req: JobRequirement, status: str, facts: list[str], interpretation: str, evidence: list[EvidenceRef]
) -> RequirementAssessment:
    return RequirementAssessment(
        requirement_id=req.id, requirement_text=req.text, mandatory=req.mandatory,
        status=status, facts=facts, interpretation=interpretation, evidence=evidence,  # type: ignore[arg-type]
    )


def _assess_years(req: JobRequirement, profile: CandidateProfile, today: date) -> RequirementAssessment:
    assert req.min_years is not None
    need = req.min_years * 12
    lower, upper, undated = experience_months(profile, today)
    roles = profile.work_experience
    evidence = [EvidenceRef(section="work_experience", quote=q) for w in roles for q in w.evidence[:1]][:3]

    if not roles:
        return _result(req, "unknown", ["no work experience was extracted from the cv."],
                       "absence of extracted roles is missing information, not proof of no experience.", [])
    facts = [f"dated roles add up to at least {lower / 12:.1f} years (overlaps merged, year-only dates counted conservatively)."]
    if undated:
        facts.append(f"{undated} role(s) have no usable start/end date and are not counted.")
    note = "counts all professional roles; whether they are relevant to this job is not assessed."

    if lower >= need:
        return _result(req, "met", facts, note, evidence)
    if upper < need and undated == 0:
        facts.append(f"even with the most generous reading of the dates the total is at most {upper / 12:.1f} years.")
        return _result(req, "not_met", facts, note + " based only on the roles extracted; verify nothing was missed.", evidence)
    return _result(req, "unknown", facts,
                   "dates are too incomplete or approximate to decide whether the threshold is reached.", evidence)


def _assess_education(req: JobRequirement, profile: CandidateProfile, today: date) -> RequirementAssessment:
    assert req.education_level is not None
    need = _LEVEL_RANK[req.education_level]
    entries = profile.education
    if not entries:
        return _result(req, "unknown", ["no education was extracted from the cv."],
                       "absence of extracted education is missing information, not proof of no degree.", [])

    def rank(e: Education) -> int | None:
        return _LEVEL_RANK.get(e.degree_level) if e.degree_level else None

    qualifying = [e for e in entries if (rank(e) or 0) >= need]
    done = [e for e in qualifying if _is_completed(e, today)]
    note = "only the degree level is compared; field of study and institution are not assessed."
    if done:
        e = done[0]
        return _result(req, "met", [f"'{e.degree}' ({e.degree_level}) is listed as completed."], note,
                       [EvidenceRef(section="education", quote=q) for q in e.evidence[:2]])
    if qualifying:
        e = qualifying[0]
        return _result(req, "unknown", [f"'{e.degree}' is listed but completion is not clearly stated."],
                       "the reviewer should confirm the degree was completed.",
                       [EvidenceRef(section="education", quote=q) for q in e.evidence[:2]])
    if any(rank(e) is None for e in entries):
        return _result(req, "unknown", ["at least one education entry has no recognisable degree level."],
                       "the level could not be established from the cv text.", [])
    best = max(entries, key=lambda e: rank(e) or 0)
    return _result(req, "not_met", [f"highest listed level is '{best.degree_level}', below the required '{req.education_level}'."],
                   note + " based only on the entries extracted; verify nothing was missed.",
                   [EvidenceRef(section="education", quote=q) for q in best.evidence[:2]])


def _assess_terms(req: JobRequirement, profile: CandidateProfile) -> RequirementAssessment:
    sources = _sources(profile, req.certification_only)
    found: dict[str, list[_Source]] = {t: [s for s in sources if contains_term(s.text_norm, t)] for t in req.terms}
    hit = [t for t, srcs in found.items() if srcs]
    miss = [t for t in req.terms if t not in hit]
    facts = [f"'{t}' appears in: " + ", ".join(sorted({f'{s.section} ({s.label})' for s in found[t]})) + "." for t in hit]
    evidence = _refs([s for t in hit for s in found[t]])
    scope = "certifications only" if req.certification_only else "the extracted cv data"

    satisfied = bool(hit) if req.term_mode == "any" else not miss
    if satisfied:
        return _result(req, "met", facts,
                       "a mention shows the term is claimed in the cv, not the depth of proficiency.", evidence)
    facts.append(f"not found in {scope}: " + ", ".join(f"'{t}'" for t in miss) + ".")
    return _result(req, "unknown", facts,
                   "not finding a term is not evidence the candidate lacks it: wording may differ or the cv may omit it.",
                   evidence)


# ---------------------------------------------------------------- data gaps and caveats


def find_data_gaps(profile: CandidateProfile) -> list[str]:
    gaps: list[str] = []
    if profile.full_name is None:
        gaps.append("candidate name not extracted.")
    for label, items in (("education", profile.education), ("work experience", profile.work_experience), ("skills", profile.skills)):
        if not items:
            gaps.append(f"no {label} extracted.")
    for i, w in enumerate(profile.work_experience):
        if w.start_date is None or (w.end_date is None and not w.is_current):
            gaps.append(f"work_experience[{i}] ('{w.title or 'untitled'}') has a missing start or end date.")
        elif any(len(d) == 4 for d in (w.start_date, w.end_date) if d):
            gaps.append(f"work_experience[{i}] ('{w.title or 'untitled'}') has year-only dates; durations are approximate.")
    return gaps


def build_recommendation(
    profile: CandidateProfile,
    job: JobDescription,
    *,
    today: date | None = None,
    extra_caveats: list[Caveat] | None = None,
) -> Recommendation:
    """compare a grounded profile with job requirements; always returns decision support only."""
    today = today or date.today()
    assessments = [_assess(r, profile, today) for r in job.requirements]
    mandatory = [a for a in assessments if a.mandatory]
    optional = [a for a in assessments if not a.mandatory]

    def count(items: list[RequirementAssessment]) -> RequirementCounts:
        return RequirementCounts(
            met=sum(a.status == "met" for a in items),
            not_met=sum(a.status == "not_met" for a in items),
            unknown=sum(a.status == "unknown" for a in items),
        )

    m, o = count(mandatory), count(optional)
    known = m.met + m.not_met
    if m.not_met > 0:
        signal = "mandatory_gap"
    elif mandatory and known / len(mandatory) < 0.5:
        signal = "insufficient_information"
    elif m.unknown > 0:
        signal = "partial_support"
    elif mandatory:
        signal = "strong_support"
    else:
        signal = "insufficient_information"

    summary = (
        f"{m.met} of {len(mandatory)} mandatory requirements have supporting evidence in the cv; "
        f"{m.not_met} not met; {m.unknown} unknown (not enough information). "
        f"optional: {o.met} met, {o.not_met} not met, {o.unknown} unknown. "
        "this is an evidence summary for a human reviewer, not a hiring decision."
    )

    gaps = find_data_gaps(profile)
    caveats: list[Caveat] = [
        Caveat(code="llm_extraction_unverified", severity="warning",
               message="the profile was produced by an automated extractor and has not been verified by a human; compare it with the original cv."),
        Caveat(code="keyword_matching", severity="info",
               message="requirement matching is keyword-based; synonyms and related skills are not recognised, so 'unknown' may hide a real match."),
        Caveat(code="protected_attributes_excluded", severity="info",
               message="personal attributes (name, age, gender, nationality, photo, etc.) were not used in this comparison, and the extraction schema has no fields for protected attributes."),
    ]
    unknown_mandatory = [a.requirement_id for a in mandatory if a.status == "unknown"]
    if unknown_mandatory:
        caveats.append(Caveat(code="unassessed_requirements", severity="warning",
                              message=f"mandatory requirements with insufficient information: {', '.join(unknown_mandatory)}. treat them as open questions, not negatives."))
    if m.not_met:
        caveats.append(Caveat(code="verify_not_met", severity="warning",
                              message="'not met' relies on the extracted data being complete; confirm against the original cv before acting."))
    if job.warnings:
        caveats.append(Caveat(code="job_description_parsing", severity="warning", message=" ".join(job.warnings)))
    for gap in gaps:
        caveats.append(Caveat(code="data_gap", severity="info", message=gap))
    caveats.extend(extra_caveats or [])

    return Recommendation(
        signal=signal, summary=summary, mandatory_counts=m, optional_counts=o,  # type: ignore[arg-type]
        assessments=assessments, data_gaps=gaps, caveats=caveats,
    )
