from datetime import date

from src.recommendation import build_recommendation, experience_months, parse_job_description
from src.schemas import CandidateProfile, Certification, Education, Skill, WorkExperience

TODAY = date(2026, 11, 11)


def _profile(**over) -> CandidateProfile:
    base = dict(full_name=None, education=[], work_experience=[], skills=[], projects=[], certifications=[])
    base.update(over)
    return CandidateProfile(**base)


def _work(start, end, current=False, title="Analyst", techs=()):
    return WorkExperience(company="Acme", title=title, start_date=start, end_date=end, is_current=current,
                          responsibilities=[], technologies=list(techs), evidence=[f"{title} at Acme"])


def _edu(level, in_progress=False, end="2019"):
    return Education(institution="U", degree=f"{level} degree", degree_level=level, field_of_study=None,
                     start_date=None, end_date=end, in_progress=in_progress, evidence=["degree line"])


def _job(text):
    return parse_job_description("Requirements:\n" + text)


def _status(profile, bullet):
    rec = build_recommendation(profile, _job(f"- {bullet}"), today=TODAY)
    return rec.assessments[0].status


# ------------------------------------------------------------------ parsing


def test_parses_sample_job_description(sample_jd):
    job = parse_job_description(sample_jd)
    assert job.title == "Data Analyst"
    assert [(r.kind, r.mandatory) for r in job.requirements] == [
        ("min_years", True), ("education_level", True), ("skills", True), ("skills", True),
        ("skills", True), ("skills", False), ("skills", False),
    ]
    by_id = {r.id: r for r in job.requirements}
    assert by_id["R1"].min_years == 3
    assert by_id["R3"].terms == ["Python", "SQL"] and by_id["R3"].term_mode == "all"
    assert by_id["R4"].terms == ["Tableau", "Power BI"] and by_id["R4"].term_mode == "any"
    assert by_id["R7"].certification_only is True


def test_no_requirements_section_warns_and_treats_as_mandatory():
    job = parse_job_description("- Python\n- SQL")
    assert all(r.mandatory for r in job.requirements) and job.warnings


def test_empty_job_description_yields_no_requirements():
    assert parse_job_description("We love data.").requirements == []


# ------------------------------------------------------------------ the core rule


def test_missing_skill_is_unknown_never_not_met():
    assert _status(_profile(skills=[Skill(name="Python", evidence=["q"])]), "Docker") == "unknown"
    assert _status(_profile(), "Python") == "unknown"


def test_empty_profile_never_produces_not_met():
    rec = build_recommendation(_profile(), parse_job_description(
        "Requirements:\n- 5 years experience\n- Master's degree\n- Python"), today=TODAY)
    assert all(a.status == "unknown" for a in rec.assessments)
    assert rec.signal == "insufficient_information"


def test_skill_found_in_work_technologies_is_met_with_evidence():
    p = _profile(work_experience=[_work("2020-01", "2021-01", techs=["Docker"])])
    rec = build_recommendation(p, _job("- Docker"), today=TODAY)
    a = rec.assessments[0]
    assert a.status == "met" and a.evidence and a.facts


def test_certification_requirement_ignores_plain_skills():
    only_skill = _profile(skills=[Skill(name="Google Cloud", evidence=["q"])])
    assert _status(only_skill, "Google Cloud certification") == "unknown"
    with_cert = _profile(certifications=[Certification(name="Google Cloud Professional Engineer",
                                                       issuer=None, date=None, evidence=["cert line"])])
    assert _status(with_cert, "Google Cloud certification") == "met"


def test_any_vs_all_mode():
    p = _profile(skills=[Skill(name="Python", evidence=["q"])])
    assert _status(p, "Python or R") == "met"
    assert _status(p, "Python and R") == "unknown"


# ------------------------------------------------------------------ years


def test_years_met_not_met_and_unknown():
    assert _status(_profile(work_experience=[_work("2018-01", "2022-01")]), "3+ years of experience") == "met"
    assert _status(_profile(work_experience=[_work("2021-01", "2022-01")]), "3+ years of experience") == "not_met"
    # an undated role means the shortfall cannot be proven
    p = _profile(work_experience=[_work("2021-01", "2022-01"), _work(None, None)])
    assert _status(p, "3+ years of experience") == "unknown"


def test_overlapping_roles_are_not_double_counted():
    p = _profile(work_experience=[_work("2020-01", "2022-01"), _work("2021-01", "2022-01")])
    lower, upper, undated = experience_months(p, TODAY)
    assert (lower, upper, undated) == (24, 24, 0)


def test_year_only_dates_use_conservative_and_generous_bounds():
    p = _profile(work_experience=[_work("2019", "2020")])
    lower, upper, _ = experience_months(p, TODAY)
    assert lower == 1 and upper == 23
    # 1 year lies between the bounds (1..23 months) -> cannot decide
    assert _status(p, "1+ years of experience") == "unknown"
    # 2 years (24 months) exceeds even the generous bound -> provably not met
    assert _status(p, "2+ years of experience") == "not_met"


def test_current_role_counts_until_today():
    p = _profile(work_experience=[_work("2024-11", None, current=True)])
    lower, _, _ = experience_months(p, TODAY)
    assert lower == 24


# ------------------------------------------------------------------ education


def test_education_levels():
    assert _status(_profile(education=[_edu("master")]), "Bachelor's degree") == "met"
    assert _status(_profile(education=[_edu("diploma")]), "Bachelor's degree") == "not_met"
    assert _status(_profile(education=[]), "Bachelor's degree") == "unknown"
    assert _status(_profile(education=[_edu("bachelor", in_progress=True, end=None)]), "Bachelor's degree") == "unknown"
    assert _status(_profile(education=[_edu("other")]), "Bachelor's degree") == "unknown"


# ------------------------------------------------------------------ output contract


def test_recommendation_is_never_final_and_has_caveats():
    rec = build_recommendation(_profile(), _job("- Python"), today=TODAY)
    assert rec.requires_human_review is True and rec.is_final_decision is False
    codes = {c.code for c in rec.caveats}
    assert {"llm_extraction_unverified", "keyword_matching", "unassessed_requirements"} <= codes


def test_signal_values():
    full = _profile(skills=[Skill(name="Python", evidence=["q"])])
    assert build_recommendation(full, _job("- Python"), today=TODAY).signal == "strong_support"
    gap = _profile(education=[_edu("diploma")], skills=[Skill(name="Python", evidence=["q"])])
    job = parse_job_description("Requirements:\n- Python\n- Bachelor's degree")
    assert build_recommendation(gap, job, today=TODAY).signal == "mandatory_gap"
    partial = _profile(skills=[Skill(name="Python", evidence=["q"])])
    job = parse_job_description("Requirements:\n- Python\n- Docker")
    assert build_recommendation(partial, job, today=TODAY).signal == "partial_support"
