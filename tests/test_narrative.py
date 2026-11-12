"""tests for the narrative layer: the llm explains the deterministic result and is
verified programmatically. the central guarantee is that it cannot change a status."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from src.narrative import generate_narrative, verify_narrative
from src.recommendation import build_recommendation, parse_job_description
from src.schemas import (
    CandidateProfile,
    NarrativeReport,
    RequirementExplanation,
    Skill,
)

TODAY = date(2026, 11, 12)
CV = "Dewi Uji\nSKILLS\nPython, SQL\nEXPERIENCE\nAnalyst at PT Uji | 2023 - 2024"
JD = "Requirements:\n- Python and SQL proficiency\n- Master's degree\n- Docker"


@pytest.fixture
def setup():
    profile = CandidateProfile(
        full_name="Dewi Uji",
        education=[], work_experience=[], projects=[], certifications=[],
        skills=[Skill(name="Python", evidence=["Python, SQL"]),
                Skill(name="SQL", evidence=["Python, SQL"])],
    )
    job = parse_job_description(JD)
    rec = build_recommendation(profile, job, today=TODAY)
    return profile, job, rec


def _report(**over) -> NarrativeReport:
    base = dict(
        candidate_summary="The candidate lists Python and SQL.",
        confirmed_strengths=["R1: Python and SQL are listed."],
        requirement_explanations=[],
        potential_gaps=[],
        uncertainties=[],
        reviewer_questions=[],
    )
    base.update(over)
    return NarrativeReport(**base)


def _expl(rid, status, explanation="ok.", quotes=()):
    return RequirementExplanation(requirement_id=rid, status_restated=status,
                                  explanation=explanation, evidence_quotes=list(quotes))


# ------------------------------------------------------- the core guarantee


def test_model_cannot_change_a_status(setup):
    _, _, rec = setup
    authoritative = {a.requirement_id: a.status for a in rec.assessments}
    assert authoritative["R1"] == "met" and authoritative["R3"] == "unknown"

    # the model claims R3 is met and R1 is not_met
    tampered = _report(requirement_explanations=[
        _expl("R1", "not_met"), _expl("R3", "met"),
    ])
    verified, violations = verify_narrative(tampered, rec, CV)
    restated = {e.requirement_id: e.status_restated for e in verified.requirement_explanations}
    assert restated["R1"] == "met" and restated["R3"] == "unknown"
    assert sum("restored" in v for v in violations) == 2


def test_invented_and_duplicate_requirements_are_removed(setup):
    _, _, rec = setup
    report = _report(requirement_explanations=[
        _expl("R1", "met"), _expl("R1", "met"), _expl("R99", "met"),
    ])
    verified, violations = verify_narrative(report, rec, CV)
    assert [e.requirement_id for e in verified.requirement_explanations] == ["R1"]
    assert any("unknown requirement 'R99'" in v for v in violations)
    assert any("duplicate" in v for v in violations)


def test_missing_explanations_are_reported(setup):
    _, _, rec = setup
    verified, violations = verify_narrative(_report(), rec, CV)
    assert any("no explanation produced for" in v for v in violations)


def test_unverifiable_quotes_are_dropped(setup):
    _, _, rec = setup
    report = _report(requirement_explanations=[
        _expl("R1", "met", quotes=["Python, SQL", "Expert in Kubernetes at Google"]),
    ])
    verified, violations = verify_narrative(report, rec, CV)
    assert verified.requirement_explanations[0].evidence_quotes == ["Python, SQL"]
    assert any("unverifiable evidence quote" in v for v in violations)


# ------------------------------------------------------- banned content


@pytest.mark.parametrize("text", [
    "The candidate should be hired immediately.",
    "Should we hire this candidate?",
    "I would not reject them at this stage.",
    "Overall score: 85%.",
    "I rate this candidate 8 out of 10.",
    "This is the best candidate in the pool.",
    "They are a strong fit for the role.",
    "They rank in the top 5 of applicants.",
])
def test_scores_and_decisions_are_stripped(setup, text):
    _, _, rec = setup
    report = _report(candidate_summary=f"The candidate lists Python. {text}")
    verified, violations = verify_narrative(report, rec, CV)
    assert text not in verified.candidate_summary
    assert "The candidate lists Python." in verified.candidate_summary
    assert any("candidate_summary contained" in v for v in violations)


def test_banned_content_stripped_from_explanations_and_lists(setup):
    _, _, rec = setup
    report = _report(
        requirement_explanations=[_expl("R1", "met", "Python is listed. Score: 9/10.")],
        confirmed_strengths=["Python is listed.", "Ranked first overall."],
    )
    verified, violations = verify_narrative(report, rec, CV)
    assert "9/10" not in verified.requirement_explanations[0].explanation
    assert "Ranked first overall." not in verified.confirmed_strengths
    assert len(violations) >= 2


def test_unknown_requirement_cannot_be_presented_as_a_gap(setup):
    _, _, rec = setup
    # R3 is unknown (no docker evidence); calling it a gap treats missing info as a negative
    report = _report(potential_gaps=["R3: the candidate lacks Docker experience."])
    verified, violations = verify_narrative(report, rec, CV)
    assert verified.potential_gaps == []
    assert any("potential_gaps" in v for v in violations)


def test_genuine_not_met_gap_is_kept():
    profile = CandidateProfile(
        full_name=None, education=[], work_experience=[], projects=[], certifications=[],
        skills=[Skill(name="Python", evidence=["Python, SQL"])],
    )
    job = parse_job_description("Requirements:\n- 5 years of experience\n- Python")
    # no roles at all -> years is unknown, so build a case that really is not_met:
    from src.schemas import WorkExperience
    profile = profile.model_copy(update={"work_experience": [
        WorkExperience(company="PT Uji", title="Analyst", start_date="2023-01", end_date="2024-01",
                       is_current=False, responsibilities=[], technologies=[],
                       evidence=["Analyst at PT Uji | 2023 - 2024"])]})
    rec = build_recommendation(profile, job, today=TODAY)
    assert {a.requirement_id: a.status for a in rec.assessments}["R1"] == "not_met"
    report = _report(potential_gaps=["R1: only about one year of dated experience is listed."])
    verified, violations = verify_narrative(report, rec, CV)
    assert verified.potential_gaps  # kept
    assert not any("potential_gaps" in v for v in violations)


# ------------------------------------------------------- modes


def test_mock_narrative_is_labelled_and_consistent(setup):
    profile, job, rec = setup
    result = generate_narrative(profile, job, rec, CV, mode="mock")
    assert result.mode == "mock" and result.model is None
    assert result.prompt_version == "recommendation-v1"
    assert any("MOCK MODE" in v for v in result.violations)
    restated = {e.requirement_id: e.status_restated for e in result.report.requirement_explanations}
    assert restated == {a.requirement_id: a.status for a in rec.assessments}


def test_live_narrative_with_fake_client_is_verified(setup):
    profile, job, rec = setup
    payload = {
        "candidate_summary": "The candidate lists Python and SQL. Overall rating: 9/10.",
        "confirmed_strengths": ["Python and SQL are listed."],
        "requirement_explanations": [
            {"requirement_id": "R1", "status_restated": "met",
             "explanation": "Both skills appear in the cv.", "evidence_quotes": ["Python, SQL"]},
            {"requirement_id": "R3", "status_restated": "met",
             "explanation": "Docker is clearly present.", "evidence_quotes": ["Docker expert"]},
        ],
        "potential_gaps": [],
        "uncertainties": [],
        "reviewer_questions": ["Can you describe your SQL work?"],
    }
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(text=json.dumps(payload))))
    result = generate_narrative(profile, job, rec, CV, mode="live", client=client, model="fake")
    assert result.mode == "live" and result.model == "fake"
    restated = {e.requirement_id: e.status_restated for e in result.report.requirement_explanations}
    assert restated["R3"] == "unknown"              # tampered status restored
    assert "9/10" not in result.report.candidate_summary  # score stripped
    r3 = next(e for e in result.report.requirement_explanations if e.requirement_id == "R3")
    assert r3.evidence_quotes == []                 # fabricated quote dropped
    assert len(result.violations) >= 3


def test_narrative_prompt_receives_matching_results(setup):
    profile, job, rec = setup
    captured = {}

    def fake_generate(**kw):
        captured.update(kw)
        return SimpleNamespace(text=json.dumps({
            "candidate_summary": "x", "confirmed_strengths": [], "requirement_explanations": [],
            "potential_gaps": [], "uncertainties": [], "reviewer_questions": []}))

    client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate))
    generate_narrative(profile, job, rec, CV, mode="live", client=client)
    prompt = captured["contents"]
    assert "<matching_json>" in prompt and '"status": "unknown"' in prompt
    assert captured["config"].temperature == 0.0
