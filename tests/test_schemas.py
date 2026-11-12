import pytest
from pydantic import ValidationError

from src.schemas import CandidateProfile, Recommendation, RequirementCounts, Skill, WorkExperience


def _work(**over):
    base = dict(
        company="X", title="Y", start_date="2020-01", end_date="2021-01", is_current=False,
        responsibilities=[], technologies=[], evidence=["q"],
    )
    base.update(over)
    return WorkExperience(**base)


@pytest.mark.parametrize("bad", ["2020-13", "Jan 2020", "20-01", "2020-1", "present"])
def test_invalid_dates_rejected(bad):
    with pytest.raises(ValidationError):
        _work(start_date=bad)


@pytest.mark.parametrize("good", ["2020", "2020-01", None])
def test_valid_dates_accepted(good):
    assert _work(start_date=good).start_date == good


def test_extra_fields_are_forbidden():
    # e.g. a model trying to smuggle in a protected attribute
    with pytest.raises(ValidationError):
        Skill(name="Python", evidence=[], gender="x")
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(
            {"full_name": None, "education": [], "work_experience": [], "skills": [],
             "projects": [], "certifications": [], "age": 30}
        )


def test_recommendation_cannot_represent_a_final_decision():
    counts = RequirementCounts(met=0, not_met=0, unknown=0)
    base = dict(signal="partial_support", summary="s", mandatory_counts=counts,
                optional_counts=counts, assessments=[], data_gaps=[], caveats=[])
    assert Recommendation(**base).requires_human_review is True
    with pytest.raises(ValidationError):
        Recommendation(**base, requires_human_review=False)
    with pytest.raises(ValidationError):
        Recommendation(**base, is_final_decision=True)
