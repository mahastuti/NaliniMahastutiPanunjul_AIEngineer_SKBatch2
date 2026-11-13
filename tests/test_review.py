"""reviewer corrections: validated, recorded beside the result, never written into it."""

import pytest

from src.pipeline import run_screening
from src.review import (
    ReviewError,
    load_review_record,
    new_review_record,
    record_correction,
    resolve_target,
    review_view,
    save_review_record,
)


@pytest.fixture
def result(sample_pdf, sample_jd):
    return run_screening(
        pdf_bytes=sample_pdf,
        job_description=sample_jd,
        mode="mock",
    )


@pytest.fixture
def record(result):
    return new_review_record(result.trace.run_id)


def _correct(record, result, **kwargs):
    defaults = {
        "target": "profile.skills[0]",
        "issue": "incorrect_extraction",
        "reviewer_note": "only coursework level",
    }
    return record_correction(record, result, **{**defaults, **kwargs})


@pytest.mark.parametrize(
    "target",
    ["profile.full_name", "profile.skills[0]", "recommendation.assessments[R5]",
     "recommendation.signal"],
)
def test_valid_targets_resolve(result, target):
    resolve_target(result, target)  # must not raise


@pytest.mark.parametrize(
    "target,code",
    [
        ("profile.nope", "unknown_target"),
        ("profile.skills[99]", "unknown_target"),
        ("recommendation.assessments[R99]", "unknown_target"),
        ("profile.full_name[0]", "unknown_target"),
        ("profile..skills", "unknown_target"),
    ],
)
def test_invalid_targets_are_rejected(result, target, code):
    with pytest.raises(ReviewError) as exc:
        resolve_target(result, target)
    assert exc.value.code == code


def test_recording_a_correction_does_not_change_the_result(result, record):
    before = result.model_dump_json()
    _correct(record, result, corrected_value="Python (basic)", reviewer_id="rev-01")
    assert result.model_dump_json() == before
    assert record.ai_output_modified is False


def test_correction_against_another_run_is_rejected(result):
    with pytest.raises(ReviewError) as exc:
        _correct(new_review_record("run_somethingelse"), result)
    assert exc.value.code == "run_id_mismatch"


def test_issue_type_must_match_the_section_it_points_at(result, record):
    with pytest.raises(ReviewError) as exc:
        _correct(record, result, target="profile.skills[1]", issue="incorrect_matching")
    assert exc.value.code == "scope_mismatch"


def test_ambiguous_evidence_may_point_at_either_section(result, record):
    _correct(record, result, target="profile.skills[1]", issue="ambiguous_evidence",
             reviewer_note="quote is a list, not proof of use")
    _correct(record, result, target="recommendation.assessments[R1]", issue="ambiguous_evidence",
             reviewer_note="dates overlap, unclear")
    assert len(record.corrections) == 2


def test_review_view_shows_ai_output_next_to_the_objection(result, record):
    _correct(record, result, target="recommendation.assessments[R5]",
             issue="unverified_requirement", reviewer_note="candidate confirmed docker by phone")
    view = review_view(result, record)
    entry = view["corrections"][0]
    assert entry["ai_output"]["status"] == "unknown"
    assert entry["reviewer_note"] == "candidate confirmed docker by phone"
    assert view["issue_counts"] == {"unverified_requirement": 1}
    assert view["ai_output_modified"] is False


def test_record_survives_a_save_load_roundtrip(result, record, tmp_path):
    _correct(record, result, reviewer_id="rev-02")
    record.reviewer_decision = "reviewed"
    save_review_record(record, tmp_path)
    loaded = load_review_record(tmp_path)
    assert loaded.run_id == record.run_id
    assert loaded.reviewer_decision == "reviewed"
    assert loaded.corrections[0].reviewer_id == "rev-02"
    assert loaded.ai_output_modified is False
