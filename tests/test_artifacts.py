"""run artifacts must come from a real run, stay separated per run, and leak nothing."""

import json

import pytest

from src.artifacts import (
    CANDIDATE_FILE,
    MATCHING_FILE,
    REPORT_FILE,
    TRACE_FILE,
    build_requirement_matching,
    write_failed_run,
    write_run_artifacts,
)
from src.pipeline import run_screening
from src.tracing import TraceBuilder


@pytest.fixture
def result(sample_pdf, sample_jd):
    return run_screening(
        pdf_bytes=sample_pdf,
        job_description=sample_jd,
        mode="mock",
        include_narrative=True,
    )


def test_all_four_artifacts_are_written(result, tmp_path):
    directory = write_run_artifacts(result, tmp_path)
    written = {p.name for p in directory.iterdir()}
    assert written == {CANDIDATE_FILE, MATCHING_FILE, REPORT_FILE, TRACE_FILE}
    for path in directory.iterdir():
        json.loads(path.read_text(encoding="utf-8"))  # all valid json


def test_each_run_gets_its_own_directory(result, sample_pdf, sample_jd, tmp_path):
    second = run_screening(
        pdf_bytes=sample_pdf,
        job_description=sample_jd,
        mode="mock",
    )
    first_dir = write_run_artifacts(result, tmp_path)
    second_dir = write_run_artifacts(second, tmp_path)
    assert first_dir != second_dir
    assert first_dir.exists() and second_dir.exists()


def test_artifacts_carry_the_run_id_of_their_trace(result, tmp_path):
    directory = write_run_artifacts(result, tmp_path)
    for name in (CANDIDATE_FILE, MATCHING_FILE, REPORT_FILE):
        payload = json.loads((directory / name).read_text(encoding="utf-8"))
        assert payload["run_id"] == result.trace.run_id
    assert directory.name == result.trace.run_id


def test_matching_artifact_reports_category_and_marks_unknown_as_missing_info(result):
    payload = build_requirement_matching(result)
    rows = {row["requirement_id"]: row for row in payload["requirements"]}
    assert rows["R1"]["category"] == "min_years"
    assert rows["R2"]["category"] == "education_level"
    unknown = [row for row in payload["requirements"] if row["status"] == "unknown"]
    assert unknown, "sample case should contain at least one unknown"
    for row in unknown:
        assert "not evidence that the candidate lacks" in row["caveat"]
    for row in payload["requirements"]:
        if row["status"] != "unknown":
            assert row["caveat"] is None


def test_report_artifact_keeps_the_human_review_flags(result, tmp_path):
    directory = write_run_artifacts(result, tmp_path)
    payload = json.loads((directory / REPORT_FILE).read_text(encoding="utf-8"))
    assert payload["human_review"]["requires_human_review"] is True
    assert payload["human_review"]["is_final_decision"] is False


def test_trace_artifact_contains_no_cv_text(result, tmp_path, sample_pdf):
    directory = write_run_artifacts(result, tmp_path)
    trace_text = (directory / TRACE_FILE).read_text(encoding="utf-8")
    for secret in ("Dewi Contoh", "PT Contoh Analitika", "Tableau"):
        assert secret not in trace_text


def test_failed_run_writes_only_a_trace_with_its_error(tmp_path):
    tracer = TraceBuilder("mock")
    with tracer.stage("document_processing"):
        pass
    directory = write_failed_run(tracer.finish("failed"), "corrupt_pdf", "bad file", tmp_path)
    assert {p.name for p in directory.iterdir()} == {TRACE_FILE}
    payload = json.loads((directory / TRACE_FILE).read_text(encoding="utf-8"))
    assert payload["outcome"] == "failed"
    assert payload["error"] == {"code": "corrupt_pdf", "message": "bad file"}
