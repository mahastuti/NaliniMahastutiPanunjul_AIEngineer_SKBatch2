"""the execution trace must be accurate, complete on failure, and free of sensitive data."""

import pytest

from src.schemas import CallStats, ScreeningError, TokenUsage
from src.tracing import TraceBuilder, categorize, new_run_id


def test_run_ids_are_unique_and_opaque():
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("run_") and len(i) == 16 for i in ids)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("corrupt_pdf", "input_document"),
        ("no_requirements", "input_document"),
        ("missing_api_key", "configuration"),
        ("api_error", "api"),
        ("empty_response", "api"),
        ("invalid_output", "model_output"),
        ("something_new", "pipeline"),
    ],
)
def test_error_codes_map_to_categories(code, expected):
    assert categorize(code) == expected


def test_successful_stage_is_recorded_with_detail():
    tracer = TraceBuilder("mock")
    with tracer.stage("matching") as t:
        t["detail"] = "signal=partial_support"
    trace = tracer.finish()
    assert len(trace.stages) == 1
    stage = trace.stages[0]
    assert (stage.stage, stage.status, stage.detail) == ("matching", "ok", "signal=partial_support")
    assert stage.duration_ms >= 0 and stage.error_code is None


def test_failing_stage_is_recorded_then_the_error_propagates():
    tracer = TraceBuilder("live")
    with pytest.raises(ScreeningError):
        with tracer.stage("llm_extraction"):
            raise ScreeningError("api_error", "boom")
    trace = tracer.finish("failed")
    assert trace.outcome == "failed"
    assert trace.stages[0].error_code == "api_error"
    assert trace.stages[0].error_category == "api"
    assert trace.error_categories == ["api"]


def test_unexpected_exception_is_still_traced():
    tracer = TraceBuilder("mock")
    with pytest.raises(ZeroDivisionError):
        with tracer.stage("matching"):
            1 / 0
    trace = tracer.finish("failed")
    assert trace.stages[0].error_code == "ZeroDivisionError"
    assert trace.stages[0].error_category == "pipeline"


def test_call_stats_are_attached_to_their_stage_and_summed():
    tracer = TraceBuilder("live")
    with tracer.stage("llm_extraction"):
        pass
    with tracer.stage("recommendation"):
        pass
    tracer.add_call("llm_extraction", CallStats(
        attempts=2, retries=1, retry_reasons=["api_error_http_503"],
        tokens=TokenUsage(prompt_tokens=100, response_tokens=50, total_tokens=150,
                          source="api_reported"),
    ))
    tracer.add_call("recommendation", CallStats(
        attempts=1, retries=0,
        tokens=TokenUsage(prompt_tokens=20, response_tokens=10, total_tokens=30,
                          source="api_reported"),
    ))
    trace = tracer.finish()
    assert trace.retry_total == 1
    assert trace.token_usage.total_tokens == 180
    assert trace.token_usage.source == "api_reported"
    extraction = next(s for s in trace.stages if s.stage == "llm_extraction")
    assert extraction.retries == 1
    assert "api_error_http_503" in extraction.detail


def test_token_usage_is_unavailable_rather_than_zero_when_the_api_reports_nothing():
    tracer = TraceBuilder("live")
    with tracer.stage("llm_extraction"):
        pass
    tracer.add_call("llm_extraction", CallStats(attempts=1))
    trace = tracer.finish()
    assert trace.token_usage.source == "unavailable"
    assert trace.token_usage.total_tokens is None


def test_trace_states_that_it_holds_no_sensitive_data():
    trace = TraceBuilder("mock").finish()
    assert any("no cv text" in note for note in trace.notes)
    assert trace.pipeline_version


# ---------------------------------------------------------------- end-to-end trace


def test_pipeline_records_every_stage_in_order(sample_pdf, sample_jd):
    from src.pipeline import run_screening

    result = run_screening(pdf_bytes=sample_pdf, job_description=sample_jd,
                           mode="mock", include_narrative=True)
    stages = [s.stage for s in result.trace.stages]
    assert stages == [
        "job_parsing", "document_processing", "text_extraction", "llm_extraction",
        "schema_validation", "grounding", "matching", "recommendation",
    ]
    assert all(s.status == "ok" for s in result.trace.stages)
    assert result.trace.outcome == "success"
    assert result.trace.mode == "mock"
    assert result.trace.total_duration_ms > 0


def test_narrative_stage_is_marked_skipped_when_not_requested(sample_pdf, sample_jd):
    from src.pipeline import run_screening

    result = run_screening(pdf_bytes=sample_pdf, job_description=sample_jd, mode="mock")
    recommendation = next(s for s in result.trace.stages if s.stage == "recommendation")
    assert recommendation.status == "skipped"


def test_a_failing_run_still_yields_the_stages_it_completed(sample_jd):
    from src.pipeline import run_screening
    from src.schemas import ScreeningError

    tracer = TraceBuilder("mock")
    with pytest.raises(ScreeningError):
        run_screening(pdf_bytes=b"not a pdf", job_description=sample_jd,
                      mode="mock", tracer=tracer)
    trace = tracer.finish("failed")
    stages = {s.stage: s for s in trace.stages}
    assert stages["job_parsing"].status == "ok"
    assert stages["text_extraction"].status == "error"
    assert stages["text_extraction"].error_code == "unsupported_format"
    assert trace.error_categories == ["input_document"]
