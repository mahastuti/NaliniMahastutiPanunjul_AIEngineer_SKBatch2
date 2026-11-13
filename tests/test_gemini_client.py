"""regression tests for the schema sent to gemini.

bug being guarded: passing the pydantic class directly made the sdk send
`additional_properties` (from extra="forbid"), which the api rejects with
http 400 "unknown name". the request body is captured offline; no network is used.
"""

import json

import pytest
from google import genai
from google.genai import types

from src.gemini_client import response_schema_for
from src.schemas import CandidateProfile, NarrativeReport


class _Captured(Exception):
    pass


def _request_body(response_schema) -> str:
    client = genai.Client(api_key="dummy-offline-key")
    box: dict = {}

    def fake_request(*args, **kwargs):
        box["args"] = args
        raise _Captured

    client._api_client.request = fake_request
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=response_schema
    )
    with pytest.raises(_Captured):
        client.models.generate_content(model="m", contents="x", config=config)
    return json.dumps(box["args"], default=str)


def test_passing_the_pydantic_class_directly_is_what_broke():
    """documents the failure mode, so the fix cannot be silently reverted."""
    body = _request_body(CandidateProfile)
    assert "additional_properties" in body or "additionalProperties" in body


@pytest.mark.parametrize("model_cls", [CandidateProfile, NarrativeReport])
def test_cleaned_schema_never_reaches_the_api_with_additional_properties(model_cls):
    body = _request_body(response_schema_for(model_cls))
    assert "additional_properties" not in body
    assert "additionalProperties" not in body


@pytest.mark.parametrize("model_cls", [CandidateProfile, NarrativeReport])
def test_cleaned_schema_is_self_contained(model_cls):
    text = json.dumps(response_schema_for(model_cls))
    assert "$ref" not in text and "$defs" not in text
    assert '"null"' not in text  # nullable is expressed as a flag, not a null branch


def test_field_named_title_survives_keyword_stripping():
    schema = response_schema_for(CandidateProfile)
    work = schema["properties"]["work_experience"]["items"]["properties"]
    assert work["title"] == {"type": "string", "nullable": True}


def test_optional_fields_become_nullable():
    schema = response_schema_for(CandidateProfile)
    work = schema["properties"]["work_experience"]["items"]["properties"]
    assert work["is_current"] == {"type": "boolean", "nullable": True}
    assert work["responsibilities"]["type"] == "array"


def test_enum_fields_keep_their_allowed_values():
    schema = response_schema_for(CandidateProfile)
    level = schema["properties"]["education"]["items"]["properties"]["degree_level"]
    assert "bachelor" in level["enum"] and level["nullable"] is True


def test_strictness_still_enforced_on_our_side():
    """dropping additionalProperties from the api schema must not weaken validation."""
    from pydantic import ValidationError

    payload = {"full_name": None, "age": 30, "education": [], "work_experience": [],
               "skills": [], "projects": [], "certifications": []}
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(payload)


# ---------------------------------------------------------------- call accounting


class _Usage:
    prompt_token_count = 1200
    candidates_token_count = 340
    total_token_count = 1540


class _Response:
    def __init__(self, text: str, usage=None):
        self.text = text
        self.usage_metadata = usage


class _Models:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.models = _Models(responses)


_VALID = NarrativeReport(
    candidate_summary="s", confirmed_strengths=[], requirement_explanations=[],
    potential_gaps=[], uncertainties=[], reviewer_questions=[],
).model_dump_json()


def _generate(responses, stats):
    from src.gemini_client import generate_structured
    from src.schemas import ScreeningError

    return generate_structured(
        client=_FakeClient(responses), model="m",
        render=lambda feedback: ("sys", "user"), schema=NarrativeReport,
        temperature=0.0, error_cls=ScreeningError, stats=stats,
    )


def test_reported_token_usage_is_recorded(monkeypatch):
    from src.schemas import CallStats

    stats = CallStats()
    _generate([_Response(_VALID, _Usage())], stats)
    assert stats.attempts == 1 and stats.retries == 0
    assert stats.tokens.source == "api_reported"
    assert stats.tokens.total_tokens == 1540


def test_missing_token_usage_is_reported_as_unavailable_not_zero():
    from src.schemas import CallStats

    stats = CallStats()
    _generate([_Response(_VALID)], stats)
    assert stats.tokens.source == "unavailable"
    assert stats.tokens.total_tokens is None


def test_a_schema_retry_is_counted_with_its_reason(monkeypatch):
    from src import gemini_client
    from src.schemas import CallStats

    monkeypatch.setattr(gemini_client, "RETRY_DELAY_S", 0)
    stats = CallStats()
    _generate([_Response("{not json"), _Response(_VALID, _Usage())], stats)
    assert stats.attempts == 2 and stats.retries == 1
    assert stats.retry_reasons == ["schema_validation_failed"]
