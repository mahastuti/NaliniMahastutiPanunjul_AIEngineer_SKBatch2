"""shared gemini call: structured output, retry, error classification.

both prompts (extraction and recommendation) use this, so the retry and error-handling
policy exists once. prompt text lives in src/prompts.py; this module only transports it.
"""

import logging
import os
import time
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.schemas import ScreeningError

logger = logging.getLogger(__name__)

# model names get retired; override with GEMINI_MODEL in .env when the api says "no longer available".
# gemini-2.5-flash was rejected (404) for a new account; this name comes from google's own error
# message and has not been verified by a successful call from this repo.
DEFAULT_MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 2
RETRY_DELAY_S = 1.5

T = TypeVar("T", bound=BaseModel)
# render(validation_feedback) -> (system_instruction, user_prompt)
Renderer = Callable[[str | None], tuple[str, str]]


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def build_client(api_key: str | None, error_cls: type[ScreeningError]) -> Any:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise error_cls(
            "missing_api_key", "live mode needs GEMINI_API_KEY (see .env.example); none was found."
        )
    return genai.Client(api_key=key)


_DROPPED_KEYWORDS = {"$defs", "additionalProperties", "title", "default"}


def response_schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """json schema of `model_cls` in the subset the gemini api accepts.

    passing the pydantic class straight to the sdk fails: `extra="forbid"` becomes
    `additional_properties`, which the api rejects (http 400 "unknown name"). the strictness
    is kept where it belongs: the response is still validated by `model_cls` afterwards, so
    forbidden fields (age, gender, ...) are rejected on our side. here we only drop schema
    keywords the api does not know, inline `$ref`s, and express `X | None` as nullable.
    """
    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return clean(defs[node["$ref"].split("/")[-1]])
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROPPED_KEYWORDS:
                continue
            if key == "properties":
                # property *names* may collide with keywords (e.g. a field called "title")
                out[key] = {name: clean(sub) for name, sub in value.items()}
            elif key == "anyOf":
                branches = [clean(b) for b in value]
                non_null = [b for b in branches if b.get("type") != "null"]
                if len(non_null) == 1 and len(branches) == 2:
                    out.update(non_null[0])
                    out["nullable"] = True
                else:
                    out[key] = branches
            else:
                out[key] = clean(value)
        return out

    return clean(raw)


def _describe_api_error(exc: genai_errors.APIError) -> str:
    """status + the service's own message, truncated. the message describes the problem
    (bad key, unknown model, rejected schema); it does not contain the api key or the prompt."""
    detail = " ".join(str(getattr(exc, "message", "") or "").split())[:300]
    status = getattr(exc, "status", None)
    head = f"gemini api error (http {exc.code}" + (f", {status})" if status else ")")
    return f"{head}: {detail}" if detail else f"{head}."


def summarize_validation_error(exc: ValidationError) -> str:
    """locations and error types only: error inputs may echo cv content."""
    parts = [f"{'.'.join(str(p) for p in e['loc'])} ({e['type']})" for e in exc.errors()[:5]]
    return "; ".join(parts)


def generate_structured(
    *,
    client: Any,
    model: str,
    render: Renderer,
    schema: type[T],
    temperature: float,
    error_cls: type[ScreeningError],
) -> T:
    """call gemini and return a validated `schema` instance, or raise `error_cls`.

    error codes: api_error, empty_response, invalid_output.
    """
    feedback: str | None = None
    last_error: ScreeningError | None = None
    api_schema = response_schema_for(schema)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(RETRY_DELAY_S)
        system, user = render(feedback)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=api_schema,
        )
        try:
            response = client.models.generate_content(model=model, contents=user, config=config)
        except genai_errors.APIError as exc:
            retryable = exc.code == 429 or (isinstance(exc.code, int) and exc.code >= 500)
            last_error = error_cls("api_error", _describe_api_error(exc))
            last_error.__cause__ = exc
            if retryable:
                continue
            raise last_error from exc
        except Exception as exc:  # network and sdk failures: surface, never swallow
            raise error_cls("api_error", f"gemini call failed: {type(exc).__name__}.") from exc

        text = getattr(response, "text", None)
        if not text:
            last_error = error_cls("empty_response", "the model returned no text (possibly blocked).")
            feedback = None
            continue
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            feedback = summarize_validation_error(exc)
            last_error = error_cls("invalid_output", f"model output failed schema validation: {feedback}")
            last_error.__cause__ = exc

    assert last_error is not None
    raise last_error
