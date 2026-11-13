"""shared gemini call: structured output, retry, error classification.

both prompts (extraction and recommendation) use this, so the retry and error-handling
policy exists once. prompt text lives in src/prompts.py; this module only transports it.
"""

import logging
import os
import random
import time
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.schemas import CallStats, ScreeningError, TokenUsage

logger = logging.getLogger(__name__)

# model names get retired; override with GEMINI_MODEL in .env when the api says "no longer available".
# history: gemini-2.5-flash returned 404 for a new account. gemini-3.5-flash is the name that
# actually completed a live run from this repo (recorded in the result's provenance), so it is
# the default. a model appearing in client.models.list() does not mean it is callable.
DEFAULT_MODEL = "gemini-3.5-flash"
# two separate budgets, because the two failure kinds need different treatment.
# a schema-invalid answer is a content problem: feeding the same validation feedback back a
# third time rarely helps, so one retry and then fail loudly.
MAX_ATTEMPTS = 2
# a 429/5xx is purely time-based ("this model is currently experiencing high demand"). one
# retry 1.5 s later is far too impatient: a demand spike outlives it, and a failure here
# throws away every api call the run already paid for. so: more attempts, exponential
# backoff with jitter (~1.5s, 3s, 6s), and these do NOT consume a content attempt.
MAX_TRANSIENT_RETRIES = 3
RETRY_DELAY_S = 1.5
# a 429 is not a 503 wearing a different number. "high demand" clears in seconds, but
# "you exceeded your current quota" is a budget that refills on the provider's clock, and
# hammering it makes the refill later rather than sooner. so rate limiting waits far longer,
# and when the service states its own retry delay we use that instead of guessing.
RATE_LIMIT_DELAY_S = 20.0
MAX_RATE_LIMIT_DELAY_S = 90.0

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


def read_token_usage(response: Any) -> TokenUsage:
    """token counts as reported by the api. absent on some responses and in fakes, so the
    result says `unavailable` rather than guessing a number."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return TokenUsage()
    prompt = getattr(meta, "prompt_token_count", None)
    response_tokens = getattr(meta, "candidates_token_count", None)
    total = getattr(meta, "total_token_count", None)
    if prompt is None and response_tokens is None and total is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=prompt,
        response_tokens=response_tokens,
        total_tokens=total,
        source="api_reported",
    )


def summarize_validation_error(exc: ValidationError) -> str:
    """locations and error types only: error inputs may echo cv content."""
    parts = [f"{'.'.join(str(p) for p in e['loc'])} ({e['type']})" for e in exc.errors()[:5]]
    return "; ".join(parts)


def server_requested_delay(exc: Exception) -> float | None:
    """the retry delay the service asked for, if it sent one.

    google's quota errors carry a RetryInfo entry such as {"retryDelay": "27s"}. honouring it
    beats any constant we could pick, because only the service knows when the budget refills.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    entries = details.get("error", {}).get("details") if "error" in details else details.get("details")
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("retryDelay") or entry.get("retry_delay")
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return min(float(raw[:-1]), MAX_RATE_LIMIT_DELAY_S)
            except ValueError:
                return None
    return None


def transient_backoff_seconds(retry_number: int, *, rate_limited: bool = False) -> float:
    """exponential backoff with jitter. jitter matters when several screenings retry at
    once: without it they all come back to a struggling endpoint at the same moment.

    `rate_limited` selects the slower schedule used for 429 responses.
    """
    base_delay = RATE_LIMIT_DELAY_S if rate_limited else RETRY_DELAY_S
    base = min(base_delay * (2 ** (retry_number - 1)), MAX_RATE_LIMIT_DELAY_S)
    return base + random.uniform(0, base * 0.25)


def generate_structured(
    *,
    client: Any,
    model: str,
    render: Renderer,
    schema: type[T],
    temperature: float,
    error_cls: type[ScreeningError],
    stats: CallStats | None = None,
) -> T:
    """call gemini and return a validated `schema` instance, or raise `error_cls`.

    error codes: api_error, empty_response, invalid_output.
    `stats`, when given, is filled in with attempts, retries and reported token usage so
    the caller can record them in an execution trace.
    """
    feedback: str | None = None
    last_error: ScreeningError | None = None
    api_schema = response_schema_for(schema)
    attempt = 0          # content attempts (schema-invalid or empty answers)
    transient = 0        # 429/5xx retries, which do not count as content attempts
    calls = 0

    while True:
        attempt += 1
        if attempt > MAX_ATTEMPTS:
            break
        if calls:
            time.sleep(RETRY_DELAY_S)
        system, user = render(feedback)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=api_schema,
        )

        while True:  # transient-failure loop: same request, later
            calls += 1
            if stats is not None:
                stats.attempts = calls
                stats.retries = calls - 1
            try:
                response = client.models.generate_content(
                    model=model, contents=user, config=config
                )
            except genai_errors.APIError as exc:
                rate_limited = exc.code == 429
                retryable = rate_limited or (isinstance(exc.code, int) and exc.code >= 500)
                last_error = error_cls("api_error", _describe_api_error(exc))
                last_error.__cause__ = exc
                if retryable and transient < MAX_TRANSIENT_RETRIES:
                    transient += 1
                    delay = server_requested_delay(exc) or transient_backoff_seconds(
                        transient, rate_limited=rate_limited
                    )
                    if stats is not None:
                        stats.retry_reasons.append(f"api_error_http_{exc.code}")
                    logger.warning(
                        "gemini http %s; retry %d/%d in %.1fs",
                        exc.code, transient, MAX_TRANSIENT_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                raise last_error from exc
            except Exception as exc:  # network and sdk failures: surface, never swallow
                raise error_cls("api_error", f"gemini call failed: {type(exc).__name__}.") from exc
            break

        if stats is not None:
            stats.tokens = read_token_usage(response)

        text = getattr(response, "text", None)
        if not text:
            last_error = error_cls("empty_response", "the model returned no text (possibly blocked).")
            feedback = None
            if stats is not None:
                stats.retry_reasons.append("empty_response")
            continue
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            feedback = summarize_validation_error(exc)
            last_error = error_cls("invalid_output", f"model output failed schema validation: {feedback}")
            last_error.__cause__ = exc
            if stats is not None:
                stats.retry_reasons.append("schema_validation_failed")

    assert last_error is not None
    raise last_error
