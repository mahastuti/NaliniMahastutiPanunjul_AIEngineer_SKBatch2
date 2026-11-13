"""execution trace: what the pipeline observably did, per run.

the trace answers operational questions — which stage failed, how long each took, how
many retries the api needed, how many tokens a run cost — without recording anything
sensitive. three rules are enforced here rather than left to callers:

1. no cv text, no candidate data, no api key is ever written into a trace;
2. the run id is random, so a trace can be shared without identifying a candidate;
3. token counts are only reported when the api reported them; nothing is estimated.

the trace records observable steps, not model reasoning: there is no field for a
"thought process", and none is requested from the model.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from src.schemas import (
    PIPELINE_VERSION,
    CallStats,
    ErrorCategory,
    ExecutionTrace,
    Mode,
    ScreeningError,
    StageName,
    StageTrace,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# machine-readable error code -> coarse bucket, so failures can be counted without
# parsing messages. unknown codes fall back to "pipeline".
ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    # src/pdf_parser.py
    "empty_file": "input_document",
    "too_large": "input_document",
    "unsupported_format": "input_document",
    "corrupt_pdf": "input_document",
    "encrypted_pdf": "input_document",
    "empty_document": "input_document",
    "too_many_pages": "input_document",
    "scanned_pdf": "input_document",
    "no_text": "input_document",
    "unreadable_text": "input_document",
    "too_long": "input_document",
    # src/pipeline.py
    "no_requirements": "input_document",
    # configuration
    "missing_api_key": "configuration",
    "invalid_mode": "configuration",
    # src/gemini_client.py
    "api_error": "api",
    "empty_response": "api",
    "invalid_output": "model_output",
}


def categorize(code: str) -> ErrorCategory:
    return ERROR_CATEGORIES.get(code, "pipeline")


def new_run_id() -> str:
    """random identifier for one run. deliberately not derived from the cv or the
    candidate: two screenings of the same person must not share an id."""
    return f"run_{uuid.uuid4().hex[:12]}"


def _sum_tokens(parts: list[TokenUsage]) -> TokenUsage:
    reported = [t for t in parts if t.source == "api_reported"]
    if not reported:
        return TokenUsage()

    def add(attr: str) -> int | None:
        values = [getattr(t, attr) for t in reported if getattr(t, attr) is not None]
        return sum(values) if values else None

    return TokenUsage(
        prompt_tokens=add("prompt_tokens"),
        response_tokens=add("response_tokens"),
        total_tokens=add("total_tokens"),
        source="api_reported",
    )


class TraceBuilder:
    """collects stage records during one pipeline run, then freezes them into an
    `ExecutionTrace`. a stage that raises is recorded as an error before the exception
    continues upward, so a failed run still produces a usable trace."""

    def __init__(self, mode: Mode, run_id: str | None = None) -> None:
        self.run_id = run_id or new_run_id()
        self.mode: Mode = mode
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self.stages: list[StageTrace] = []
        self._token_parts: list[TokenUsage] = []
        self._retries = 0
        self.model: str | None = None
        self.extraction_prompt_version: str | None = None
        self.recommendation_prompt_version: str | None = None
        self.notes: list[str] = []

    # ------------------------------------------------------------------ recording

    @contextmanager
    def stage(self, name: StageName, detail: str | None = None) -> Iterator[dict]:
        """time one stage. the yielded dict accepts a `detail` key the body can set."""
        box: dict = {"detail": detail}
        start = time.perf_counter()
        try:
            yield box
        except ScreeningError as exc:
            self._record(name, "error", start, box.get("detail"), exc.code)
            raise
        except Exception as exc:  # unexpected failures are still traced, then re-raised
            self._record(name, "error", start, box.get("detail"), type(exc).__name__)
            raise
        else:
            self._record(name, "ok", start, box.get("detail"))

    def _record(
        self,
        name: StageName,
        status: str,
        start: float,
        detail: str | None,
        code: str | None = None,
    ) -> None:
        self.stages.append(
            StageTrace(
                stage=name,
                status=status,  # type: ignore[arg-type]
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
                detail=detail,
                error_code=code,
                error_category=categorize(code) if code else None,
            )
        )

    def mark(self, name: StageName, status: str, detail: str | None = None) -> None:
        """record a stage with no measurable duration of its own (e.g. a step that
        happened inside another call, or one that was skipped)."""
        self.stages.append(
            StageTrace(stage=name, status=status, duration_ms=0.0, detail=detail)  # type: ignore[arg-type]
        )

    def add_call(self, stage: StageName, stats: CallStats | None) -> None:
        """attach one llm call's retries and token usage to an already recorded stage."""
        if stats is None:
            return
        self._retries += stats.retries
        self._token_parts.append(stats.tokens)
        for entry in self.stages:
            if entry.stage == stage:
                entry.retries = stats.retries
                if stats.retry_reasons:
                    reasons = ", ".join(stats.retry_reasons)
                    entry.detail = f"{entry.detail + '; ' if entry.detail else ''}retried: {reasons}"

    # ------------------------------------------------------------------ output

    def finish(self, outcome: str = "success") -> ExecutionTrace:
        categories: list[ErrorCategory] = []
        for entry in self.stages:
            if entry.error_category and entry.error_category not in categories:
                categories.append(entry.error_category)
        trace = ExecutionTrace(
            run_id=self.run_id,
            started_at=self.started_at,
            pipeline_version=PIPELINE_VERSION,
            mode=self.mode,
            model=self.model,
            extraction_prompt_version=self.extraction_prompt_version,
            recommendation_prompt_version=self.recommendation_prompt_version,
            stages=self.stages,
            total_duration_ms=round((time.perf_counter() - self._t0) * 1000, 3),
            retry_total=self._retries,
            error_categories=categories,
            token_usage=_sum_tokens(self._token_parts),
            outcome=outcome,  # type: ignore[arg-type]
            notes=[
                "this trace records observable processing steps only: no cv text, no "
                "candidate data, no credentials, and no model reasoning.",
                *self.notes,
            ],
        )
        logger.info(
            "run %s finished: outcome=%s stages=%d retries=%d duration_ms=%.1f",
            trace.run_id, outcome, len(trace.stages), trace.retry_total, trace.total_duration_ms,
        )
        return trace
