"""human-in-the-loop: record what a reviewer disagrees with, without touching the output.

the design rule is that the ai output is immutable evidence. if a reviewer's correction
were written into the profile, nobody could later tell what the system actually produced,
and the evaluation numbers would quietly improve every time a human fixed something.

so corrections live in a separate `ReviewRecord`, keyed by run id and by a path into the
result (`profile.skills[2]`, `recommendation.assessments[R3]`). paths are validated when
the correction is recorded, so a correction can never point at something that does not
exist. `review_view()` then shows the original and the reviewer's note side by side.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.schemas import ReviewerCorrection, ReviewIssue, ReviewRecord, ScreeningError, ScreeningResult

logger = logging.getLogger(__name__)

REVIEW_FILE = "review_record.json"

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?$")

# which part of the result each issue type is about; used to catch a correction filed
# against the wrong section (an extraction complaint pointing at a matching result).
ISSUE_SCOPE: dict[str, str] = {
    "incorrect_extraction": "profile",
    "unsupported_qualification": "profile",
    "unverified_requirement": "recommendation",
    "ambiguous_evidence": "",  # may be either
    "incorrect_matching": "recommendation",
    "other": "",
}


class ReviewError(ScreeningError):
    """codes: unknown_target, run_id_mismatch, scope_mismatch."""


def new_review_record(run_id: str) -> ReviewRecord:
    return ReviewRecord(run_id=run_id)


def resolve_target(result: ScreeningResult, target: str) -> Any:
    """follow a dotted path into the result. raises ReviewError if it does not exist.

    supported: attribute access (`profile.full_name`), list index (`profile.skills[2]`)
    and assessment lookup by requirement id (`recommendation.assessments[R3]`).
    """
    node: Any = result
    for segment in target.split("."):
        match = _SEGMENT.match(segment)
        if not match:
            raise ReviewError("unknown_target", f"malformed path segment: {segment!r}")
        name, key = match.group(1), match.group(2)
        if not hasattr(node, name):
            raise ReviewError("unknown_target", f"{target!r}: no field {name!r}")
        node = getattr(node, name)
        if key is None:
            continue
        if not isinstance(node, list):
            raise ReviewError("unknown_target", f"{target!r}: {name!r} is not a list")
        if key.isdigit():
            index = int(key)
            if index >= len(node):
                raise ReviewError(
                    "unknown_target", f"{target!r}: index {index} out of range ({len(node)} items)"
                )
            node = node[index]
        else:  # requirement id, e.g. assessments[R3]
            matches = [item for item in node if getattr(item, "requirement_id", None) == key]
            if not matches:
                raise ReviewError("unknown_target", f"{target!r}: no requirement with id {key!r}")
            node = matches[0]
    return node


def record_correction(
    record: ReviewRecord,
    result: ScreeningResult,
    *,
    target: str,
    issue: ReviewIssue,
    reviewer_note: str,
    corrected_value: str | None = None,
    reviewer_id: str | None = None,
) -> ReviewRecord:
    """append one correction. the result is read for validation only and never modified."""
    if record.run_id != result.trace.run_id:
        raise ReviewError(
            "run_id_mismatch",
            f"record is for run {record.run_id}, result is run {result.trace.run_id}.",
        )
    resolve_target(result, target)  # raises if the path is wrong
    scope = ISSUE_SCOPE.get(issue, "")
    if scope and not target.startswith(scope):
        raise ReviewError(
            "scope_mismatch",
            f"issue {issue!r} belongs to the {scope!r} section, but the target is {target!r}.",
        )
    record.corrections.append(
        ReviewerCorrection(
            target=target,
            issue=issue,
            reviewer_note=reviewer_note,
            corrected_value=corrected_value,
            reviewer_id=reviewer_id,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )
    logger.info("correction recorded: run=%s issue=%s", record.run_id, issue)
    return record


def review_view(result: ScreeningResult, record: ReviewRecord) -> dict:
    """original value next to the reviewer's objection, for every correction.

    this is a read-only projection: both inputs come back unchanged, which is what makes
    the ai output auditable after review.
    """
    entries = []
    for correction in record.corrections:
        original = resolve_target(result, correction.target)
        entries.append({
            "target": correction.target,
            "issue": correction.issue,
            "ai_output": original.model_dump() if hasattr(original, "model_dump") else original,
            "reviewer_note": correction.reviewer_note,
            "reviewer_suggested_value": correction.corrected_value,
            "reviewer_id": correction.reviewer_id,
            "recorded_at": correction.recorded_at,
        })
    return {
        "run_id": record.run_id,
        "reviewer_decision": record.reviewer_decision,
        "ai_output_modified": record.ai_output_modified,
        "corrections": entries,
        "issue_counts": {
            issue: sum(1 for c in record.corrections if c.issue == issue)
            for issue in sorted({c.issue for c in record.corrections})
        },
        "notice": record.notice,
    }


def save_review_record(record: ReviewRecord, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REVIEW_FILE
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_review_record(directory: Path) -> ReviewRecord:
    path = directory / REVIEW_FILE
    return ReviewRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
