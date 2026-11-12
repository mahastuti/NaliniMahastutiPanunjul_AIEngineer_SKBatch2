"""orchestration: pdf -> text -> extraction -> grounding -> recommendation -> one result."""

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from src.llm_extractor import extract_profile
from src.pdf_parser import extract_text
from src.recommendation import build_recommendation, parse_job_description
from src.schemas import Caveat, Mode, Provenance, ScreeningError, ScreeningResult

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(previous|above|system)",
    r"system prompt",
    r"you are (now )?(an? )?(ai|assistant|chatgpt|gpt|llm)\b",
    r"(rate|score|rank) (this|the) (candidate|cv|resume)",
    r"(recommend|hire) this candidate",
]


class PipelineError(ScreeningError):
    """codes: no_requirements."""


def run_screening(
    *,
    pdf_bytes: bytes,
    job_description: str,
    mode: Mode = "mock",
    api_key: str | None = None,
    model: str | None = None,
    today: date | None = None,
    client: Any | None = None,
) -> ScreeningResult:
    """run the full flow in memory. nothing is written to disk and no cv text is logged."""
    # step 1: parse the job description first, so bad input fails before any api call
    job = parse_job_description(job_description)
    if not job.requirements:
        raise PipelineError("no_requirements", " ".join(job.warnings))

    # step 2: pdf -> text (raises PDFParseError)
    parsed = extract_text(pdf_bytes)

    # step 3: extraction + grounding (raises ExtractionError)
    extraction = extract_profile(parsed.text, mode=mode, api_key=api_key, model=model, client=client)

    # step 4: caveats that come from the process rather than from the profile
    extra: list[Caveat] = []
    if mode == "mock":
        for warning in extraction.warnings:
            extra.append(Caveat(code="mock_mode", severity="warning", message=warning))
    for issue in extraction.grounding_issues:
        extra.append(Caveat(code="grounding_issue", severity="warning", message=issue))
    if _looks_like_injection(parsed.text):
        extra.append(Caveat(
            code="possible_prompt_injection", severity="warning",
            message="the cv contains phrases that look like instructions to an ai system. they were treated as data and had no effect on the comparison, but review the cv manually.",
        ))

    # step 5: deterministic comparison
    recommendation = build_recommendation(extraction.profile, job, today=today, extra_caveats=extra)

    logger.info(
        "screening done: mode=%s requirements=%d signal=%s",
        mode, len(job.requirements), recommendation.signal,
    )
    return ScreeningResult(
        mode=mode,
        provenance=Provenance(
            profile_source="mock_canned" if mode == "mock" else "gemini_llm", model=extraction.model
        ),
        job=job,
        profile=extraction.profile,
        recommendation=recommendation,
        notices=[
            "llm output is not verified fact: every extracted item carries evidence quotes that were checked against the cv text, but a human must confirm them.",
            recommendation.decision_notice,
        ],
    )


def _looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in _INJECTION_PATTERNS)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="screen a cv pdf against a job description")
    parser.add_argument("--cv", required=True, type=Path)
    parser.add_argument("--jd", required=True, type=Path, help="text file with the job description")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--out", type=Path, help="optional path to write the json result")
    args = parser.parse_args(argv)

    if args.mode == "mock":
        print("MOCK MODE: canned demo output, no model is called.", file=sys.stderr)
    try:
        result = run_screening(
            pdf_bytes=args.cv.read_bytes(),
            job_description=args.jd.read_text(encoding="utf-8"),
            mode=args.mode,
        )
    except (ScreeningError, OSError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"error [{code}]: {exc}", file=sys.stderr)
        return 1

    output = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
