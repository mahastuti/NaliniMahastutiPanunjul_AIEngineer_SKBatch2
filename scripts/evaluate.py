"""measured evaluation of the screening system.

what is measured per mode:
- always (deterministic, no api): matching-engine quality against hand-labeled cases,
  schema-validation reliability and grounding containment on simulated llm outputs,
  and end-to-end pipeline reliability on the synthetic cvs.

prompt-level evaluation (prompt construction, injection containment, narrative rule
enforcement, and live field-level extraction accuracy) lives in scripts/evaluate_prompts.py
so the two concerns are not duplicated.

usage: python scripts/evaluate.py [--mode mock|live] [--out outputs/evaluation_report.json]
"""

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_extractor import ground_profile  # noqa: E402
from src.pipeline import run_screening  # noqa: E402
from src.recommendation import build_recommendation, parse_job_description  # noqa: E402
from src.schemas import CandidateProfile, ScreeningError  # noqa: E402

# fixed so results are reproducible regardless of the day the script runs
EVAL_TODAY = date(2026, 11, 12)
STATUSES = ("met", "not_met", "unknown")


# ---------------------------------------------------------------- matching engine


def eval_matching() -> dict:
    spec = json.loads((ROOT / "data/evaluation/matching_cases.json").read_text(encoding="utf-8"))
    confusion = {s: {t: 0 for t in STATUSES} for s in STATUSES}  # expected -> predicted
    case_results, exact_cases = [], 0

    for case in spec["cases"]:
        profile = CandidateProfile.model_validate(case["profile"])
        jd_text = (ROOT / case["job_description_file"]).read_text(encoding="utf-8")
        rec = build_recommendation(profile, parse_job_description(jd_text), today=EVAL_TODAY)
        predicted = {a.requirement_id: a.status for a in rec.assessments}
        expected = case["expected"]
        mismatches = {r: {"expected": e, "predicted": predicted.get(r)}
                      for r, e in expected.items() if predicted.get(r) != e}
        for r, e in expected.items():
            if predicted.get(r) in STATUSES:
                confusion[e][predicted[r]] += 1
        exact_cases += not mismatches
        case_results.append({
            "case": case["name"],
            "signal": rec.signal,
            "n_labels": len(expected),
            "n_correct": len(expected) - len(mismatches),
            "exact_match": not mismatches,
            "mismatches": mismatches,
        })

    per_status = {}
    for s in STATUSES:
        tp = confusion[s][s]
        support = sum(confusion[s].values())
        predicted_s = sum(confusion[e][s] for e in STATUSES)
        per_status[s] = {
            "support": support,
            "recall": round(tp / support, 3) if support else None,
            "precision": round(tp / predicted_s, 3) if predicted_s else None,
        }
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[s][s] for s in STATUSES)
    return {
        "cases": case_results,
        "n_cases": len(case_results),
        "exact_match_cases": exact_cases,
        "requirement_level_accuracy": round(correct / total, 3),
        "n_requirement_labels": total,
        "per_status": per_status,
        "confusion_expected_vs_predicted": confusion,
    }


# ---------------------------------------------------------------- validation + grounding


def _item_keys(profile: CandidateProfile) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    keys.update(("skills", s.name) for s in profile.skills)
    keys.update(("certifications", c.name) for c in profile.certifications)
    keys.update(("work_experience", w.company or "") for w in profile.work_experience)
    keys.update(("education", e.institution or "") for e in profile.education)
    keys.update(("projects", p.name or "") for p in profile.projects)
    return keys


def eval_validation_and_grounding() -> dict:
    spec = json.loads((ROOT / "data/evaluation/simulated_llm_outputs.json").read_text(encoding="utf-8"))
    cv_text = spec["cv_text"]
    val_correct = 0
    fab_total = fab_dropped = genuine_total = genuine_kept = 0
    details = []

    for fx in spec["fixtures"]:
        try:
            profile = CandidateProfile.model_validate_json(fx["llm_output"])
            valid = True
        except ValidationError:
            profile, valid = None, False
        val_ok = valid == fx["expect_schema_valid"]
        val_correct += val_ok
        row = {"fixture": fx["name"], "schema_valid": valid, "validation_as_expected": val_ok}

        if valid and "items" in fx:
            grounded, issues = ground_profile(profile, cv_text)
            survived = _item_keys(grounded)
            for item in fx["items"]:
                key = (item["section"], item["key"])
                if item["fabricated"]:
                    fab_total += 1
                    fab_dropped += key not in survived
                else:
                    genuine_total += 1
                    genuine_kept += key in survived
            row["grounding_issues_reported"] = len(issues)
        details.append(row)

    return {
        "fixtures": details,
        "n_fixtures": len(spec["fixtures"]),
        "validation_decisions_correct": f"{val_correct}/{len(spec['fixtures'])}",
        "fabricated_items_total": fab_total,
        "fabricated_items_dropped": fab_dropped,
        "hallucination_containment_rate": round(fab_dropped / fab_total, 3) if fab_total else None,
        "genuine_items_total": genuine_total,
        "genuine_items_kept": genuine_kept,
        "genuine_retention_rate": round(genuine_kept / genuine_total, 3) if genuine_total else None,
    }


# ---------------------------------------------------------------- end-to-end reliability


def eval_pipeline_reliability(mode: str) -> dict:
    jd = (ROOT / "data/job_requirements/data_analyst.txt").read_text(encoding="utf-8")
    runs = []
    for pdf in sorted((ROOT / "data/sample_cvs").glob("*.pdf")):
        t0 = time.perf_counter()
        try:
            result = run_screening(pdf_bytes=pdf.read_bytes(), job_description=jd,
                                   mode=mode, today=EVAL_TODAY)
            runs.append({
                "cv": pdf.name, "ok": True, "seconds": round(time.perf_counter() - t0, 3),
                "signal": result.recommendation.signal,
                "extracted_items_survived": len(_item_keys(result.profile)),
                "requires_human_review": result.recommendation.requires_human_review,
            })
        except ScreeningError as exc:
            runs.append({"cv": pdf.name, "ok": False, "seconds": round(time.perf_counter() - t0, 3),
                         "error_code": exc.code})

    # structured-error paths must fail with a code, not crash or fabricate
    error_paths = []
    for name, data, expected in [
        ("not_a_pdf", b"plain text pretending", "unsupported_format"),
        ("corrupt_pdf", b"%PDF-1.7 broken body", "corrupt_pdf"),
        ("empty_file", b"", "empty_file"),
    ]:
        try:
            run_screening(pdf_bytes=data, job_description=jd, mode=mode, today=EVAL_TODAY)
            error_paths.append({"case": name, "structured_error": False, "got": "no error"})
        except ScreeningError as exc:
            error_paths.append({"case": name, "structured_error": exc.code == expected, "got": exc.code})

    return {
        "mode": mode,
        "runs": runs,
        "success_rate": f"{sum(r['ok'] for r in runs)}/{len(runs)}",
        "mean_seconds": round(sum(r["seconds"] for r in runs) / len(runs), 3) if runs else None,
        "error_paths": error_paths,
        "error_paths_correct": f"{sum(e['structured_error'] for e in error_paths)}/{len(error_paths)}",
    }


# ---------------------------------------------------------------- main


# ---------------------------------------------------------------- per-case csv


CSV_COLUMNS = ["section", "case", "metric", "value", "expected", "passed"]


def write_csv(report: dict, path: Path) -> Path:
    """flatten the report into one row per evaluated case.

    the json report is the full record; this file exists so the per-case results can be
    sorted, filtered and pasted into a table without a json parser. every row comes from
    the same run that produced the json — nothing here is re-derived by hand.
    """
    rows: list[dict] = []

    for case in report["matching_engine"]["cases"]:
        rows.append({
            "section": "matching", "case": case["case"],
            "metric": "requirement_labels_correct",
            "value": f"{case['n_correct']}/{case['n_labels']}",
            "expected": f"{case['n_labels']}/{case['n_labels']}",
            "passed": case["exact_match"],
        })
        rows.append({
            "section": "matching", "case": case["case"], "metric": "signal",
            "value": case["signal"], "expected": "", "passed": "",
        })

    for fixture in report["validation_and_grounding"]["fixtures"]:
        rows.append({
            "section": "validation", "case": fixture["fixture"],
            "metric": "schema_valid", "value": fixture["schema_valid"],
            "expected": fixture["schema_valid"] if fixture["validation_as_expected"] else "other",
            "passed": fixture["validation_as_expected"],
        })
        if "grounding_issues_reported" in fixture:
            rows.append({
                "section": "grounding", "case": fixture["fixture"],
                "metric": "grounding_issues_reported",
                "value": fixture["grounding_issues_reported"], "expected": "", "passed": "",
            })

    for run in report["pipeline_reliability"]["runs"]:
        rows.append({
            "section": "pipeline", "case": Path(run["cv"]).name, "metric": "end_to_end_success",
            "value": run["ok"], "expected": True, "passed": run["ok"],
        })
        rows.append({
            "section": "pipeline", "case": Path(run["cv"]).name, "metric": "seconds",
            "value": run["seconds"], "expected": "", "passed": "",
        })
        rows.append({
            "section": "pipeline", "case": Path(run["cv"]).name, "metric": "signal",
            "value": run["signal"], "expected": "", "passed": "",
        })

    for case in report["pipeline_reliability"]["error_paths"]:
        rows.append({
            "section": "error_handling", "case": case["case"], "metric": "error_code",
            "value": case["got"], "expected": case["case"],
            "passed": case["structured_error"],
        })

    matching = report["matching_engine"]
    summary = [
        ("matching", "requirement_level_accuracy", matching["requirement_level_accuracy"]),
        ("matching", "exact_match_cases",
         f"{matching['exact_match_cases']}/{matching['n_cases']}"),
        ("validation", "validation_decisions_correct",
         report["validation_and_grounding"]["validation_decisions_correct"]),
        ("grounding", "hallucination_containment_rate",
         report["validation_and_grounding"]["hallucination_containment_rate"]),
        ("grounding", "genuine_retention_rate",
         report["validation_and_grounding"]["genuine_retention_rate"]),
        ("pipeline", "success_rate", report["pipeline_reliability"]["success_rate"]),
        ("pipeline", "mean_seconds", report["pipeline_reliability"]["mean_seconds"]),
        ("error_handling", "error_paths_correct",
         report["pipeline_reliability"]["error_paths_correct"]),
    ]
    for section, metric, value in summary:
        rows.append({"section": section, "case": "ALL", "metric": metric,
                     "value": value, "expected": "", "passed": ""})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    # live mode reads GEMINI_API_KEY from .env, exactly like the pipeline cli does
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation_report.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs/evaluation_results.csv",
                        help="per-case results, flattened from the same run")
    args = parser.parse_args(argv)

    report = {
        "mode": args.mode,
        "eval_date_fixed": EVAL_TODAY.isoformat(),
        "matching_engine": eval_matching(),
        "validation_and_grounding": eval_validation_and_grounding(),
        "pipeline_reliability": eval_pipeline_reliability(args.mode),
        "extraction_quality": {
            "measured_elsewhere": "scripts/evaluate_prompts.py --mode live",
            "reason": "extraction quality depends on the prompt and the model; the mock profile "
                      "is a fixture, so scoring it here would be circular.",
        },
        "limitations": [
            "tiny hand-made dataset (5 matching cases, 6 simulated outputs, 4 synthetic cvs); results show behavior on covered scenarios, not general performance.",
            "ground truth was written by the same author as the rules, so matching metrics are regression checks, not an independent benchmark.",
            "simulated llm outputs test this system's defenses, not gemini's real error distribution.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m, v, r = report["matching_engine"], report["validation_and_grounding"], report["pipeline_reliability"]
    print(f"mode: {args.mode}")
    print(f"matching: requirement-level accuracy {m['requirement_level_accuracy']} "
          f"({m['n_requirement_labels']} labels), exact cases {m['exact_match_cases']}/{m['n_cases']}")
    for s in STATUSES:
        ps = m["per_status"][s]
        print(f"  {s:8} precision={ps['precision']} recall={ps['recall']} (n={ps['support']})")
    print(f"validation decisions correct: {v['validation_decisions_correct']}")
    print(f"hallucination containment: {v['fabricated_items_dropped']}/{v['fabricated_items_total']} "
          f"dropped; genuine retention: {v['genuine_items_kept']}/{v['genuine_items_total']}")
    print(f"pipeline: success {r['success_rate']}, mean {r['mean_seconds']}s, "
          f"error paths {r['error_paths_correct']}")
    print("extraction quality: measured by scripts/evaluate_prompts.py --mode live")
    csv_path = write_csv(report, args.csv)
    print(f"report written to {args.out}")
    print(f"per-case csv written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())