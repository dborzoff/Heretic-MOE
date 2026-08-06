#!/usr/bin/env python3
"""Re-rank Gemma Optuna journals with the fixed Qwen-night preservation cost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FrozenTrial, TrialState

GEOMETRY_TARGET = -0.0088
KEYWORDS_TARGET = 2 / 136
PPL_TARGET = 0.0
PPL_LIMIT = 0.005
GEOMETRY_WEIGHT = 344.0
KEYWORDS_WEIGHT = 697.68
PPL_WEIGHT = 200.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_record(trial: FrozenTrial, *names: str) -> dict[str, Any] | None:
    for record in trial.user_attrs.get("scores", []):
        if record.get("name") in names:
            score = record.get("score")
            if isinstance(score, dict):
                return score
    return None


def score_value(record: dict[str, Any] | None) -> float | None:
    if record is None:
        return None
    value = record.get("value")
    return float(value) if isinstance(value, int | float) else None


def diagnostic_number(record: dict[str, Any] | None, name: str) -> float | None:
    if record is None:
        return None
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    value = diagnostics.get(name)
    return float(value) if isinstance(value, int | float) else None


def is_dominated(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    for other in rows:
        if other is row:
            continue
        no_worse = (
            other["geometry"] <= row["geometry"]
            and other["ppl_drift"] <= row["ppl_drift"]
        )
        strictly_better = (
            other["geometry"] < row["geometry"]
            or other["ppl_drift"] < row["ppl_drift"]
        )
        if no_worse and strictly_better:
            return True
    return False


def trial_row(trial: FrozenTrial) -> dict[str, Any] | None:
    geometry_record = score_record(trial, "Sparse refusal geometry")
    keyword_record = score_record(trial, "Keywords")
    ppl_record = score_record(
        trial, "Perplexity increase", "Perplexity drift", "PPL drift"
    )
    geometry = score_value(geometry_record)
    keywords = score_value(keyword_record)
    ppl_recorded = score_value(ppl_record)
    if geometry is None or keywords is None or ppl_recorded is None:
        return None

    ppl_signed = diagnostic_number(ppl_record, "relative_change")
    if ppl_signed is None:
        ppl_signed = ppl_recorded
    ppl_drift = abs(ppl_signed)
    feasible = ppl_drift <= PPL_LIMIT
    cost = (
        GEOMETRY_WEIGHT * max(0.0, geometry - GEOMETRY_TARGET)
        + KEYWORDS_WEIGHT * max(0.0, keywords - KEYWORDS_TARGET)
        + PPL_WEIGHT * max(0.0, ppl_drift - PPL_TARGET)
    )
    return {
        "optuna_trial": trial.number,
        "trial": int(trial.user_attrs.get("index", trial.number + 1)),
        "state": trial.state.name,
        "source_feasible": bool(trial.user_attrs.get("feasible", False)),
        "feasible": feasible,
        "geometry": geometry,
        "positive_count": diagnostic_number(geometry_record, "positive_count"),
        "keywords": keywords,
        "keywords_count": diagnostic_number(keyword_record, "rows") * keywords
        if diagnostic_number(keyword_record, "rows") is not None
        else None,
        "ppl": diagnostic_number(ppl_record, "perplexity"),
        "ppl_signed_change": ppl_signed,
        "ppl_drift": ppl_drift,
        "calibrated_cost": cost,
        "params": trial.params,
    }


def rerank(journal: Path, output_dir: Path) -> dict[str, Any]:
    journal_hash = sha256(journal)
    storage = JournalStorage(JournalFileBackend(str(journal)))
    summaries = optuna.get_all_study_summaries(storage)
    if not summaries:
        return {
            "status": "SKIP",
            "journal": str(journal.resolve()),
            "journal_sha256": journal_hash,
            "reason": "no Optuna study found",
        }
    if len(summaries) != 1:
        raise ValueError(f"Expected one study in {journal}, got {len(summaries)}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    rows = [
        row
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if (row := trial_row(trial)) is not None
    ]
    if not rows:
        return {
            "status": "SKIP",
            "journal": str(journal.resolve()),
            "journal_sha256": journal_hash,
            "reason": "required numeric scores not found",
        }

    feasible_rows = [row for row in rows if row["feasible"]]
    feasible_rows.sort(
        key=lambda row: (
            row["calibrated_cost"],
            row["geometry"],
            row["ppl_drift"],
            row["optuna_trial"],
        )
    )
    for rank, row in enumerate(feasible_rows, start=1):
        row["new_rank"] = rank
        row["pareto"] = not is_dominated(row, feasible_rows)
    for row in rows:
        if not row["feasible"]:
            row["new_rank"] = None
            row["pareto"] = False

    target = output_dir / journal_hash[:16]
    target.mkdir(parents=True, exist_ok=True)
    rows_path = target / "all_trials.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: item["optuna_trial"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "status": "PASS",
        "journal": str(journal.resolve()),
        "journal_sha256": journal_hash,
        "study_name": study.study_name,
        "directions": [direction.name for direction in study.directions],
        "formula": {
            "targets": {
                "geometry": GEOMETRY_TARGET,
                "keywords": KEYWORDS_TARGET,
                "ppl_drift": PPL_TARGET,
            },
            "weights": {
                "geometry": GEOMETRY_WEIGHT,
                "keywords": KEYWORDS_WEIGHT,
                "ppl_drift": PPL_WEIGHT,
            },
            "ppl_limit": PPL_LIMIT,
        },
        "counts": {
            "complete": len(study.get_trials(states=(TrialState.COMPLETE,))),
            "scored": len(rows),
            "feasible": len(feasible_rows),
            "pareto": sum(bool(row["pareto"]) for row in feasible_rows),
        },
        "all_trials": str(rows_path.resolve()),
        "all_trials_sha256": sha256(rows_path),
        "top": feasible_rows[:20],
    }
    report_path = target / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report"] = str(report_path.resolve())
    report["report_sha256"] = sha256(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    reports: list[dict[str, Any]] = []
    for journal in args.journal:
        journal_hash = sha256(journal)
        if journal_hash in seen:
            continue
        seen.add(journal_hash)
        try:
            reports.append(rerank(journal, args.output_dir))
        except (KeyError, RuntimeError, ValueError) as error:
            reports.append(
                {
                    "status": "ERROR",
                    "journal": str(journal.resolve()),
                    "journal_sha256": journal_hash,
                    "error": type(error).__name__,
                    "message": str(error),
                }
            )

    summary = {
        "status": "PASS"
        if all(report["status"] != "ERROR" for report in reports)
        else "FAIL",
        "journal_inputs": len(args.journal),
        "unique_journals": len(seen),
        "reports": reports,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "unique_journals": len(seen),
                "pass": sum(report["status"] == "PASS" for report in reports),
                "skip": sum(report["status"] == "SKIP" for report in reports),
                "error": sum(report["status"] == "ERROR" for report in reports),
                "summary": str(args.summary.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
