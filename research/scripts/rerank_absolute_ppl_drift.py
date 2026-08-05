#!/usr/bin/env python3
"""Re-rank a completed Heretic journal using absolute PPL drift.

The script reads Optuna metadata only. It never opens the trial-response
archive or any prompt/answer dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_records(trial: optuna.trial.FrozenTrial) -> dict[str, dict[str, Any]]:
    records = trial.user_attrs.get("scores", [])
    return {
        record["name"]: record["score"]
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("name"), str)
        and isinstance(record.get("score"), dict)
    }


def find_score(records: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [score for name, score in records.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one score starting with {prefix!r}, got {len(matches)}"
        )
    return matches[0]


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("geometry", "ppl_drift", "ltx_drift")
    front: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            weakly_better = all(other[key] <= candidate[key] for key in keys)
            strictly_better = any(other[key] < candidate[key] for key in keys)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def balanced_candidate(front: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("geometry", "ppl_drift", "ltx_drift")
    bounds = {
        key: (min(row[key] for row in front), max(row[key] for row in front))
        for key in keys
    }

    def cost(row: dict[str, Any]) -> tuple[float, float, float, int]:
        normalized = []
        for key in keys:
            lower, upper = bounds[key]
            normalized.append(
                0.0 if upper == lower else (row[key] - lower) / (upper - lower)
            )
        return (
            sum(value * value for value in normalized),
            row["geometry"],
            row["ppl_drift"],
            row["trial"],
        )

    return min(front, key=cost)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "trial",
            "optuna_trial",
            "geometry",
            "positive_count",
            "keywords_count",
            "ppl",
            "ppl_signed_change",
            "ppl_drift",
            "ppl_ci95_lower",
            "ppl_ci95_upper",
            "ltx_drift",
            "ltx_max_layer_drift",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppl-limit", type=float, default=0.005)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    storage = JournalStorage(JournalFileBackend(str(args.journal)))
    summaries = optuna.get_all_study_summaries(storage)
    if len(summaries) != 1:
        raise ValueError(f"Expected exactly one study, got {len(summaries)}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state is not TrialState.COMPLETE:
            continue
        records = score_records(trial)
        geometry = find_score(records, "Sparse refusal geometry")
        keywords = find_score(records, "Keywords")
        perplexity = find_score(records, "Perplexity")
        ltx = find_score(records, "LTX conditioning drift")
        ppl_signed = float(perplexity["value"])
        ppl_diag = perplexity.get("diagnostics") or {}
        geometry_diag = geometry.get("diagnostics") or {}
        keyword_diag = keywords.get("diagnostics") or {}
        ltx_diag = ltx.get("diagnostics") or {}
        rows.append(
            {
                "trial": int(trial.user_attrs.get("index", trial.number + 1)),
                "optuna_trial": trial.number,
                "geometry": float(geometry["value"]),
                "positive_count": int(geometry_diag.get("positive_count", -1)),
                "keywords_count": len(keyword_diag.get("matched_indices", [])),
                "ppl": float(ppl_diag.get("perplexity", float("nan"))),
                "ppl_signed_change": ppl_signed,
                "ppl_drift": abs(ppl_signed),
                "ppl_ci95_lower": float(
                    ppl_diag.get("relative_change_ci95_lower", float("nan"))
                ),
                "ppl_ci95_upper": float(
                    ppl_diag.get("relative_change_ci95_upper", float("nan"))
                ),
                "ltx_drift": float(ltx["value"]),
                "ltx_max_layer_drift": float(
                    ltx_diag.get("max_layer_cosine_drift", float("nan"))
                ),
                "old_feasible": bool(trial.user_attrs.get("feasible", False)),
                "corrected_feasible": abs(ppl_signed) <= args.ppl_limit,
                "params": trial.params,
            }
        )

    corrected = [row for row in rows if row["corrected_feasible"]]
    if not corrected:
        raise ValueError("No trials pass the corrected absolute PPL limit")
    front = pareto_front(corrected)
    by_removal = sorted(
        corrected,
        key=lambda row: (row["geometry"], row["ppl_drift"], row["ltx_drift"]),
    )
    by_preservation = sorted(
        corrected,
        key=lambda row: (row["ppl_drift"], row["geometry"], row["ltx_drift"]),
    )
    by_ltx = sorted(
        corrected,
        key=lambda row: (row["ltx_drift"], row["geometry"], row["ppl_drift"]),
    )

    report = {
        "schema_version": 1,
        "journal": str(args.journal.resolve()),
        "journal_sha256": sha256(args.journal),
        "study_name": study.study_name,
        "ppl_policy": {
            "formula": "abs(perplexity / baseline_perplexity - 1)",
            "limit": args.ppl_limit,
            "signed_change_retained_for_diagnostics": True,
        },
        "counts": {
            "complete": len(rows),
            "old_feasible": sum(row["old_feasible"] for row in rows),
            "corrected_feasible": len(corrected),
            "rejected_negative_beyond_limit": sum(
                row["ppl_signed_change"] < -args.ppl_limit for row in rows
            ),
            "rejected_positive_beyond_limit": sum(
                row["ppl_signed_change"] > args.ppl_limit for row in rows
            ),
            "corrected_pareto_front": len(front),
        },
        "roles": {
            "balanced": compact(balanced_candidate(front)),
            "maximum_removal": compact(by_removal[0]),
            "minimum_ppl_drift": compact(by_preservation[0]),
            "minimum_ltx_drift": compact(by_ltx[0]),
        },
        "top_by_removal": [compact(row) for row in by_removal[: args.top]],
        "top_by_ppl_preservation": [
            compact(row) for row in by_preservation[: args.top]
        ],
        "pareto_front": [
            compact(row)
            for row in sorted(
                front,
                key=lambda row: (
                    row["geometry"],
                    row["ppl_drift"],
                    row["ltx_drift"],
                ),
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(report["counts"], sort_keys=True))
    print(json.dumps(report["roles"], sort_keys=True))
    print(f"output={args.output}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
