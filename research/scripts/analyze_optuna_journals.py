#!/usr/bin/env python3
"""Compare successful and destructive Heretic trials using text-free journals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FrozenTrial, TrialState


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def normalized_param(trial: FrozenTrial, name: str) -> float | None:
    value = safe_float(trial.params.get(name))
    if value is None:
        return None
    distribution = trial.distributions.get(name)
    if isinstance(distribution, (FloatDistribution, IntDistribution)):
        low = float(distribution.low)
        high = float(distribution.high)
        if high > low:
            return (value - low) / (high - low)
    return value


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2
        for position in range(index, end):
            result[order[position]] = rank
        index = end
    return result


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float | None:
    return correlation(ranks(left), ranks(right))


def sample_evenly(values: list[float], maximum: int = 128) -> list[float]:
    if len(values) <= maximum:
        return values
    return [values[round(index * (len(values) - 1) / (maximum - 1))] for index in range(maximum)]


def cliffs_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left = sample_evenly(sorted(left))
    right = sample_evenly(sorted(right))
    greater = 0
    lower = 0
    for x in left:
        for y in right:
            greater += x > y
            lower += x < y
    return (greater - lower) / (len(left) * len(right))


def minimize_values(trial: FrozenTrial, directions: list[optuna.study.StudyDirection]) -> list[float]:
    assert trial.values is not None
    return [
        float(value) if direction.name == "MINIMIZE" else -float(value)
        for value, direction in zip(trial.values, directions)
    ]


def nondominated(
    trials: list[FrozenTrial], directions: list[optuna.study.StudyDirection]
) -> list[FrozenTrial]:
    result: list[FrozenTrial] = []
    vectors = {trial.number: minimize_values(trial, directions) for trial in trials}
    for trial in trials:
        current = vectors[trial.number]
        dominated = False
        for other in trials:
            if other.number == trial.number:
                continue
            candidate = vectors[other.number]
            if all(a <= b for a, b in zip(candidate, current)) and any(
                a < b for a, b in zip(candidate, current)
            ):
                dominated = True
                break
        if not dominated:
            result.append(trial)
    return result


def elite_trials(
    acceptable: list[FrozenTrial],
    directions: list[optuna.study.StudyDirection],
    maximum: int = 32,
) -> list[FrozenTrial]:
    front = nondominated(acceptable, directions)
    if len(front) >= 5:
        return sorted(front, key=lambda trial: trial.number)[:maximum]
    if not acceptable:
        return []
    objective_count = len(acceptable[0].values or [])
    objective_ranks: list[dict[int, float]] = []
    for objective_index in range(objective_count):
        values = [minimize_values(trial, directions)[objective_index] for trial in acceptable]
        ranked = ranks(values)
        objective_ranks.append(
            {trial.number: rank for trial, rank in zip(acceptable, ranked)}
        )
    target = min(maximum, max(5, math.ceil(len(acceptable) * 0.1)))
    return sorted(
        acceptable,
        key=lambda trial: (
            sum(ranking[trial.number] for ranking in objective_ranks),
            trial.number,
        ),
    )[:target]


def param_contrast(
    left: list[FrozenTrial],
    right: list[FrozenTrial],
    all_trials: list[FrozenTrial],
) -> list[dict[str, Any]]:
    names = sorted(set.intersection(*(set(trial.params) for trial in all_trials))) if all_trials else []
    output: list[dict[str, Any]] = []
    for name in names:
        left_values = [normalized_param(trial, name) for trial in left]
        right_values = [normalized_param(trial, name) for trial in right]
        left_numeric = [value for value in left_values if value is not None]
        right_numeric = [value for value in right_values if value is not None]
        if len(left_numeric) < 3 or len(right_numeric) < 3:
            continue
        objective0_x: list[float] = []
        objective0_y: list[float] = []
        objective1_y: list[float] = []
        for trial in all_trials:
            value = normalized_param(trial, name)
            if value is None or trial.values is None:
                continue
            objective0_x.append(value)
            objective0_y.append(float(trial.values[0]))
            if len(trial.values) > 1:
                objective1_y.append(float(trial.values[1]))
        left_median = statistics.median(left_numeric)
        right_median = statistics.median(right_numeric)
        output.append(
            {
                "parameter": name,
                "left_n": len(left_numeric),
                "right_n": len(right_numeric),
                "left_median_normalized": left_median,
                "right_median_normalized": right_median,
                "median_shift": left_median - right_median,
                "cliffs_delta": cliffs_delta(left_numeric, right_numeric),
                "spearman_objective0": spearman(objective0_x, objective0_y),
                "spearman_objective1": (
                    spearman(objective0_x, objective1_y)
                    if len(objective1_y) == len(objective0_x)
                    else None
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: abs(float(row["cliffs_delta"] or 0.0)),
        reverse=True,
    )


def telemetry_scalars(trial: FrozenTrial) -> dict[str, float]:
    telemetry = trial.user_attrs.get("telemetry")
    if not isinstance(telemetry, dict):
        return {}
    result: dict[str, float] = {}
    runtime = safe_float(telemetry.get("runtime_seconds"))
    if runtime is not None:
        result["runtime_seconds"] = runtime
    edit = telemetry.get("edit")
    if isinstance(edit, dict) and isinstance(edit.get("total"), dict):
        for name, value in edit["total"].items():
            numeric = safe_float(value)
            if numeric is not None:
                result[f"edit.total.{name}"] = numeric
    peaks = telemetry.get("cuda_peaks")
    if isinstance(peaks, list) and peaks:
        allocated = [
            safe_float(item.get("max_allocated_bytes"))
            for item in peaks
            if isinstance(item, dict)
        ]
        allocated = [value for value in allocated if value is not None]
        if allocated:
            result["cuda.max_allocated_bytes"] = max(allocated)
    return result


def analyze_study(
    study: optuna.study.Study,
    journal: Path,
    relative_path: str,
    ppl_cap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.values is not None
        and len(trial.values) >= 2
        and all(math.isfinite(float(value)) for value in trial.values)
    ]
    acceptable = [trial for trial in complete if float(trial.values[1]) <= ppl_cap]
    quality_fail = [trial for trial in complete if float(trial.values[1]) > ppl_cap]
    elite = elite_trials(acceptable, list(study.directions))
    acceptable_sorted = sorted(acceptable, key=lambda trial: float(trial.values[0]))
    quartile = max(3, len(acceptable_sorted) // 4)
    low_refusal = acceptable_sorted[:quartile]
    high_refusal = acceptable_sorted[-quartile:] if acceptable_sorted else []
    feasible_front = nondominated(acceptable, list(study.directions))
    telemetry_trials = [trial for trial in complete if telemetry_scalars(trial)]
    telemetry_names = sorted(
        {name for trial in telemetry_trials for name in telemetry_scalars(trial)}
    )
    telemetry_report: list[dict[str, Any]] = []
    for name in telemetry_names:
        pairs = [
            (telemetry_scalars(trial).get(name), trial)
            for trial in telemetry_trials
        ]
        pairs = [(value, trial) for value, trial in pairs if value is not None]
        if len(pairs) < 3:
            continue
        values = [float(value) for value, _ in pairs]
        telemetry_report.append(
            {
                "metric": name,
                "n": len(values),
                "spearman_objective0": spearman(
                    values, [float(trial.values[0]) for _, trial in pairs]
                ),
                "spearman_objective1": spearman(
                    values, [float(trial.values[1]) for _, trial in pairs]
                ),
            }
        )
    best = sorted(
        acceptable,
        key=lambda trial: (float(trial.values[0]), float(trial.values[1]), trial.number),
    )[:10]
    report = {
        "journal": relative_path,
        "journal_sha256": sha256(journal),
        "study_name": study.study_name,
        "directions": [direction.name for direction in study.directions],
        "trials_total": len(study.trials),
        "states": dict(Counter(trial.state.name for trial in study.trials)),
        "complete_numeric": len(complete),
        "ppl_cap": ppl_cap,
        "acceptable": len(acceptable),
        "quality_fail": len(quality_fail),
        "feasible_front": len(feasible_front),
        "elite": len(elite),
        "telemetry_trials": len(telemetry_trials),
        "best_under_cap": [
            {"trial": trial.number, "values": [float(value) for value in trial.values]}
            for trial in best
        ],
        "elite_vs_quality_fail": param_contrast(elite, quality_fail, complete),
        "low_vs_high_refusal_under_cap": param_contrast(
            low_refusal, high_refusal, acceptable
        ),
        "telemetry_correlations": telemetry_report,
    }
    effects: list[dict[str, Any]] = []
    for contrast_name in ("elite_vs_quality_fail", "low_vs_high_refusal_under_cap"):
        for row in report[contrast_name]:
            effects.append(
                {
                    "journal": relative_path,
                    "contrast": contrast_name,
                    **row,
                }
            )
    return report, effects


def aggregate_effects(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        grouped[(row["contrast"], row["parameter"])].append(row)
    output: list[dict[str, Any]] = []
    for (contrast, parameter), rows in grouped.items():
        deltas = [float(row["cliffs_delta"]) for row in rows if row["cliffs_delta"] is not None]
        shifts = [float(row["median_shift"]) for row in rows]
        if not deltas:
            continue
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        output.append(
            {
                "contrast": contrast,
                "parameter": parameter,
                "journals": len(rows),
                "median_cliffs_delta": statistics.median(deltas),
                "median_normalized_shift": statistics.median(shifts),
                "sign_consistency": max(positive, negative) / len(deltas),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["journals"]),
            float(row["sign_consistency"]),
            abs(float(row["median_cliffs_delta"])),
        ),
        reverse=True,
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Text-free Heretic journal analysis",
        "",
        "This report compares numeric objectives and parameters only. It does not contain prompts or responses.",
        "",
        f"PPL cap: `{report['ppl_cap']}`",
        "",
        "## Journals",
        "",
        "| Journal | Complete | Under cap | PPL failures | Pareto under cap | Telemetry |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["journals"]:
        lines.append(
            f"| `{row['journal']}` | {row['complete_numeric']} | {row['acceptable']} | "
            f"{row['quality_fail']} | {row['feasible_front']} | {row['telemetry_trials']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-journal parameter signals",
            "",
            "These are associations, not causal effects. Layer-position parameters are normalized to their declared search bounds.",
            "",
            "| Contrast | Parameter | Journals | Median Cliff delta | Sign consistency |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["aggregate_effects"][:40]:
        lines.append(
            f"| {row['contrast']} | `{row['parameter']}` | {row['journals']} | "
            f"{row['median_cliffs_delta']:.3f} | {row['sign_consistency']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppl-cap", type=float, default=0.005)
    args = parser.parse_args()

    candidates = sorted(
        path
        for path in args.runs_root.rglob("*.jsonl")
        if "checkpoints" in path.parts
        and not path.name.endswith((".importance.json", ".merge.json"))
    )
    journals: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for journal in candidates:
        digest = sha256(journal)
        if digest in seen_hashes:
            skipped.append(
                {
                    "journal": str(journal.relative_to(args.runs_root)),
                    "reason": "exact_duplicate",
                }
            )
            continue
        seen_hashes.add(digest)
        try:
            storage = JournalStorage(JournalFileBackend(str(journal)))
            summaries = optuna.get_all_study_summaries(storage=storage)
            if not summaries:
                raise ValueError("no_study")
            for summary in summaries:
                study = optuna.load_study(
                    storage=storage, study_name=summary.study_name
                )
                relative = str(journal.relative_to(args.runs_root))
                if len(summaries) > 1:
                    relative += f"#{summary.study_name}"
                journal_report, journal_effects = analyze_study(
                    study, journal, relative, args.ppl_cap
                )
                journals.append(journal_report)
                effects.extend(journal_effects)
        except Exception as error:
            skipped.append(
                {
                    "journal": str(journal.relative_to(args.runs_root)),
                    "reason": type(error).__name__,
                }
            )

    report = {
        "schema_version": 1,
        "ppl_cap": args.ppl_cap,
        "journals": journals,
        "skipped": skipped,
        "aggregate_effects": aggregate_effects(effects),
        "text_emitted": False,
        "limitations": [
            "Objective scales can differ across datasets and scorer configurations.",
            "Cross-journal parameter effects are associative and not causal.",
            "Ancestor and descendant journals can share trials even when file hashes differ.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "journal_good_bad_analysis.json"
    md_path = args.output_dir / "journal_good_bad_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(report, md_path)
    print(
        f"journals={len(journals)} skipped={len(skipped)} "
        f"effects={len(report['aggregate_effects'])}"
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
