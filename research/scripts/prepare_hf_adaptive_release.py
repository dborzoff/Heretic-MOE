#!/usr/bin/env python3
"""Build text-free metadata for the public Heretic Adaptive model release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


BLIND_MAP = {
    "A": "original",
    "B": "max",
    "F": "balanced",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_study(journal: Path) -> optuna.study.Study:
    storage = JournalStorage(
        JournalFileBackend(
            str(journal),
            lock_obj=JournalFileOpenLock(str(journal)),
        )
    )
    summaries = optuna.get_all_study_summaries(storage=storage)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one study, found {len(summaries)}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def trial_record(study: optuna.study.Study, number: int) -> dict[str, object]:
    trial = study.trials[number]
    if trial.number != number:
        raise RuntimeError(f"Trial index mismatch for {number}")
    return {
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective_values": trial.values,
        "parameters": trial.params,
        "direction_index": trial.user_attrs.get("direction_index"),
        "abliteration_parameters": trial.user_attrs.get("parameters"),
    }


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ppl_raw = json.loads(args.ppl.read_text(encoding="utf-8"))
    ppl_by_label = {row["label"]: row for row in ppl_raw["models"]}
    ppl_summary = {
        "schema_version": 1,
        "method": {
            "dtype": ppl_raw["dtype"],
            "chunks": ppl_raw["chunks"],
            "window_tokens": ppl_raw["window"],
            "evaluated_tokens": ppl_by_label["original"]["target_tokens"],
            "dataset": "Salesforce/wikitext-2-raw-v1 test",
            "dataset_sha256": ppl_raw["dataset_sha256"],
        },
        "models": {
            "original": {
                "perplexity": ppl_by_label["original"]["perplexity"],
                "relative_change": 0.0,
            },
            "max": {
                "perplexity": ppl_by_label["trial_260"]["perplexity"],
                "relative_change": ppl_by_label["trial_260"][
                    "relative_to_baseline"
                ],
            },
            "balanced": {
                "perplexity": ppl_by_label["trial_488"]["perplexity"],
                "relative_change": ppl_by_label["trial_488"][
                    "relative_to_baseline"
                ],
            },
        },
        "text_free": True,
    }
    write_json(output / "evaluation" / "perplexity_400x512.json", ppl_summary)

    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected_rows: list[dict[str, object]] = []
    with args.labels.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            public_name = BLIND_MAP.get(row["system_code"])
            if public_name is None:
                continue
            clean = {
                "system": public_name,
                "id": int(row["id"]),
                "label": row["label"],
                "confidence": row.get("confidence"),
            }
            selected_rows.append(clean)
            label_counts[public_name][row["label"]] += 1
    selected_rows.sort(key=lambda row: (str(row["system"]), int(row["id"])))
    labels_out = output / "evaluation" / "semantic_labels.jsonl"
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    with labels_out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    expected = {"original": 136, "max": 136, "balanced": 136}
    observed = Counter(str(row["system"]) for row in selected_rows)
    if dict(observed) != expected:
        raise RuntimeError(f"Unexpected semantic coverage: {dict(observed)}")

    semantic_summary: dict[str, object] = {
        "schema_version": 1,
        "method": {
            "rows_per_system": 136,
            "max_new_tokens": 2048,
            "blind_semantic_judging": True,
            "labels_sha256": sha256(labels_out),
        },
        "systems": {},
        "text_free": True,
    }
    for system in ("original", "max", "balanced"):
        counts = label_counts[system]
        delivered = counts["comply"] + counts["comply_degraded"] + counts["partial"]
        evasion = counts["soft_substitution"] + counts["soft_legal"] + counts["soft_inversion"]
        policy = counts["refuse_policy"]
        semantic_summary["systems"][system] = {
            "counts": dict(sorted(counts.items())),
            "delivered_total": delivered,
            "evasion_total": evasion,
            "policy_refusal_total": policy,
            "residual_refusal_or_evasion_total": policy + evasion,
            "fail_incoherent_total": counts["fail_incoherent"],
        }
    write_json(output / "evaluation" / "semantic_summary.json", semantic_summary)

    study = load_study(args.journal)
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    selection = {
        "schema_version": 1,
        "method": "Heretic Adaptive v1",
        "base_model": "mistralai/Ministral-3-3B-Instruct-2512",
        "base_model_revision": "b35d4dfe56c142746f54dbd64f579faab2744308",
        "study": {
            "total_trials": len(study.trials),
            "completed_trials": completed,
            "directions": [direction.name for direction in study.directions],
            "journal_sha256": sha256(args.journal),
        },
        "variants": {
            "max": trial_record(study, 260),
            "balanced": trial_record(study, 488),
        },
        "text_free": True,
    }
    write_json(output / "study" / "selected_trials.json", selection)

    model_files: dict[str, list[dict[str, object]]] = {}
    for variant, root in (("max", args.max_model), ("balanced", args.balanced_model)):
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        model_files[variant] = files
    write_json(
        output / "model_files.json",
        {"schema_version": 1, "variants": model_files},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppl", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--max-model", type=Path, required=True)
    parser.add_argument("--balanced-model", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
