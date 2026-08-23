# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI for final multilingual HARD/SOFT candidate consensus."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .multilingual_candidate_consensus import (
    build_candidate_consensus,
    build_invalid_retry_plan,
    write_candidate_consensus_artifacts,
)
from .utils import get_file_sha256


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path.name}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path.name}:{line_number}")
            rows.append(value)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE candidate-consensus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-plan", required=True, type=Path)
    aggregate.add_argument("--target-labels", required=True, type=Path)
    aggregate.add_argument("--metadata-en", required=True, type=Path)
    aggregate.add_argument("--retry-results", type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    retries = subparsers.add_parser("plan-retries")
    retries.add_argument("--run-plan", required=True, type=Path)
    retries.add_argument("--output-dir", required=True, type=Path)
    return parser


def _load_run_rows(
    run_plan: Path,
) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
    plan = json.loads(run_plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("status") != "PASS":
        raise ValueError("run plan is not a verified schema-v1 artifact")
    model_entries = plan.get("models")
    if not isinstance(model_entries, list) or len(model_entries) < 6:
        raise ValueError("run plan must contain at least six models")
    expected_rows = int(plan["rows_per_model"])
    model_ids: list[str] = []
    rows: list[dict[str, object]] = []
    for entry in model_entries:
        if not isinstance(entry, dict):
            raise TypeError("run plan model entry must be an object")
        model_id = str(entry["model_id"])
        path = Path(str(entry["output"]))
        values = _read_jsonl(path)
        if int(entry["expected_rows"]) != expected_rows or len(values) != expected_rows:
            raise ValueError(f"model result coverage mismatch: {model_id}")
        model_ids.append(model_id)
        rows.extend(values)

    return plan, model_ids, rows


def _aggregate(args: argparse.Namespace) -> dict[str, object]:
    plan, model_ids, rows = _load_run_rows(args.run_plan)
    targets = _read_jsonl(args.target_labels)
    metadata_by_id: dict[str, dict[str, object]] = {}
    for raw in _read_jsonl(args.metadata_en):
        canonical_id = str(raw["canonical_id"])
        if canonical_id in metadata_by_id:
            raise ValueError(f"duplicate metadata ID: {canonical_id}")
        metadata_by_id[canonical_id] = raw
    target_ids = {str(target["canonical_id"]) for target in targets}
    if target_ids - set(metadata_by_id):
        raise ValueError("English metadata does not cover all target IDs")
    enriched = []
    for target in targets:
        value = dict(target)
        metadata = metadata_by_id[str(target["canonical_id"])]
        value["source"] = str(metadata.get("source", ""))
        enriched.append(value)

    report = build_candidate_consensus(
        rows,
        enriched,
        model_ids=model_ids,
        languages=[str(value) for value in plan["languages"]],
        variants=[str(value) for value in plan["variants"]],
        retry_rows=(
            _read_jsonl(args.retry_results) if args.retry_results is not None else ()
        ),
    )
    report["inputs"] = {
        "run_plan_sha256": get_file_sha256(args.run_plan),
        "target_labels_sha256": get_file_sha256(args.target_labels),
        "metadata_en_sha256": get_file_sha256(args.metadata_en),
        "retry_results_sha256": (
            get_file_sha256(args.retry_results)
            if args.retry_results is not None
            else None
        ),
        "model_result_sha256": {
            str(entry["model_id"]): get_file_sha256(Path(str(entry["output"])))
            for entry in plan["models"]
        },
    }
    return write_candidate_consensus_artifacts(args.output_dir, report)


def _plan_retries(args: argparse.Namespace) -> dict[str, object]:
    plan, model_ids, rows = _load_run_rows(args.run_plan)
    retry_rows = build_invalid_retry_plan(
        rows,
        model_ids=model_ids,
        languages=[str(value) for value in plan["languages"]],
        variants=[str(value) for value in plan["variants"]],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retry_plan.jsonl"
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in retry_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output_path)
    by_model: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for row in retry_rows:
        model_id = str(row["model_id"])
        language = str(row["language"])
        by_model[model_id] = by_model.get(model_id, 0) + 1
        by_language[language] = by_language.get(language, 0) + 1
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "private_text": False,
        "retry_cells": len(retry_rows),
        "by_model": dict(sorted(by_model.items())),
        "by_language": dict(sorted(by_language.items())),
        "retry_plan_sha256": get_file_sha256(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "aggregate":
        result = _aggregate(args)
    elif args.command == "plan-retries":
        result = _plan_retries(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
