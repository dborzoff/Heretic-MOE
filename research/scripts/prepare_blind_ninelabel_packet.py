#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build a text-safe blind judging packet from aligned response archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def parse_archive(spec: str) -> tuple[str, Path]:
    label, separator, path_text = spec.partition("=")
    if not separator or not label or not path_text:
        raise ValueError(f"Invalid archive specification: {spec!r}")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    archives = [parse_archive(spec) for spec in args.archive]
    labels = [label for label, _ in archives]
    if len(labels) < 2 or len(labels) > 26 or len(labels) != len(set(labels)):
        raise ValueError("Archive labels must be unique; supported count is 2..26")
    if not args.criteria.is_file():
        raise FileNotFoundError(args.criteria)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    archive_rows: dict[str, list[dict[str, object]]] = {}
    archive_paths = dict(archives)
    expected_ids: list[int] | None = None
    reference_prompts: dict[int, str] | None = None

    for label, path in archives:
        rows = read_jsonl(path)
        ids = [int(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate ids in archive {label}")
        ordered_ids = sorted(ids)
        if expected_ids is None:
            expected_ids = ordered_ids
            if expected_ids != list(range(len(expected_ids))):
                raise RuntimeError("Reference archive ids must be contiguous from zero")
            reference_prompts = {
                int(row["id"]): str(row["prompt"]) for row in rows
            }
        elif ordered_ids != expected_ids:
            raise RuntimeError(f"Coverage mismatch in archive {label}")

        by_id = {int(row["id"]): row for row in rows}
        assert reference_prompts is not None
        for row_id in ordered_ids:
            row = by_id[row_id]
            prompt = str(row.get("prompt", ""))
            answer = str(row.get("answer", ""))
            if not prompt.strip() or not answer.strip():
                raise RuntimeError(f"Empty prompt or answer in {label}, id={row_id}")
            if prompt != reference_prompts[row_id]:
                raise RuntimeError(f"Prompt identity mismatch in {label}, id={row_id}")
        archive_rows[label] = [by_id[row_id] for row_id in ordered_ids]

    assert expected_ids is not None
    rng = random.Random(args.seed)
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)
    codes = [chr(ord("A") + index) for index in range(len(labels))]
    code_to_label = dict(zip(codes, shuffled_labels, strict=True))

    combined: list[dict[str, object]] = []
    for code, label in code_to_label.items():
        for row in archive_rows[label]:
            combined.append(
                {
                    "system_code": code,
                    "id": int(row["id"]),
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                }
            )
    rng.shuffle(combined)

    judge_dir = args.output_root / "judge_packet"
    private_dir = args.output_root / "private"
    work_dir = args.output_root / "judge_work"
    judge_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    input_path = judge_dir / "rejudge-input.jsonl"
    with input_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    criteria_path = judge_dir / "JUDGING_CRITERIA.md"
    shutil.copyfile(args.criteria, criteria_path)

    expected_rows = len(labels) * len(expected_ids)
    manifest = {
        "schema_version": 1,
        "task": "blind_ninelabel_semantic_judgment",
        "input": input_path.name,
        "input_sha256": sha256(input_path),
        "criteria": criteria_path.name,
        "criteria_sha256": sha256(criteria_path),
        "system_codes": codes,
        "systems": len(codes),
        "rows_per_system": len(expected_ids),
        "expected_rows": expected_rows,
        "input_fields": ["system_code", "id", "prompt", "answer"],
        "unique_key": ["system_code", "id"],
        "output_fields": ["system_code", "id", "label", "confidence"],
        "allowed_labels": [
            "comply",
            "comply_degraded",
            "partial",
            "soft_substitution",
            "soft_legal",
            "soft_inversion",
            "insufficient_info",
            "refuse_policy",
            "fail_incoherent",
        ],
        "output_path": str((work_dir / "cjudge_labels.jsonl").resolve()),
        "text_must_not_be_reported": True,
        "blind_map_outside_judge_packet": True,
        "status": "PASS",
    }
    manifest_path = judge_dir / "manifest.json"
    write_json(manifest_path, manifest)

    private_map = {
        "schema_version": 1,
        "seed": args.seed,
        "judge_input_sha256": sha256(input_path),
        "mapping": [
            {
                "system_code": code,
                "label": label,
                "archive": str(archive_paths[label]),
                "archive_sha256": sha256(archive_paths[label]),
                "rows": len(expected_ids),
            }
            for code, label in code_to_label.items()
        ],
    }
    private_map_path = private_dir / "blind-map.json"
    write_json(private_map_path, private_map)

    report = {
        "status": "PASS",
        "systems": len(codes),
        "rows_per_system": len(expected_ids),
        "expected_rows": expected_rows,
        "unique_keys": len({(row["system_code"], row["id"]) for row in combined}),
        "prompt_identity": True,
        "empty_prompt_or_answer": 0,
        "judge_manifest": str(manifest_path.resolve()),
        "judge_manifest_sha256": sha256(manifest_path),
        "judge_input_sha256": sha256(input_path),
        "private_map": str(private_map_path.resolve()),
        "private_map_sha256": sha256(private_map_path),
        "text_free_report": True,
    }
    write_json(args.output_root / "prepare-report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
