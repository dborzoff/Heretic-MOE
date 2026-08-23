# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare compact aligned jobs for invalid multilingual classification cells."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def collect_invalid_retry_results(
    retry_jobs_manifest_path: str | Path,
    retry_plan_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    retry_jobs_manifest_path = Path(retry_jobs_manifest_path).resolve()
    retry_plan_path = Path(retry_plan_path).resolve()
    output_path = Path(output_path).resolve()
    manifest = json.loads(retry_jobs_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise ValueError("retry job manifest is not a verified schema-v1 artifact")
    if get_file_sha256(retry_plan_path) != str(manifest["retry_plan_sha256"]):
        raise ValueError("retry plan hash drift")

    result_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for relative_job in manifest.get("job_files", []):
        job_path = retry_jobs_manifest_path.parent / str(relative_job)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        model_id = str(job["model_id"])
        result_path = Path(str(job["output_path"]))
        for row in _read_jsonl(result_path):
            if str(row["model_id"]) != model_id or str(row["variant"]) != "number":
                raise ValueError("retry result model or variant drift")
            key = model_id, str(row["row_id"])
            if key in result_by_key:
                raise ValueError(f"duplicate retry result key: {key}")
            result_by_key[key] = row

    collected: list[dict[str, object]] = []
    for planned in _read_jsonl(retry_plan_path):
        key = str(planned["model_id"]), str(planned["row_id"])
        if key not in result_by_key:
            raise ValueError(f"missing retry result: {key}")
        raw = result_by_key[key]
        collected.append(
            {
                "canonical_id": str(planned["canonical_id"]),
                "row_id": str(planned["row_id"]),
                "model_id": str(planned["model_id"]),
                "language": str(planned["language"]),
                "classification": raw.get("classification"),
                "valid": bool(raw.get("valid")),
                "replaces_variant": str(planned["replaces_variant"]),
                "retry_variant": "number",
                "output_tokens": int(raw.get("output_tokens", 0)),
                "output_shape": str(raw.get("output_shape", "")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    _write_jsonl(temporary, collected)
    os.replace(temporary, output_path)
    valid = sum(bool(row["valid"]) for row in collected)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "private_text": False,
        "retry_cells": len(collected),
        "valid": valid,
        "invalid": len(collected) - valid,
        "result_sha256": get_file_sha256(output_path),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def materialize_invalid_retry_jobs(
    run_plan_path: str | Path,
    retry_plan_path: str | Path,
    aligned_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    run_plan_path = Path(run_plan_path).resolve()
    retry_plan_path = Path(retry_plan_path).resolve()
    aligned_manifest_path = Path(aligned_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    aligned_manifest = json.loads(aligned_manifest_path.read_text(encoding="utf-8"))
    if run_plan.get("schema_version") != 1 or run_plan.get("status") != "PASS":
        raise ValueError("run plan is not a verified schema-v1 artifact")
    if (
        aligned_manifest.get("schema_version") != 1
        or aligned_manifest.get("status") != "PASS"
    ):
        raise ValueError("aligned manifest is not a verified schema-v1 artifact")
    languages = [str(value) for value in run_plan["languages"]]
    if languages != [str(value) for value in aligned_manifest["languages"]]:
        raise ValueError("run plan and aligned manifest language order differ")
    model_entries = {
        str(entry["model_id"]): entry for entry in run_plan.get("models", [])
    }
    retry_rows = _read_jsonl(retry_plan_path)
    seen_retry: set[tuple[str, str, str]] = set()
    retry_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in retry_rows:
        key = str(row["model_id"]), str(row["row_id"]), str(row["replaces_variant"])
        if key in seen_retry:
            raise ValueError(f"duplicate retry plan key: {key}")
        seen_retry.add(key)
        if key[0] not in model_entries or str(row["language"]) not in languages:
            raise ValueError("retry plan contains an unknown model or language")
        retry_by_model[key[0]].append(row)

    file_entries = {
        (str(entry["language"]), str(entry["direction"])): entry
        for entry in aligned_manifest.get("files", [])
    }
    source_by_language: dict[str, dict[str, dict[str, object]]] = {}
    canonical_order: list[str] = []
    for language in languages:
        entry = file_entries[(language, "unsafe")]
        path = aligned_manifest_path.parent / str(entry["path"])
        if get_file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"aligned dataset hash drift: {path.name}")
        values = _read_jsonl(path)
        if len(values) != int(entry["rows"]):
            raise ValueError(f"aligned dataset row count drift: {path.name}")
        mapping = {str(row["canonical_id"]): row for row in values}
        if len(mapping) != len(values):
            raise ValueError(f"aligned dataset duplicate ID: {path.name}")
        source_by_language[language] = mapping
        order = [str(row["canonical_id"]) for row in values]
        if not canonical_order:
            canonical_order = order
        elif order != canonical_order:
            raise ValueError("aligned dataset canonical order drift")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        job_files: list[str] = []
        union_rows = 0
        for model_id in model_entries:
            planned = retry_by_model.get(model_id, [])
            if not planned:
                continue
            selected_ids = {str(row["canonical_id"]) for row in planned}
            ordered_ids = [value for value in canonical_order if value in selected_ids]
            if len(ordered_ids) != len(selected_ids):
                raise ValueError(f"retry IDs missing from aligned dataset: {model_id}")
            safe_name = _safe_name(model_id)
            dataset_root = temporary / "datasets" / safe_name
            dataset_files = []
            for language in languages:
                safe_path = dataset_root / f"direction_{language}_safe.jsonl"
                unsafe_path = dataset_root / f"direction_{language}_unsafe.jsonl"
                _write_jsonl(safe_path, [])
                selected = [source_by_language[language][value] for value in ordered_ids]
                _write_jsonl(unsafe_path, selected)
                for direction, path, count in (
                    ("safe", safe_path, 0),
                    ("unsafe", unsafe_path, len(selected)),
                ):
                    dataset_files.append(
                        {
                            "language": language,
                            "direction": direction,
                            "path": path.name,
                            "rows": count,
                            "sha256": get_file_sha256(path),
                        }
                    )
            dataset_manifest = {
                "schema_version": 1,
                "status": "PASS",
                "private_text": True,
                "languages": languages,
                "directions": {"safe": 0, "unsafe": len(ordered_ids)},
                "files": dataset_files,
            }
            dataset_manifest_path = dataset_root / "manifest.json"
            dataset_manifest_path.write_text(
                json.dumps(
                    dataset_manifest, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            final_dataset_manifest_path = (
                output_dir / dataset_manifest_path.relative_to(temporary)
            )
            source_job_path = Path(str(model_entries[model_id]["job"]))
            source_job = json.loads(source_job_path.read_text(encoding="utf-8"))
            job = dict(source_job)
            job.update(
                {
                    "schema_version": 1,
                    "dataset_manifest": str(final_dataset_manifest_path),
                    "languages": languages,
                    "variants": ["number"],
                    "system_mode": "english",
                    "max_new_tokens": 8,
                    "output_path": str(
                        output_dir / "results" / safe_name / "rows.jsonl"
                    ),
                    "shard_index": 0,
                    "shard_count": 1,
                }
            )
            job_path = temporary / "jobs" / f"{safe_name}.json"
            job_path.parent.mkdir(parents=True, exist_ok=True)
            job_path.write_text(
                json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            job_files.append(str(job_path.relative_to(temporary)))
            union_rows += len(ordered_ids) * len(languages)

        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "private_text": True,
            "retry_cells": len(retry_rows),
            "jobs": len(job_files),
            "aligned_union_rows": union_rows,
            "languages": languages,
            "job_files": job_files,
            "retry_plan": str(retry_plan_path),
            "retry_plan_sha256": get_file_sha256(retry_plan_path),
            "aligned_manifest_sha256": get_file_sha256(aligned_manifest_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise
