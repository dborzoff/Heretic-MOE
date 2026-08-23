# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize private aligned clean HARD/SOFT datasets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
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
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def materialize_clean_candidate_datasets(
    aligned_manifest_path: str | Path,
    candidate_consensus_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    aligned_manifest_path = Path(aligned_manifest_path).resolve()
    candidate_consensus_path = Path(candidate_consensus_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    aligned = json.loads(aligned_manifest_path.read_text(encoding="utf-8"))
    if aligned.get("schema_version") != 1 or aligned.get("status") != "PASS":
        raise ValueError("aligned manifest is not a verified schema-v1 artifact")
    languages = [str(value) for value in aligned["languages"]]

    clean_ids = {"HARD_REFUSE": set(), "SOFT": set()}
    seen_ids: set[str] = set()
    for row in _read_jsonl(candidate_consensus_path):
        canonical_id = str(row["canonical_id"])
        if canonical_id in seen_ids:
            raise ValueError(f"duplicate consensus ID: {canonical_id}")
        seen_ids.add(canonical_id)
        if not bool(row["clean"]):
            continue
        target = str(row["target_behavior_class"])
        if target not in clean_ids:
            raise ValueError(f"unsupported clean target: {target}")
        clean_ids[target].add(canonical_id)
    if clean_ids["HARD_REFUSE"] & clean_ids["SOFT"]:
        raise ValueError("clean HARD/SOFT ID overlap")

    entries = {
        (str(entry["language"]), str(entry["direction"])): entry
        for entry in aligned.get("files", [])
    }
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        files = []
        reference_order: dict[str, tuple[str, ...]] = {}
        for language in languages:
            entry = entries[(language, "unsafe")]
            source_path = aligned_manifest_path.parent / str(entry["path"])
            if get_file_sha256(source_path) != str(entry["sha256"]):
                raise ValueError(f"aligned dataset hash drift: {source_path.name}")
            source_rows = _read_jsonl(source_path)
            if len(source_rows) != int(entry["rows"]):
                raise ValueError(f"aligned dataset row count drift: {source_path.name}")
            source_ids = {str(row["canonical_id"]) for row in source_rows}
            if (clean_ids["HARD_REFUSE"] | clean_ids["SOFT"]) - source_ids:
                raise ValueError(f"clean IDs missing from aligned language: {language}")
            for target, label in (("HARD_REFUSE", "hard"), ("SOFT", "soft")):
                selected = [
                    row
                    for row in source_rows
                    if str(row["canonical_id"]) in clean_ids[target]
                ]
                order = tuple(str(row["canonical_id"]) for row in selected)
                previous = reference_order.setdefault(target, order)
                if order != previous:
                    raise ValueError(f"clean aligned order drift: {language}/{label}")
                path = temporary / f"clean_{label}_{language}.jsonl"
                _write_jsonl(path, selected)
                files.append(
                    {
                        "language": language,
                        "target": target,
                        "path": path.name,
                        "rows": len(selected),
                        "sha256": get_file_sha256(path),
                    }
                )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "private_text": True,
            "languages": languages,
            "clean_hard_per_language": len(clean_ids["HARD_REFUSE"]),
            "clean_soft_per_language": len(clean_ids["SOFT"]),
            "aligned_manifest_sha256": get_file_sha256(aligned_manifest_path),
            "candidate_consensus_sha256": get_file_sha256(candidate_consensus_path),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
