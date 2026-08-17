# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize a text-private classification view of the frozen v4 corpus."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .utils import get_file_sha256

LANGUAGES = ("en", "ru", "zh", "ja")
POOLS = ("map", "trial", "final")
DIRECTIONS = ("safe", "unsafe")


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


def _source_entry(
    files: dict[str, object],
    *,
    pool: str,
    language: str,
    direction: str,
) -> tuple[str, dict[str, object]]:
    matches = [
        (name, value)
        for name, value in files.items()
        if isinstance(value, dict)
        and value.get("pool") == pool
        and value.get("language") == language
        and value.get("direction_class") == direction
    ]
    if len(matches) != 1:
        raise ValueError(
            "source manifest file coverage mismatch: "
            f"{pool}/{language}/{direction}"
        )
    return matches[0]


def materialize_vote_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    languages: Sequence[str] = LANGUAGES,
) -> dict[str, Any]:
    """Concatenate map, trial, and final pools into a verified schema-v1 view."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    normalized_languages = tuple(str(value).strip().lower() for value in languages)
    if normalized_languages != LANGUAGES:
        raise ValueError(f"vote dataset languages must be {LANGUAGES}")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 4:
        raise ValueError("source manifest is not schema version 4")
    if source_manifest.get("status") != "PASS":
        raise ValueError("source manifest is not PASS")
    if tuple(source_manifest.get("languages", ())) != normalized_languages:
        raise ValueError("source manifest language coverage mismatch")
    if any(int(value) != 0 for value in source_manifest["cross_pool_overlap"].values()):
        raise ValueError("source manifest has cross-pool ID overlap")
    content_overlap = source_manifest["cross_pool_content_overlap"]
    if any(
        int(value) != 0
        for direction in DIRECTIONS
        for value in content_overlap[direction].values()
    ):
        raise ValueError("source manifest has cross-pool content overlap")
    files = source_manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("source manifest files must be an object")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        public_files: list[dict[str, object]] = []
        direction_counts: dict[str, int] = {}
        canonical_order: dict[str, tuple[str, ...]] = {}
        for language in normalized_languages:
            for direction in DIRECTIONS:
                combined: list[dict[str, object]] = []
                for pool in POOLS:
                    name, entry = _source_entry(
                        files,
                        pool=pool,
                        language=language,
                        direction=direction,
                    )
                    source_path = (source_root / name).resolve()
                    if not source_path.is_relative_to(source_root):
                        raise ValueError("source manifest file escapes its root")
                    if get_file_sha256(source_path) != str(entry["sha256"]):
                        raise ValueError(f"source file hash drift: {name}")
                    rows = _read_jsonl(source_path)
                    if len(rows) != int(entry["rows"]):
                        raise ValueError(f"source row count drift: {name}")
                    for row in rows:
                        if str(row.get("language", "")).lower() != language:
                            raise ValueError(f"source language drift: {name}")
                        if str(row.get("direction_class", "")).lower() != direction:
                            raise ValueError(f"source direction drift: {name}")
                        category_ids = row.get("category_ids")
                        if not isinstance(category_ids, list) or not category_ids:
                            raise ValueError(f"source category coverage missing: {name}")
                        value = dict(row)
                        value["source_pool"] = pool
                        combined.append(value)

                ids = tuple(str(row["canonical_id"]) for row in combined)
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate canonical IDs: {language}/{direction}")
                expected_order = canonical_order.setdefault(direction, ids)
                if expected_order != ids:
                    raise ValueError(f"cross-language ID order drift: {language}/{direction}")
                direction_counts.setdefault(direction, len(combined))
                if direction_counts[direction] != len(combined):
                    raise ValueError(f"cross-language row count drift: {language}/{direction}")

                output_name = f"vote_{language}_{direction}_{len(combined)}.jsonl"
                output_path = temporary / output_name
                output_path.write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in combined
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                public_files.append(
                    {
                        "language": language,
                        "direction": direction,
                        "path": output_name,
                        "rows": len(combined),
                        "sha256": get_file_sha256(output_path),
                    }
                )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "source_manifest_sha256": get_file_sha256(source_manifest_path),
            "source_contract_sha256": source_manifest.get("contract_sha256"),
            "source_pools": list(POOLS),
            "languages": list(normalized_languages),
            "directions": direction_counts,
            "canonical_rows": sum(direction_counts.values()),
            "rows": sum(direction_counts.values()) * len(normalized_languages),
            "files": public_files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

