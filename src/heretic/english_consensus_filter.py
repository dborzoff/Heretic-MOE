# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize cumulative English SOFT/HARD consensus pools."""

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


def materialize_cumulative_vote_pool(
    source_manifest_path: str | Path,
    consensus_rows_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_votes: int,
    panel_size: int,
) -> dict[str, object]:
    source_manifest_path = Path(source_manifest_path).resolve()
    consensus_rows_path = Path(consensus_rows_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not 1 <= int(minimum_votes) <= int(panel_size):
        raise ValueError("minimum votes must be within the panel")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != 1
        or source_manifest.get("status") != "PASS"
        or source_manifest.get("languages") != ["en"]
    ):
        raise ValueError("source manifest is not a verified English schema-v1 pool")
    entries = {
        (str(entry["language"]), str(entry["direction"])): entry
        for entry in source_manifest.get("files", [])
    }
    safe_entry = entries[("en", "safe")]
    unsafe_entry = entries[("en", "unsafe")]
    safe_source = source_manifest_path.parent / str(safe_entry["path"])
    unsafe_source = source_manifest_path.parent / str(unsafe_entry["path"])
    for entry, path in ((safe_entry, safe_source), (unsafe_entry, unsafe_source)):
        if get_file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"source dataset hash drift: {path.name}")
    safe_rows = _read_jsonl(safe_source)
    source_rows = _read_jsonl(unsafe_source)
    if len(safe_rows) != int(safe_entry["rows"]) or len(source_rows) != int(
        unsafe_entry["rows"]
    ):
        raise ValueError("source dataset row count drift")
    source_by_id = {str(row["canonical_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source dataset contains duplicate canonical IDs")

    consensus_by_id: dict[str, dict[str, object]] = {}
    labels_by_id: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(consensus_rows_path):
        canonical_id = str(row["canonical_id"])
        if canonical_id in consensus_by_id:
            raise ValueError(f"duplicate consensus ID: {canonical_id}")
        counts = row.get("model_vote_counts")
        if not isinstance(counts, dict):
            raise TypeError("model vote counts must be an object")
        valid_votes = int(row["valid_model_votes"])
        soft_votes = int(counts.get("SOFT", 0))
        hard_votes = int(counts.get("HARD_REFUSE", 0))
        invalid_votes = int(counts.get("INVALID", 0))
        classified_votes = sum(
            int(value) for key, value in counts.items() if str(key) != "INVALID"
        )
        if (
            not 0 <= valid_votes <= panel_size
            or soft_votes < 0
            or hard_votes < 0
            or invalid_votes < 0
            or classified_votes != valid_votes
            or valid_votes + invalid_votes != panel_size
        ):
            raise ValueError(f"invalid vote accounting: {canonical_id}")
        soft_selected = soft_votes >= minimum_votes
        hard_selected = hard_votes >= minimum_votes
        if soft_selected and hard_selected:
            raise ValueError(f"ambiguous SOFT/HARD threshold result: {canonical_id}")
        consensus_by_id[canonical_id] = row
        if soft_selected or hard_selected:
            labels_by_id[canonical_id] = {
                "canonical_id": canonical_id,
                "row_id": str(row["row_id"]),
                "target_behavior_class": "SOFT" if soft_selected else "HARD_REFUSE",
                "soft_votes": soft_votes,
                "hard_votes": hard_votes,
                "valid_model_votes": valid_votes,
                "panel_size": panel_size,
                "category_ids": list(row.get("category_ids", [])),
                "source": str(row.get("source", "")),
            }
    if set(consensus_by_id) != set(source_by_id):
        raise ValueError("consensus coverage does not match the source pool")

    selected_rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for source in source_rows:
        canonical_id = str(source["canonical_id"])
        label = labels_by_id.get(canonical_id)
        if label is None:
            continue
        selected = dict(source)
        selected["target_behavior_class"] = label["target_behavior_class"]
        selected["english_soft_votes"] = label["soft_votes"]
        selected["english_hard_votes"] = label["hard_votes"]
        selected["english_panel_size"] = panel_size
        selected_rows.append(selected)
        labels.append(label)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        safe_path = temporary / "direction_en_safe_strict.jsonl"
        unsafe_path = temporary / "direction_en_unsafe_strict.jsonl"
        labels_path = temporary / "selection_labels.jsonl"
        _write_jsonl(safe_path, [])
        _write_jsonl(unsafe_path, selected_rows)
        _write_jsonl(labels_path, labels)
        counts = {
            "source": len(source_rows),
            "selected": len(selected_rows),
            "rejected": len(source_rows) - len(selected_rows),
            "SOFT": sum(
                row["target_behavior_class"] == "SOFT" for row in labels
            ),
            "HARD_REFUSE": sum(
                row["target_behavior_class"] == "HARD_REFUSE" for row in labels
            ),
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "private_text": True,
            "languages": ["en"],
            "minimum_votes": int(minimum_votes),
            "panel_size": int(panel_size),
            "directions": {"safe": 0, "unsafe": len(selected_rows)},
            "counts": counts,
            "source_manifest_sha256": get_file_sha256(source_manifest_path),
            "consensus_rows_sha256": get_file_sha256(consensus_rows_path),
            "selection_labels": labels_path.name,
            "selection_labels_sha256": get_file_sha256(labels_path),
            "files": [
                {
                    "language": "en",
                    "direction": "safe",
                    "path": safe_path.name,
                    "rows": 0,
                    "sha256": get_file_sha256(safe_path),
                },
                {
                    "language": "en",
                    "direction": "unsafe",
                    "path": unsafe_path.name,
                    "rows": len(selected_rows),
                    "sha256": get_file_sha256(unsafe_path),
                },
            ],
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
