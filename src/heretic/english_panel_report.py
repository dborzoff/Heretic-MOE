# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build text-free reports for an English 10+3 model vote panel."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .utils import get_file_sha256

_CLASSES = ("SOFT", "HARD_REFUSE", "DIRECT", "PARTIAL")


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


def build_english_panel_report(
    original_consensus_path: str | Path,
    selection_labels_path: str | Path,
    additional_result_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    original_panel_size: int,
) -> dict[str, object]:
    original_consensus_path = Path(original_consensus_path).resolve()
    selection_labels_path = Path(selection_labels_path).resolve()
    output_dir = Path(output_dir).resolve()
    result_dirs = [Path(value).resolve() for value in additional_result_dirs]
    if original_panel_size < 1 or len(result_dirs) != 3:
        raise ValueError("the report requires the original panel plus three models")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    labels = _read_jsonl(selection_labels_path)
    label_by_id: dict[str, dict[str, object]] = {}
    for label in labels:
        canonical_id = str(label["canonical_id"])
        if canonical_id in label_by_id:
            raise ValueError(f"duplicate selection label: {canonical_id}")
        target = str(label["target_behavior_class"])
        if target not in ("SOFT", "HARD_REFUSE"):
            raise ValueError(f"unsupported target behavior class: {target}")
        if int(label["panel_size"]) != original_panel_size:
            raise ValueError("selection panel size drift")
        label_by_id[canonical_id] = label

    original_by_id: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(original_consensus_path):
        canonical_id = str(row["canonical_id"])
        if canonical_id in label_by_id:
            original_by_id[canonical_id] = row
    if set(original_by_id) != set(label_by_id):
        raise ValueError("original consensus does not cover every selected ID")

    additional_by_model: dict[str, dict[str, dict[str, object]]] = {}
    model_inputs = []
    for result_dir in result_dirs:
        manifest_path = result_dir / "manifest.json"
        rows_path = result_dir / "rows.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "PASS"
            or manifest.get("system_mode") != "english"
            or int(manifest["rows"]) != len(labels)
            or get_file_sha256(rows_path) != str(manifest["result_sha256"])
        ):
            raise ValueError(f"additional model manifest failed validation: {result_dir}")
        model_id = str(manifest["model_id"])
        if model_id in additional_by_model:
            raise ValueError(f"duplicate additional model ID: {model_id}")
        values: dict[str, dict[str, object]] = {}
        for row in _read_jsonl(rows_path):
            canonical_id = str(row["canonical_id"])
            if canonical_id in values:
                raise ValueError(f"duplicate additional result ID: {model_id}/{canonical_id}")
            if str(row["model_id"]) != model_id or str(row["language"]) != "en":
                raise ValueError("additional result model or language drift")
            values[canonical_id] = row
        if set(values) != set(label_by_id):
            raise ValueError(f"additional result coverage drift: {model_id}")
        additional_by_model[model_id] = values
        model_inputs.append(
            {
                "model_id": model_id,
                "manifest_sha256": get_file_sha256(manifest_path),
                "result_sha256": get_file_sha256(rows_path),
                "variant": str(manifest["variants"][0]),
            }
        )

    combined_rows: list[dict[str, object]] = []
    model_stats: dict[str, dict[str, object]] = {
        model_id: {
            "model_id": model_id,
            "rows": 0,
            "valid": 0,
            "invalid": 0,
            "target_agreement": 0,
            "target_disagreement": 0,
            "class_counts": {**{name: 0 for name in _CLASSES}, "INVALID": 0},
        }
        for model_id in additional_by_model
    }
    support_distribution = Counter()
    for label in labels:
        canonical_id = str(label["canonical_id"])
        original = original_by_id[canonical_id]
        raw_counts = original.get("model_vote_counts")
        if not isinstance(raw_counts, dict):
            raise TypeError("original model vote counts must be an object")
        counts = {name: int(raw_counts.get(name, 0)) for name in _CLASSES}
        counts["INVALID"] = int(raw_counts.get("INVALID", 0))
        if sum(counts.values()) != original_panel_size:
            raise ValueError(f"original vote accounting drift: {canonical_id}")
        target = str(label["target_behavior_class"])
        new_votes: dict[str, str] = {}
        for model_id, values in additional_by_model.items():
            row = values[canonical_id]
            classification = row.get("classification")
            valid = bool(row.get("valid")) and classification in _CLASSES
            vote = str(classification) if valid else "INVALID"
            new_votes[model_id] = vote
            counts[vote] += 1
            stats = model_stats[model_id]
            stats["rows"] = int(stats["rows"]) + 1
            stats["valid"] = int(stats["valid"]) + int(valid)
            stats["invalid"] = int(stats["invalid"]) + int(not valid)
            stats["class_counts"][vote] += 1
            stats["target_agreement"] = int(stats["target_agreement"]) + int(
                valid and vote == target
            )
            stats["target_disagreement"] = int(stats["target_disagreement"]) + int(
                valid and vote != target
            )
        target_support = counts[target]
        support_distribution[target_support] += 1
        combined_rows.append(
            {
                "canonical_id": canonical_id,
                "row_id": str(label["row_id"]),
                "target_behavior_class": target,
                "category_ids": list(label.get("category_ids", [])),
                "source": str(label.get("source", "")),
                "original_vote_counts": {
                    name: int(raw_counts.get(name, 0))
                    for name in (*_CLASSES, "INVALID")
                },
                "additional_votes": new_votes,
                "combined_vote_counts": counts,
                "target_support": target_support,
                "panel_size": original_panel_size + len(result_dirs),
            }
        )

    category_rows = []
    category_groups: dict[tuple[str, str], list[int]] = {}
    source_groups: dict[tuple[str, str], list[int]] = {}
    for row in combined_rows:
        target = str(row["target_behavior_class"])
        support = int(row["target_support"])
        for category in row["category_ids"]:
            values = category_groups.setdefault((str(category), target), [0, 0])
            values[0] += 1
            values[1] += support
        source_values = source_groups.setdefault((str(row["source"]), target), [0, 0])
        source_values[0] += 1
        source_values[1] += support
    for (category, target), values in sorted(category_groups.items()):
        category_rows.append(
            {
                "category_id": category,
                "target_behavior_class": target,
                "rows": values[0],
                "target_support_sum": values[1],
            }
        )
    source_rows = [
        {
            "source": source,
            "target_behavior_class": target,
            "rows": values[0],
            "target_support_sum": values[1],
        }
        for (source, target), values in sorted(source_groups.items())
    ]
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(combined_rows),
        "panel_size": original_panel_size + len(result_dirs),
        "target_counts": dict(Counter(str(row["target_behavior_class"]) for row in combined_rows)),
        "target_support_distribution": {
            str(key): value for key, value in sorted(support_distribution.items())
        },
        "additional_models": list(model_stats.values()),
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        outputs = (
            ("combined_rows.jsonl", combined_rows),
            ("model_summary.jsonl", list(model_stats.values())),
            ("category_summary.jsonl", category_rows),
            ("source_summary.jsonl", source_rows),
        )
        files = []
        for name, rows in outputs:
            path = temporary / name
            _write_jsonl(path, rows)
            files.append(
                {"path": name, "rows": len(rows), "sha256": get_file_sha256(path)}
            )
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        files.append(
            {
                "path": summary_path.name,
                "rows": 1,
                "sha256": get_file_sha256(summary_path),
            }
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "PASS",
            "private_text": False,
            "counts": {
                "rows": len(combined_rows),
                "SOFT": sum(row["target_behavior_class"] == "SOFT" for row in combined_rows),
                "HARD_REFUSE": sum(
                    row["target_behavior_class"] == "HARD_REFUSE"
                    for row in combined_rows
                ),
            },
            "original_consensus_sha256": get_file_sha256(original_consensus_path),
            "selection_labels_sha256": get_file_sha256(selection_labels_path),
            "additional_model_inputs": model_inputs,
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
