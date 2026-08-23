# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build text-free cross-language reports for a fixed model panel."""

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


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path.name}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row: {path.name}:{line_number}")
            yield value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def build_language_panel_report(
    selection_labels_path: str | Path,
    model_inputs: Sequence[Mapping[str, object]],
    output_dir: str | Path,
    *,
    languages: Sequence[str],
    support_threshold: int,
) -> dict[str, object]:
    selection_labels_path = Path(selection_labels_path).resolve()
    output_dir = Path(output_dir).resolve()
    selected_languages = tuple(str(value) for value in languages)
    panel_size = len(model_inputs)
    if (
        not selected_languages
        or len(set(selected_languages)) != len(selected_languages)
        or not 1 <= support_threshold <= panel_size
    ):
        raise ValueError("languages and support threshold must define a valid panel")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    labels = list(_read_jsonl(selection_labels_path))
    label_by_id: dict[str, dict[str, object]] = {}
    for row in labels:
        canonical_id = str(row["canonical_id"])
        if canonical_id in label_by_id:
            raise ValueError(f"duplicate selection label: {canonical_id}")
        target = str(row["target_behavior_class"])
        if target not in ("SOFT", "HARD_REFUSE"):
            raise ValueError(f"invalid target class: {target}")
        label_by_id[canonical_id] = row

    english_by_model: dict[str, dict[str, str]] = {}
    language_by_model: dict[str, dict[tuple[str, str], str]] = {}
    input_hashes = []
    for raw_input in model_inputs:
        model_id = str(raw_input["model_id"])
        if model_id in english_by_model:
            raise ValueError(f"duplicate model input: {model_id}")
        english_path = Path(raw_input["english_result_path"]).resolve()
        language_path = Path(raw_input["language_result_path"]).resolve()
        manifest_path = Path(raw_input["language_manifest_path"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_language_rows = len(labels) * len(selected_languages)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "PASS"
            or str(manifest["model_id"]) != model_id
            or int(manifest["rows"]) != expected_language_rows
            or get_file_sha256(language_path) != str(manifest["result_sha256"])
        ):
            raise ValueError(f"language model manifest validation failed: {model_id}")

        english: dict[str, str] = {}
        for row in _read_jsonl(english_path):
            canonical_id = str(row["canonical_id"])
            if canonical_id not in label_by_id:
                continue
            if canonical_id in english:
                raise ValueError(f"duplicate English vote: {model_id}/{canonical_id}")
            valid = bool(row.get("valid")) and row.get("classification") in _CLASSES
            english[canonical_id] = str(row["classification"]) if valid else "INVALID"
        if set(english) != set(label_by_id):
            raise ValueError(f"English vote coverage mismatch: {model_id}")

        multilingual: dict[tuple[str, str], str] = {}
        for row in _read_jsonl(language_path):
            canonical_id = str(row["canonical_id"])
            language = str(row["language"])
            if canonical_id not in label_by_id or language not in selected_languages:
                raise ValueError(f"unexpected language result row: {model_id}")
            key = canonical_id, language
            if key in multilingual:
                raise ValueError(f"duplicate language vote: {model_id}/{key}")
            valid = bool(row.get("valid")) and row.get("classification") in _CLASSES
            multilingual[key] = (
                str(row["classification"]) if valid else "INVALID"
            )
        expected_keys = {
            (canonical_id, language)
            for canonical_id in label_by_id
            for language in selected_languages
        }
        if set(multilingual) != expected_keys:
            raise ValueError(f"language vote coverage mismatch: {model_id}")
        english_by_model[model_id] = english
        language_by_model[model_id] = multilingual
        input_hashes.append(
            {
                "model_id": model_id,
                "english_result_sha256": get_file_sha256(english_path),
                "language_manifest_sha256": get_file_sha256(manifest_path),
                "language_result_sha256": get_file_sha256(language_path),
            }
        )

    support = {
        canonical_id: {language: 0 for language in selected_languages}
        for canonical_id in label_by_id
    }
    model_language_summary = []
    for model_id in english_by_model:
        for language in selected_languages:
            counts = Counter()
            target_agreement = 0
            english_agreement = 0
            english_comparable = 0
            for canonical_id, label in label_by_id.items():
                vote = language_by_model[model_id][(canonical_id, language)]
                english_vote = english_by_model[model_id][canonical_id]
                counts[vote] += 1
                target = str(label["target_behavior_class"])
                target_agreement += int(vote == target)
                if vote != "INVALID" and english_vote != "INVALID":
                    english_comparable += 1
                    english_agreement += int(vote == english_vote)
                support[canonical_id][language] += int(vote == target)
            model_language_summary.append(
                {
                    "model_id": model_id,
                    "language": language,
                    "rows": len(labels),
                    "valid": len(labels) - counts["INVALID"],
                    "invalid": counts["INVALID"],
                    "class_counts": {
                        **{name: counts[name] for name in _CLASSES},
                        "INVALID": counts["INVALID"],
                    },
                    "target_agreement": target_agreement,
                    "english_vote_comparable": english_comparable,
                    "english_vote_agreement": english_agreement,
                }
            )

    combined_rows = []
    confirmed_all = 0
    confirmed_two = 0
    language_distributions = {language: Counter() for language in selected_languages}
    category_groups: dict[tuple[str, str, str], list[int]] = {}
    source_groups: dict[tuple[str, str, str], list[int]] = {}
    for label in labels:
        canonical_id = str(label["canonical_id"])
        target = str(label["target_behavior_class"])
        values = support[canonical_id]
        confirmed = [
            language
            for language in selected_languages
            if values[language] >= support_threshold
        ]
        confirmed_all += int(len(confirmed) == len(selected_languages))
        confirmed_two += int(len(confirmed) >= 2)
        for language in selected_languages:
            language_distributions[language][values[language]] += 1
            for category in label.get("category_ids", []):
                item = category_groups.setdefault(
                    (str(category), language, target), [0, 0, 0]
                )
                item[0] += 1
                item[1] += values[language]
                item[2] += int(values[language] >= support_threshold)
            source_item = source_groups.setdefault(
                (str(label.get("source", "")), language, target), [0, 0, 0]
            )
            source_item[0] += 1
            source_item[1] += values[language]
            source_item[2] += int(values[language] >= support_threshold)
        combined_rows.append(
            {
                "canonical_id": canonical_id,
                "row_id": str(label["row_id"]),
                "target_behavior_class": target,
                "category_ids": list(label.get("category_ids", [])),
                "source": str(label.get("source", "")),
                "target_support": values,
                "confirmed_languages": confirmed,
            }
        )

    language_summary = {
        language: {
            "support_distribution": {
                str(key): value
                for key, value in sorted(language_distributions[language].items())
            },
            "confirmed": sum(
                value
                for key, value in language_distributions[language].items()
                if key >= support_threshold
            ),
        }
        for language in selected_languages
    }
    category_rows = [
        {
            "category_id": key[0],
            "language": key[1],
            "target_behavior_class": key[2],
            "rows": values[0],
            "target_support_sum": values[1],
            "confirmed": values[2],
        }
        for key, values in sorted(category_groups.items())
    ]
    source_rows = [
        {
            "source": key[0],
            "language": key[1],
            "target_behavior_class": key[2],
            "rows": values[0],
            "target_support_sum": values[1],
            "confirmed": values[2],
        }
        for key, values in sorted(source_groups.items())
    ]
    counts = {
        "rows": len(labels),
        "panel_size": panel_size,
        "support_threshold": support_threshold,
        "confirmed_all_languages": confirmed_all,
        "confirmed_at_least_two_languages": confirmed_two,
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "counts": counts,
        "languages": list(selected_languages),
        "language_summary": language_summary,
    }

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        outputs = (
            ("combined_rows.jsonl", combined_rows),
            ("model_language_summary.jsonl", model_language_summary),
            ("category_language_summary.jsonl", category_rows),
            ("source_language_summary.jsonl", source_rows),
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
            "counts": counts,
            "selection_labels_sha256": get_file_sha256(selection_labels_path),
            "model_inputs": input_hashes,
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
