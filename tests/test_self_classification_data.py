from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heretic.self_classification import (
    BehaviorClass,
    ClassificationResult,
    PromptVariant,
)
from heretic.self_classification_data import (
    append_result_atomic,
    load_classification_rows,
    load_completed_keys,
    verify_result_coverage,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path) -> Path:
    files = []
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = root / f"direction_{language}_{direction}.jsonl"
            row = {
                "canonical_id": f"{direction}-1",
                "row_id": f"{language.upper()}-{direction}-1",
                "language": language,
                "direction_class": direction,
                "category_id": "C1",
                "category_ids": ["C1"],
                "prompt": f"PRIVATE_{language}_{direction}_SENTINEL",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            files.append(
                {
                    "language": language,
                    "direction": direction,
                    "path": path.name,
                    "rows": 1,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "canonical_rows": 2,
        "directions": {"safe": 1, "unsafe": 1},
        "languages": ["en", "ru"],
        "rows": 4,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def result(row_id: str = "EN-safe-1") -> ClassificationResult:
    return ClassificationResult(
        model_id="model-a",
        canonical_id="safe-1",
        row_id=row_id,
        language="en",
        category_ids=("C1",),
        direction_class="safe",
        variant=PromptVariant.NUMBER,
        classification=BehaviorClass.DIRECT,
        valid=True,
        output_tokens=1,
    )


def test_manifest_loader_validates_hashes_and_preserves_alignment(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)

    rows = load_classification_rows(manifest, ("en", "ru"))

    assert len(rows) == 4
    assert [row.row_id for row in rows] == [
        "EN-safe-1",
        "EN-unsafe-1",
        "RU-safe-1",
        "RU-unsafe-1",
    ]
    assert {row.canonical_id for row in rows} == {"safe-1", "unsafe-1"}
    assert all("PRIVATE_" in row.prompt for row in rows)


def test_manifest_loader_rejects_file_hash_drift(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    (tmp_path / "direction_en_safe.jsonl").write_text("tampered\n")

    with pytest.raises(ValueError, match="hash drift"):
        load_classification_rows(manifest, ("en", "ru"))


def test_checkpoint_contains_only_public_result_fields(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    append_result_atomic(path, result())

    serialized = path.read_text(encoding="utf-8")
    assert "PRIVATE_" not in serialized
    assert "prompt" not in serialized
    assert "raw_output" not in serialized
    assert load_completed_keys(path) == {("model-a", "EN-safe-1", "number")}


def test_completed_key_loader_rejects_duplicate_records(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    append_result_atomic(path, result())
    append_result_atomic(path, result())

    with pytest.raises(ValueError, match="duplicate result key"):
        load_completed_keys(path)


def test_coverage_verifier_requires_exact_keys(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    append_result_atomic(path, result())
    expected = {("model-a", "EN-safe-1", "number")}

    summary = verify_result_coverage(path, expected)

    assert summary == {"expected": 1, "actual": 1, "missing": 0, "extra": 0}
    with pytest.raises(ValueError, match="coverage mismatch"):
        verify_result_coverage(
            path,
            expected | {("model-a", "RU-safe-1", "number")},
        )
