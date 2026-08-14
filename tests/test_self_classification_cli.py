from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from heretic import cli
from heretic.self_classification_cli import (
    discover_candidate_models,
    main,
    merge_result_files,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path) -> Path:
    files = []
    languages = ("en", "ru", "zh", "ko")
    for language in languages:
        for direction in ("safe", "unsafe"):
            path = root / f"{language}-{direction}.jsonl"
            value = {
                "canonical_id": f"{direction}-1",
                "row_id": f"{language.upper()}-{direction}-1",
                "language": language,
                "direction_class": direction,
                "category_ids": ["C1"],
                "prompt": "PRIVATE_SENTINEL",
            }
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
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
        "languages": list(languages),
        "rows": 8,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def public_result(row_id: str) -> dict[str, object]:
    return {
        "model_id": "model-a",
        "canonical_id": row_id.split("-", 1)[1],
        "row_id": row_id,
        "language": row_id[:2].lower(),
        "category_ids": ["C1"],
        "direction_class": "safe",
        "variant": "number",
        "classification": "DIRECT",
        "valid": True,
        "output_tokens": 1,
    }


def test_cli_dispatches_self_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(
        "heretic.self_classification_cli.main",
        lambda arguments: received.append(list(arguments)),
    )
    monkeypatch.setattr(sys, "argv", ["hereticMOE", "self-classify", "pilot"])

    cli.main()

    assert received == [["pilot"]]


def test_pilot_dry_run_reports_exact_task_count(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)

    result = main(
        [
            "pilot",
            "--dataset-manifest",
            str(manifest),
            "--model",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            "0,1",
            "--dry-run",
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "pilot-dry-run",
        "rows": 8,
        "variants": 3,
        "tasks": 24,
        "workers": 2,
    }
    assert not (tmp_path / "output").exists()


def test_model_discovery_filters_non_chat_and_oversized_weights(tmp_path: Path) -> None:
    def model(name: str, *, chat: bool, weight_bytes: int) -> None:
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"model_type": "qwen", "architectures": ["QwenForCausalLM"]})
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "template" if chat else None})
        )
        (root / "model.safetensors").write_bytes(b"x" * weight_bytes)

    model("good", chat=True, weight_bytes=50)
    model("base", chat=False, weight_bytes=50)
    model("huge", chat=True, weight_bytes=101)

    discovered = discover_candidate_models(
        tmp_path,
        max_weight_bytes=100,
        max_models=10,
    )

    assert [path.name for path in discovered] == ["good"]


def test_result_merge_rejects_duplicates_and_writes_canonical_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text(json.dumps(public_result("RU-P1")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(public_result("EN-P1")) + "\n", encoding="utf-8")

    rows = merge_result_files([first, second], output)

    assert [row["row_id"] for row in rows] == ["EN-P1", "RU-P1"]
    second.write_text(json.dumps(public_result("RU-P1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate result key"):
        merge_result_files([first, second], output)
