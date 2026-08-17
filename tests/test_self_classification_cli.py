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
    languages = ("en", "ru", "zh", "ja", "fr")
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
        "rows": 10,
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
        "rows": 10,
        "variants": 3,
        "tasks": 30,
        "workers": 2,
        "languages": ["en", "ru", "zh", "ja", "fr"],
    }
    assert not (tmp_path / "output").exists()


def test_consensus_dry_run_uses_eight_variants_and_excludes_base_model(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(tmp_path)
    kept = tmp_path / "chat-model"
    excluded = tmp_path / "Qwen__Qwen3-0.6B-Base"

    result = main(
        [
            "consensus",
            "--dataset-manifest",
            str(manifest),
            "--model-root",
            str(tmp_path),
            "--model",
            str(kept),
            "--model",
            str(excluded),
            "--exclude-model-id",
            excluded.name,
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            "0,1",
            "--dry-run",
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "consensus-dry-run",
        "rows": 10,
        "models": 1,
        "variants": 8,
        "tasks": 80,
        "workers": 2,
        "system_mode": "english",
        "max_new_tokens": 8,
        "languages": ["en", "ru", "zh", "ja", "fr"],
    }
    assert not (tmp_path / "output").exists()


def test_consensus_accepts_an_explicit_language_subset(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)
    model = tmp_path / "chat-model"

    result = main(
        [
            "consensus",
            "--dataset-manifest",
            str(manifest),
            "--model-root",
            str(tmp_path),
            "--model",
            str(model),
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            "0,1",
            "--languages",
            "en,ru,zh,ja",
            "--dry-run",
        ]
    )

    assert result["rows"] == 8
    assert result["tasks"] == 64
    assert result["languages"] == ["en", "ru", "zh", "ja"]


def test_vote4_dry_run_uses_four_short_code_variants(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path)

    result = main(
        [
            "vote4",
            "--dataset-manifest",
            str(manifest),
            "--model",
            str(tmp_path / "chat-model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            "0,1",
            "--languages",
            "en,ru,zh,ja",
            "--dry-run",
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "vote4-dry-run",
        "rows": 8,
        "variants": 4,
        "tasks": 32,
        "workers": 2,
        "system_mode": "english",
        "max_new_tokens": 2,
        "languages": ["en", "ru", "zh", "ja"],
    }


def test_vote4_shards_one_model_across_all_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heretic.self_classification_data import load_classification_rows

    manifest = build_manifest(tmp_path)
    observed_jobs: list[dict[str, object]] = []

    def run_jobs(jobs, devices):
        assert tuple(devices) == ("0", "1")
        statuses = []
        for job_path, worker_id in jobs:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            observed_jobs.append(job)
            rows = load_classification_rows(job["dataset_manifest"], job["languages"])
            rows = rows[int(job["shard_index"]) :: int(job["shard_count"])]
            output = Path(str(job["output_path"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            payloads = []
            for row in rows:
                for variant in job["variants"]:
                    payloads.append(
                        {
                            "model_id": job["model_id"],
                            "canonical_id": row.canonical_id,
                            "row_id": row.row_id,
                            "language": row.language,
                            "category_ids": list(row.category_ids),
                            "direction_class": row.direction_class,
                            "variant": variant,
                            "classification": (
                                "DIRECT"
                                if row.direction_class == "safe"
                                else "HARD_REFUSE"
                            ),
                            "valid": True,
                            "output_tokens": 1,
                            "output_shape": "exact",
                        }
                    )
            output.write_text(
                "".join(json.dumps(value, sort_keys=True) + "\n" for value in payloads),
                encoding="utf-8",
            )
            statuses.append(
                {
                    "job_path": str(job_path),
                    "worker_id": worker_id,
                    "device": devices[int(job["shard_index"])],
                    "return_code": 0,
                }
            )
        return statuses

    monkeypatch.setattr("heretic.self_classification_cli._run_jobs", run_jobs)
    result = main(
        [
            "vote4",
            "--dataset-manifest",
            str(manifest),
            "--model",
            str(tmp_path / "chat-model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--devices",
            "0,1",
            "--languages",
            "en,ru,zh,ja",
        ]
    )

    assert result["status"] == "PASS"
    assert result["tasks"] == 32
    assert result["valid"] == 32
    assert result["invalid"] == 0
    assert result["ready_for_search"] is True
    assert (tmp_path / "output" / "vote4" / "vote4_summary.json").is_file()
    assert (tmp_path / "output" / "vote4" / "vote4_report.html").is_file()
    assert [job["shard_index"] for job in observed_jobs] == [0, 1]
    assert {job["shard_count"] for job in observed_jobs} == {2}
    assert {job["system_mode"] for job in observed_jobs} == {"english"}
    assert {job["max_new_tokens"] for job in observed_jobs} == {2}
    assert {tuple(job["variants"]) for job in observed_jobs} == {
        ("code_permuted", "code_shift_1", "code_shift_2", "code_shift_3")
    }


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
