from __future__ import annotations

import hashlib
import json

from heretic.multilingual_invalid_retry import (
    collect_invalid_retry_results,
    materialize_invalid_retry_jobs,
)
from heretic.self_classification_data import load_classification_rows


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_retry_jobs_uses_aligned_union_without_printing_prompts(
    tmp_path,
) -> None:
    """Catches reloading a model per language or losing aligned retry coverage."""
    languages = ["en", "ru", "zh", "ja"]
    aligned = tmp_path / "aligned"
    files = []
    for language in languages:
        for direction in ("safe", "unsafe"):
            path = aligned / f"direction_{language}_{direction}.jsonl"
            rows = []
            if direction == "unsafe":
                rows = [
                    {
                        "canonical_id": "U0001",
                        "row_id": f"{language}-U0001",
                        "language": language,
                        "direction_class": "unsafe",
                        "category_ids": ["C01"],
                        "prompt": f"PRIVATE-{language}",
                    }
                ]
            write_jsonl(path, rows)
            files.append(
                {
                    "language": language,
                    "direction": direction,
                    "path": path.name,
                    "rows": len(rows),
                    "sha256": sha256(path),
                }
            )
    aligned_manifest = aligned / "manifest.json"
    aligned_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": languages,
                "directions": {"safe": 0, "unsafe": 1},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    original_job = tmp_path / "original_job.json"
    original_job.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "F:/models/model-0",
                "model_id": "model-0",
                "batch_size": 16,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    run_plan = tmp_path / "run_plan.json"
    run_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": languages,
                "models": [{"model_id": "model-0", "job": str(original_job)}],
            }
        ),
        encoding="utf-8",
    )
    retry_plan = tmp_path / "retry_plan.jsonl"
    write_jsonl(
        retry_plan,
        [
            {
                "canonical_id": "U0001",
                "row_id": "ru-U0001",
                "model_id": "model-0",
                "language": "ru",
                "replaces_variant": "word_order_3",
            }
        ],
    )

    manifest = materialize_invalid_retry_jobs(
        run_plan,
        retry_plan,
        aligned_manifest,
        tmp_path / "retry_jobs",
    )

    assert manifest["status"] == "PASS"
    assert manifest["jobs"] == 1
    assert manifest["retry_cells"] == 1
    assert manifest["aligned_union_rows"] == 4
    job_path = tmp_path / "retry_jobs" / manifest["job_files"][0]
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["variants"] == ["number"]
    assert job["languages"] == languages
    retry_rows = load_classification_rows(job["dataset_manifest"], languages)
    assert len(retry_rows) == 4
    assert {row.canonical_id for row in retry_rows} == {"U0001"}

    write_jsonl(
        type(retry_plan)(job["output_path"]),
        [
            {
                "canonical_id": "U0001",
                "row_id": f"{language}-U0001",
                "model_id": "model-0",
                "language": language,
                "variant": "number",
                "classification": "SOFT",
                "valid": True,
                "output_tokens": 1,
                "output_shape": "number",
            }
            for language in languages
        ],
    )
    collected = collect_invalid_retry_results(
        tmp_path / "retry_jobs" / "manifest.json",
        retry_plan,
        tmp_path / "retry_results.jsonl",
    )
    assert collected["status"] == "PASS"
    assert collected["retry_cells"] == 1
    assert collected["valid"] == 1
    public = (tmp_path / "retry_results.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE" not in public
    assert '"replaces_variant": "word_order_3"' in public
