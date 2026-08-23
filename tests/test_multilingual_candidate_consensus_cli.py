from __future__ import annotations

import json

from heretic.multilingual_candidate_consensus_cli import main


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_aggregate_cli_verifies_run_plan_and_keeps_metadata_text_private(tmp_path) -> None:
    """Catches trusting incomplete model files or copying prompts into public output."""
    models = [f"model-{index}" for index in range(10)]
    languages = ["en", "ru", "zh", "ja"]
    variants = [f"variant-{index}" for index in range(8)]
    votes_root = tmp_path / "votes"
    model_entries = []
    for model_id in models:
        rows = []
        for language in languages:
            labels = ["SOFT"] * 8
            if model_id == "model-0" and language == "ru":
                labels = ["SOFT"] * 5 + ["HARD_REFUSE"] * 2 + [None]
            for variant, label in zip(variants, labels, strict=True):
                rows.append(
                    {
                        "canonical_id": "U0001",
                        "row_id": f"{language}-U0001",
                        "model_id": model_id,
                        "language": language,
                        "variant": variant,
                        "classification": label,
                        "valid": label is not None,
                        "category_ids": ["C01"],
                    }
                )
        output = votes_root / model_id / "rows.jsonl"
        write_jsonl(output, rows)
        model_entries.append(
            {"model_id": model_id, "output": str(output), "expected_rows": 32}
        )
    run_plan = tmp_path / "run_plan.json"
    run_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "models": model_entries,
                "languages": languages,
                "variants": variants,
                "rows_per_model": 32,
            }
        ),
        encoding="utf-8",
    )
    targets = tmp_path / "target_labels.jsonl"
    write_jsonl(
        targets,
        [
            {
                "canonical_id": "U0001",
                "target_behavior_class": "SOFT",
                "category_ids": ["C01"],
                "candidate_group": "strict_soft",
            }
        ],
    )
    metadata = tmp_path / "direction_en_unsafe.jsonl"
    write_jsonl(
        metadata,
        [{"canonical_id": "U0001", "source": "fixture", "prompt": "PRIVATE"}],
    )
    output_dir = tmp_path / "output"
    retry_results = tmp_path / "retry_results.jsonl"
    write_jsonl(
        retry_results,
        [
            {
                "canonical_id": "U0001",
                "row_id": "ru-U0001",
                "model_id": "model-0",
                "language": "ru",
                "classification": "SOFT",
                "valid": True,
                "replaces_variant": "variant-7",
            }
        ],
    )

    result = main(
        [
            "aggregate",
            "--run-plan",
            str(run_plan),
            "--target-labels",
            str(targets),
            "--metadata-en",
            str(metadata),
            "--retry-results",
            str(retry_results),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result["status"] == "PASS"
    assert result["counts"]["clean_soft"] == 1
    assert set(result["inputs"]["model_result_sha256"]) == set(models)
    assert len(result["inputs"]["run_plan_sha256"]) == 64
    assert len(result["inputs"]["retry_results_sha256"]) == 64
    public = (output_dir / "candidate_consensus.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE" not in public
    assert '"source": "fixture"' in public
    cells = [
        json.loads(line)
        for line in (output_dir / "model_language_cells.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    retried = next(
        row
        for row in cells
        if row["model_id"] == "model-0" and row["language"] == "ru"
    )
    assert retried["stable_support"] == 6
    assert retried["invalid_votes"] == 0


def test_plan_retries_writes_only_unstable_invalid_model_language_cells(tmp_path) -> None:
    """Catches sending stable cells or prompt text into the invalid retry queue."""
    models = [f"model-{index}" for index in range(10)]
    languages = ["en", "ru", "zh", "ja"]
    variants = [f"variant-{index}" for index in range(8)]
    model_entries = []
    for model_id in models:
        rows = []
        for language in languages:
            labels = ["SOFT"] * 8
            if model_id == "model-0" and language == "ru":
                labels = ["HARD_REFUSE"] * 5 + ["SOFT"] * 2 + [None]
            for variant, label in zip(variants, labels, strict=True):
                rows.append(
                    {
                        "canonical_id": "U0001",
                        "row_id": f"{language}-U0001",
                        "model_id": model_id,
                        "language": language,
                        "variant": variant,
                        "classification": label,
                        "valid": label is not None,
                        "category_ids": ["C01"],
                    }
                )
        output = tmp_path / "votes" / model_id / "rows.jsonl"
        write_jsonl(output, rows)
        model_entries.append(
            {"model_id": model_id, "output": str(output), "expected_rows": 32}
        )
    run_plan = tmp_path / "run_plan.json"
    run_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "models": model_entries,
                "languages": languages,
                "variants": variants,
                "rows_per_model": 32,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "retry"

    result = main(
        [
            "plan-retries",
            "--run-plan",
            str(run_plan),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result["status"] == "PASS"
    assert result["retry_cells"] == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "retry_plan.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert rows == [
        {
            "canonical_id": "U0001",
            "language": "ru",
            "model_id": "model-0",
            "replaces_variant": "variant-7",
            "row_id": "ru-U0001",
        }
    ]
