from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
from types import MethodType, SimpleNamespace

import numpy as np

import pytest

from heretic.scorer import Score
from heretic.scorers.sparse_refusal_geometry import SparseRefusalGeometry
from heretic.srg_benchmark import build_jobs, result_payload, summarize_results
from heretic.utils import Prompt


def test_build_jobs_preserves_model_order_and_uses_stable_ids(tmp_path: Path) -> None:
    models = [tmp_path / "Alpha Model", tmp_path / "Beta-Model"]
    for model in models:
        model.mkdir()

    jobs = build_jobs(models, tmp_path / "results")

    assert [job["model_index"] for job in jobs] == [0, 1]
    assert [job["model_id"] for job in jobs] == ["alpha-model", "beta-model"]
    assert [Path(job["model_path"]) for job in jobs] == models
    assert [Path(job["result_path"]).name for job in jobs] == [
        "000-alpha-model.json",
        "001-beta-model.json",
    ]


def test_build_jobs_rejects_duplicate_models(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="duplicate model path"):
        build_jobs([model, model], tmp_path / "results")


def test_summarize_results_reports_cross_model_srg_without_text_fields() -> None:
    results = [
        {
            "model_index": 1,
            "model_id": "beta",
            "model_path": "B:/beta",
            "rows": 132,
            "mean_margin": -0.01,
            "positive_rate": 0.25,
            "empty_count": 2,
            "prototype_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "status": "PASS",
        },
        {
            "model_index": 0,
            "model_id": "alpha",
            "model_path": "A:/alpha",
            "rows": 132,
            "mean_margin": 0.03,
            "positive_rate": 0.75,
            "empty_count": 0,
            "prototype_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
            "status": "PASS",
        },
    ]

    report = summarize_results(results)

    assert report["status"] == "PASS"
    assert report["models"] == 2
    assert report["rows_per_model"] == 132
    assert report["mean_margin_range"] == [-0.01, 0.03]
    assert report["positive_rate_range"] == [0.25, 0.75]
    assert [row["model_id"] for row in report["results"]] == ["alpha", "beta"]
    assert not ({"prompt", "response", "answer", "text"} & set(report))


def test_summarize_results_rejects_mixed_frozen_inputs() -> None:
    base = {
        "model_index": 0,
        "model_id": "alpha",
        "model_path": "A:/alpha",
        "rows": 132,
        "mean_margin": 0.0,
        "positive_rate": 0.5,
        "empty_count": 0,
        "prototype_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "status": "PASS",
    }
    changed = {**base, "model_index": 1, "model_id": "beta", "prompt_sha256": "c" * 64}

    with pytest.raises(ValueError, match="prompt SHA-256"):
        summarize_results([base, changed])


def test_cli_dry_run_writes_a_text_free_pinned_manifest(tmp_path: Path) -> None:
    models = [tmp_path / "model-a", tmp_path / "model-b"]
    for model in models:
        model.mkdir()
    prototypes = tmp_path / "prototypes.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    prototypes.write_bytes(b"prototype-bank\n")
    prompts.write_bytes(b"prompt-set\n")
    output = tmp_path / "output"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from heretic.cli import main; main()",
            "srg-benchmark",
            "--models",
            *(str(model) for model in models),
            "--output",
            str(output),
            "--prototypes",
            str(prototypes),
            "--prompts",
            str(prompts),
            "--devices",
            "0,1",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY"
    assert manifest["devices"] == ["0", "1"]
    assert manifest["model_count"] == 2
    assert len(manifest["prototype_sha256"]) == 64
    assert len(manifest["prompt_sha256"]) == 64
    assert not ({"prompt", "response", "answer", "text"} & set(manifest))


def test_result_payload_keeps_numeric_diagnostics_and_rejects_text() -> None:
    job = {
        "model_index": 0,
        "model_id": "alpha",
        "model_path": "A:/alpha",
    }
    score = Score(
        value=-0.02,
        rich_display="unused",
        md_display="unused",
        diagnostics={
            "rows": 132,
            "mean_margin": -0.02,
            "positive_rate": 0.25,
            "empty_indices": [3],
            "margins": [-0.1, 0.2],
        },
    )

    result = result_payload(
        job,
        score,
        prototype_sha256="a" * 64,
        prompt_sha256="b" * 64,
        elapsed_seconds=12.5,
    )

    assert result["status"] == "PASS"
    assert result["rows"] == 132
    assert result["mean_margin"] == -0.02
    assert result["positive_rate"] == 0.25
    assert result["empty_count"] == 1
    assert result["diagnostics"]["margins"] == [-0.1, 0.2]
    assert "rich_display" not in result

    score.diagnostics["prompt"] = "must never be persisted"
    with pytest.raises(ValueError, match="sensitive field"):
        result_payload(
            job,
            score,
            prototype_sha256="a" * 64,
            prompt_sha256="b" * 64,
            elapsed_seconds=12.5,
        )


def test_cli_rejects_zero_batch_instead_of_reaching_model_batchify(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    prototypes = tmp_path / "prototypes.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    prototypes.write_bytes(b"prototype-bank\n")
    prompts.write_bytes(b"prompt-set\n")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "heretic.srg_benchmark",
            "--models",
            str(model),
            "--output",
            str(tmp_path / "out"),
            "--prototypes",
            str(prototypes),
            "--prompts",
            str(prompts),
            "--batch-size",
            "0",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "batch size must be positive" in completed.stderr


def test_sparse_geometry_scores_pre_generated_batches_without_regenerating() -> None:
    scorer = object.__new__(SparseRefusalGeometry)
    scorer.settings = SimpleNamespace(empty_response_margin=1.0)

    def class_scores(_self, prompts, responses):
        assert prompts == ["safe-id-1", "safe-id-2"]
        assert responses == ["generated", ""]
        return {
            "delivered": np.asarray([0.6, 0.6]),
            "soft": np.asarray([0.2, 0.2]),
            "refuse": np.asarray([0.1, 0.1]),
        }

    scorer._class_scores = MethodType(class_scores, scorer)

    score = scorer.score_responses(
        [Prompt(system="", user="safe-id-1"), Prompt(system="", user="safe-id-2")],
        ["generated", ""],
    )

    assert score.value == pytest.approx(0.3)
    assert score.diagnostics["positive_count"] == 1
    assert score.diagnostics["empty_indices"] == [1]
