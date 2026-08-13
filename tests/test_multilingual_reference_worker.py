from __future__ import annotations

import json
from pathlib import Path

import torch

from heretic.language_map_data import GeometryRow
from heretic.multilingual_reference_worker import run_worker_job


def test_reference_worker_writes_only_its_contiguous_shard(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
model = "fake"
batch_size = 2
[multilingual_search]
enabled = true
dataset_root = "dataset"
languages = ["en", "ru", "zh", "es", "fr"]
direction_rows_per_cell = 1000
trial_rows_per_cell = 400
final_holdout_rows_per_language = 132
""".strip(),
        encoding="utf-8",
    )
    rows = tuple(
        GeometryRow(
            canonical_id=f"S{index}",
            row_id=f"EN-S{index}",
            language="en",
            direction="safe",
            category_id="C01",
            prompt=f"private-{index}",
            source_path=tmp_path / "fixture.jsonl",
            source_line=index,
        )
        for index in range(4)
    )
    bundle = type(
        "Bundle",
        (),
        {"trial_rows": rows, "manifest": {"contract_sha256": "a" * 64}},
    )()
    profile = type(
        "Profile",
        (),
        {"consensus_refusal_direction": torch.ones((2, 2))},
    )()
    monkeypatch.setattr(
        "heretic.multilingual_contract.load_multilingual_dataset_bundle",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        "heretic.language_map_directions.load_direction_map_package",
        lambda path: (profile, {"package_sha256": "b" * 64}),
    )

    class FakeModel:
        def __init__(self):
            self.prepared = []
            self.pinned = False

        def prepare_prompt_cache(self, prompts):
            self.prepared.extend(prompts)

        def pin_prompt_cache(self):
            self.pinned = True

        def get_response_artifacts_with_prefill_residuals(self, prompts, **kwargs):
            count = len(prompts)
            return ["answer"] * count, [[1]] * count, torch.ones((count, 2, 2))

        def get_conditional_nll(self, prompts, targets):
            return [0.1] * len(prompts)

    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_path": str(config),
                "runtime_root": str(tmp_path / "runtime"),
                "shards_root": str(tmp_path / "shards"),
                "model": "fake",
                "model_fingerprint": "fake-model",
                "batch_size": 2,
                "max_response_length": 100,
            }
        ),
        encoding="utf-8",
    )

    model = FakeModel()
    result = run_worker_job(
        job,
        device="0",
        worker_id="gpu-0",
        start=1,
        end=3,
        model_factory=lambda settings: model,
    )

    assert result["rows"] == 2
    assert len(model.prepared) == 2
    assert model.pinned is True
    records = (
        tmp_path / "shards" / "00000001-00000003" / "private" / "records.jsonl"
    ).read_text(encoding="utf-8")
    assert '"row_id":"EN-S1"' in records
    assert '"row_id":"EN-S2"' in records
    assert '"row_id":"EN-S0"' not in records
