from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from heretic.multilingual_contract import CalibrationRow
from heretic.multilingual_final_holdout import (
    build_final_holdout_archive,
    evaluate_final_holdout,
    load_final_holdout_archive,
    merge_final_holdout_archives,
)


def _rows(tmp_path: Path) -> list[CalibrationRow]:
    rows = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for language in ("en", "ru"):
            for index in range(2):
                rows.append(
                    CalibrationRow(
                        base_id=f"{prefix}{index + 1:04d}",
                        row_id=f"{language.upper()}-{prefix}{index + 1:04d}",
                        language=language,
                        direction=direction,
                        category_id=f"C0{index + 1}",
                        prompt=f"private-{direction}-{language}-{index}",
                        source_path=tmp_path / "private.jsonl",
                        source_line=index + 1,
                    )
                )
    return rows


class _Model:
    def __init__(self, *, candidate: bool = False) -> None:
        self.candidate = candidate
        self.calls = 0

    def get_response_artifacts_with_prefill_residuals_batched(self, prompts, **kwargs):
        self.calls += 1
        prefix = "candidate" if self.candidate else "clean"
        residual = 0.5 if self.candidate else 1.0
        return (
            [f"{prefix}-{index}" for index in range(len(prompts))],
            [[index + 1, index + 2] for index in range(len(prompts))],
            torch.full((len(prompts), 2, 2), residual),
        )


class _Scorer:
    def score_responses(self, prompts, responses):
        clean = all(response.startswith("clean") for response in responses)
        margins = []
        for prompt in prompts:
            unsafe = "-unsafe-" in prompt.user
            if clean:
                margins.append(1.0 if unsafe else -1.0)
            else:
                margins.append(-1.0 if unsafe else 1.0)
        return SimpleNamespace(diagnostics={"margins": margins})


def _profile() -> dict[str, object]:
    return {
        "rows": 660,
        "scale": [1.0] * 660,
        "weight": [1.0] * 660,
        "global_scale": 1.0,
        "group_scale": {},
        "group_weight": {},
    }


def test_final_holdout_freezes_clean_reference_then_rechecks_candidate(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path)
    clean_model = _Model()
    archive = tmp_path / "r_holdout"

    manifest = build_final_holdout_archive(
        model=clean_model,
        rows=rows,
        refusal_direction=torch.ones((2, 2)),
        srg_scorer=_Scorer(),
        srg_profile=_profile(),
        output_dir=archive,
        dataset_contract_sha256="a" * 64,
        model_fingerprint="clean-model-v1",
        top_six_contract_sha256="b" * 64,
        max_response_length=1024,
    )
    loaded_manifest, clean_records = load_final_holdout_archive(archive)
    candidate_model = _Model(candidate=True)
    measurement = evaluate_final_holdout(
        trial_number=3,
        model=candidate_model,
        rows=rows,
        clean_records=clean_records,
        refusal_direction=torch.ones((2, 2)),
        layer_reliability=torch.ones(2),
        srg_scorer=_Scorer(),
        srg_profile=_profile(),
        private_records_path=tmp_path / "private" / "candidate-3.jsonl",
        max_response_length=1024,
    )

    assert clean_model.calls == 1
    assert candidate_model.calls == 1
    assert manifest == loaded_manifest
    assert manifest["rows"] == 8
    assert manifest["max_response_length"] == 1024
    assert manifest["top_six_contract_sha256"] == "b" * 64
    assert measurement["rows"] == 8
    assert measurement["direction_rows"] == {"safe": 4, "unsafe": 4}
    assert measurement["srg"]["rows"] == 4
    assert measurement["removal"] > 0.0
    assert measurement["srg_gain"] > 0.0
    assert measurement["r_gain"] > 0.0
    assert measurement["unsafe_geometry_gain"] > 0.0
    assert measurement["safe_geometry_damage"] > 0.0
    assert measurement["geometry"]["safe_rows"] == 4
    assert measurement["geometry"]["unsafe_rows"] == 4
    assert set(measurement["groups"]["languages"]) == {"en", "ru"}
    serialized = json.dumps(measurement, sort_keys=True)
    assert not any(key in serialized for key in ('"prompt"', '"response"', '"text"'))


def test_final_holdout_contract_rejects_generation_backend_drift(
    tmp_path: Path,
) -> None:
    common = {
        "model": _Model(),
        "rows": _rows(tmp_path),
        "refusal_direction": torch.ones((2, 2)),
        "srg_scorer": _Scorer(),
        "srg_profile": _profile(),
        "output_dir": tmp_path / "r_holdout",
        "dataset_contract_sha256": "a" * 64,
        "model_fingerprint": "clean-model-v1",
        "top_six_contract_sha256": "b" * 64,
        "max_response_length": 100,
    }
    build_final_holdout_archive(
        **common,
        generation_contract={
            "backend": "dynamic_eager",
            "prompt_bucket_multiple": 0,
            "compile_mode": "default",
        },
    )

    with pytest.raises(ValueError, match="contract differs"):
        build_final_holdout_archive(
            **common,
            generation_contract={
                "backend": "compiled_static",
                "prompt_bucket_multiple": 64,
                "compile_mode": "default",
            },
        )


def test_final_holdout_contract_rejects_direction_drift(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    common = {
        "model": _Model(),
        "refusal_direction": torch.ones((2, 2)),
        "srg_scorer": _Scorer(),
        "srg_profile": _profile(),
        "output_dir": tmp_path / "r_holdout",
        "dataset_contract_sha256": "a" * 64,
        "model_fingerprint": "clean-model-v1",
        "top_six_contract_sha256": "b" * 64,
        "max_response_length": 100,
    }
    build_final_holdout_archive(rows=rows, **common)
    changed = [replace(rows[0], direction="unsafe"), *rows[1:]]

    with pytest.raises(ValueError, match="contract differs"):
        build_final_holdout_archive(rows=changed, **common)


def test_final_holdout_rejects_wrong_clean_order(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    records = [
        {
            "row_id": row.row_id,
            "clean_response": "clean",
            "clean_margin": 1.0,
            "clean_prompt_residual_projection": [1.0, 1.0],
        }
        for row in reversed(rows)
    ]

    with pytest.raises(ValueError, match="order"):
        evaluate_final_holdout(
            trial_number=0,
            model=_Model(candidate=True),
            rows=rows,
            clean_records=records,
            refusal_direction=torch.ones((2, 2)),
            layer_reliability=torch.ones(2),
            srg_scorer=_Scorer(),
            srg_profile=_profile(),
            private_records_path=tmp_path / "candidate.jsonl",
            max_response_length=1024,
        )


def test_final_holdout_shards_merge_in_canonical_order(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    direction = torch.ones((2, 2))
    shard_dirs = []
    for index, shard_rows in enumerate((rows[:2], rows[2:])):
        shard = tmp_path / f"shard-{index}"
        build_final_holdout_archive(
            model=_Model(),
            rows=shard_rows,
            refusal_direction=direction,
            srg_scorer=_Scorer(),
            srg_profile=_profile(),
            output_dir=shard,
            dataset_contract_sha256="a" * 64,
            model_fingerprint="model-v1",
            top_six_contract_sha256="b" * 64,
            max_response_length=100,
        )
        shard_dirs.append(shard)

    output = tmp_path / "merged"
    manifest = merge_final_holdout_archives(
        shard_dirs=shard_dirs,
        rows=rows,
        refusal_direction=direction,
        srg_profile=_profile(),
        output_dir=output,
        dataset_contract_sha256="a" * 64,
        model_fingerprint="model-v1",
        top_six_contract_sha256="b" * 64,
        max_response_length=100,
    )
    _, records = load_final_holdout_archive(output)

    assert manifest["status"] == "PASS"
    assert manifest["shards"] == 2
    assert [record["row_id"] for record in records] == [row.row_id for row in rows]
