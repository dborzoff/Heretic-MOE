from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from heretic.clean_reference_archive import build_clean_reference_archive
from heretic.language_map_data import GeometryRow


def _rows(tmp_path: Path) -> list[GeometryRow]:
    values = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for index, language in enumerate(("en", "ru"), 1):
            values.append(
                GeometryRow(
                    canonical_id=f"{prefix}{index:04d}",
                    row_id=f"{language.upper()}-{prefix}{index:04d}",
                    language=language,
                    direction=direction,
                    category_id="C01",
                    prompt=f"private-{direction}-{language}",
                    source_path=tmp_path / f"{language}.jsonl",
                    source_line=index,
                )
            )
    return values


class _FakeModel:
    def __init__(self) -> None:
        self.generation_calls = 0
        self.nll_calls = 0

    def get_response_artifacts_with_prefill_residuals(self, prompts, **kwargs):
        self.generation_calls += 1
        responses = [f"private-answer-{prompt.user}" for prompt in prompts]
        token_ids = [[index + 2, index + 3] for index in range(len(prompts))]
        residuals = torch.stack(
            [
                torch.tensor(
                    [
                        [float(index + 1), 0.0],
                        [0.0, float(index + 1)],
                    ]
                )
                for index in range(len(prompts))
            ]
        )
        return responses, token_ids, residuals

    def get_conditional_nll(self, prompts, target_token_ids):
        self.nll_calls += 1
        return [0.1 + 0.01 * index for index in range(len(prompts))]


def test_archive_is_private_resumable_hashed_and_public_manifest_is_text_free(
    tmp_path: Path,
) -> None:
    model = _FakeModel()
    output = tmp_path / "archive"
    direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    manifest = build_clean_reference_archive(
        model=model,
        rows=_rows(tmp_path),
        refusal_direction=direction,
        output_dir=output,
        dataset_contract_sha256="a" * 64,
        direction_sha256="b" * 64,
        model_fingerprint="fake-model-v1",
        max_response_length=512,
        batch_size=2,
    )

    assert manifest["status"] == "PASS"
    assert manifest["rows"] == 4
    assert manifest["safe_rows_with_nll"] == 2
    assert manifest["layers"] == 2
    assert model.generation_calls == 2
    assert model.nll_calls == 1
    serialized_manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert not any(
        f'"{field}"' in serialized_manifest
        for field in ("prompt", "response", "answer", "text")
    )
    records = [
        json.loads(line)
        for line in (output / "private" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["row_id"] for row in records] == [row.row_id for row in _rows(tmp_path)]
    assert records[0]["clean_response"].startswith("private-answer-")
    assert records[0]["clean_conditional_nll"] == pytest.approx(0.1)
    assert records[2]["clean_conditional_nll"] is None
    assert records[0]["clean_prompt_residual_projection"] == [1.0, 1.0]

    resumed = build_clean_reference_archive(
        model=model,
        rows=_rows(tmp_path),
        refusal_direction=direction,
        output_dir=output,
        dataset_contract_sha256="a" * 64,
        direction_sha256="b" * 64,
        model_fingerprint="fake-model-v1",
        max_response_length=512,
        batch_size=2,
    )

    assert resumed == manifest
    assert model.generation_calls == 2
    assert model.nll_calls == 1


def test_archive_rejects_direction_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="direction"):
        build_clean_reference_archive(
            model=_FakeModel(),
            rows=_rows(tmp_path),
            refusal_direction=torch.ones((3, 2)),
            output_dir=tmp_path / "archive",
            dataset_contract_sha256="a" * 64,
            direction_sha256="b" * 64,
            model_fingerprint="fake",
            max_response_length=512,
            batch_size=2,
        )
