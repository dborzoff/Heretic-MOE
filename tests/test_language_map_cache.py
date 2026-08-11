import hashlib
from pathlib import Path

import pytest
import torch

from heretic.language_map_cache import (
    capture_residual_cache,
    load_residual_cache,
)
from heretic.language_map_data import GeometryRow
from heretic.model import Model
from heretic.utils import Prompt


def sample_rows(tmp_path: Path, count: int = 4) -> list[GeometryRow]:
    rows = []
    for index in range(count):
        direction = "safe" if index < count // 2 else "unsafe"
        canonical_prefix = "S" if direction == "safe" else "U"
        canonical_id = f"{canonical_prefix}{index % (count // 2) + 1:04d}"
        rows.append(
            GeometryRow(
                canonical_id=canonical_id,
                row_id=f"EN-{canonical_id}",
                language="en",
                direction=direction,
                category_id="C01",
                prompt=f"private prompt {index}",
                source_path=tmp_path / f"{direction}.jsonl",
                source_line=index + 1,
            )
        )
    return rows


class FakeResidualModel:
    def __init__(self):
        self.seen_users: list[str] = []

    def iter_residual_batches(self, prompts, batch_size):
        offset = 0
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            self.seen_users.extend(prompt.user for prompt in batch)
            values = torch.arange(
                offset,
                offset + len(batch) * 2 * 3,
                dtype=torch.float32,
            ).reshape(len(batch), 2, 3)
            offset += len(batch) * 2 * 3
            yield values


class FailingResidualModel(FakeResidualModel):
    def iter_residual_batches(self, prompts, batch_size):
        yield torch.zeros((batch_size, 2, 3), dtype=torch.float32)
        raise RuntimeError("synthetic interruption")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_capture_measures_each_row_once_and_round_trips(tmp_path: Path):
    rows = sample_rows(tmp_path)
    model = FakeResidualModel()

    manifest = capture_residual_cache(
        model,
        rows,
        batch_size=2,
        output_dir=tmp_path / "cache",
        system_prompt="system",
        metadata={"model": "fake/model", "dtype": "bfloat16"},
    )
    index, residuals, loaded_manifest = load_residual_cache(tmp_path / "cache")

    assert model.seen_users == [row.prompt for row in rows]
    assert manifest["rows"] == len(rows) == residuals.shape[0]
    assert residuals.shape == (4, 2, 3)
    assert loaded_manifest["measurement_position"] == "first_generated_token"
    assert all("prompt" not in row for row in index)
    assert loaded_manifest["files"]["residuals.safetensors"]["sha256"] == sha256(
        tmp_path / "cache" / "residuals.safetensors"
    )


def test_interruption_never_publishes_final_cache(tmp_path: Path):
    output_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        capture_residual_cache(
            FailingResidualModel(),
            sample_rows(tmp_path),
            batch_size=2,
            output_dir=output_dir,
            system_prompt="system",
        )

    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "residuals.safetensors").exists()
    assert not (output_dir / "row_index.jsonl").exists()


def test_loader_detects_tampered_tensor(tmp_path: Path):
    output_dir = tmp_path / "cache"
    capture_residual_cache(
        FakeResidualModel(),
        sample_rows(tmp_path),
        batch_size=2,
        output_dir=output_dir,
        system_prompt="system",
    )
    tensor_path = output_dir / "residuals.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_residual_cache(output_dir)


def test_model_residual_iterator_preserves_batch_order():
    model = object.__new__(Model)
    seen: list[list[str]] = []

    def fake_get_residuals(batch):
        seen.append([prompt.user for prompt in batch])
        return torch.full((len(batch), 1, 1), float(len(seen)))

    model.get_residuals = fake_get_residuals
    prompts = [Prompt(system="s", user=f"p{index}") for index in range(5)]

    batches = list(model.iter_residual_batches(prompts, batch_size=2))

    assert seen == [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert [batch.shape[0] for batch in batches] == [2, 2, 1]
