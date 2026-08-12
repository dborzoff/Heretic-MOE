from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from heretic.language_map_cache import _capture_fingerprint
from heretic.language_map_data import LanguageFile, load_aligned_corpus
from heretic.language_map_worker import run_worker_job
from heretic.range_work_queue import RangeWorkQueue


class FakeModel:
    def __init__(self):
        self.prepared = []
        self.pinned = False

    def prepare_prompt_cache(self, prompts):
        self.prepared.extend(prompts)

    def pin_prompt_cache(self):
        self.pinned = True

    def iter_residual_batches(self, prompts, batch_size):
        for start in range(0, len(prompts), batch_size):
            count = min(batch_size, len(prompts) - start)
            yield torch.ones((count, 2, 3), dtype=torch.float32)


def _write_cell(path: Path, direction: str) -> None:
    prefix = "A" if direction == "safe" else "B"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "canonical_id": f"{prefix}{number:04d}",
                    "row_id": f"EN-{prefix}{number:04d}",
                    "language": "en",
                    "direction_class": direction,
                    "category_id": "C01",
                    "prompt": f"fixture-{prefix}-{number}",
                }
            )
            + "\n"
            for number in range(1, 3)
        ),
        encoding="utf-8",
    )


def _job(tmp_path: Path) -> tuple[Path, RangeWorkQueue]:
    safe = tmp_path / "safe.jsonl"
    unsafe = tmp_path / "unsafe.jsonl"
    _write_cell(safe, "safe")
    _write_cell(unsafe, "unsafe")
    specifications = [
        LanguageFile("en", "safe", safe),
        LanguageFile("en", "unsafe", unsafe),
    ]
    rows = load_aligned_corpus(specifications, ("en",), 2)
    metadata = {"mode": "test"}
    fingerprint = _capture_fingerprint(rows, 2, "system", metadata)
    queue = RangeWorkQueue(tmp_path / "queue.sqlite3")
    queue.initialize(row_count=4, rows_per_task=2, fingerprint=fingerprint)
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "languages": ["en"],
        "rows_per_cell": 2,
        "limit_per_cell": None,
        "files": [
            {"language": "en", "direction": "safe", "path": str(safe)},
            {"language": "en", "direction": "unsafe", "path": str(unsafe)},
        ],
        "model": "fake",
        "dtype": "bfloat16",
        "batch_size": 2,
        "system_prompt": "system",
        "metadata": metadata,
        "queue_path": str(queue.path),
        "parts_dir": str(tmp_path / "parts"),
        "cpu_threads": 1,
    }
    job = tmp_path / "job.json"
    job.write_text(json.dumps(payload), encoding="utf-8")
    return job, queue


def test_worker_validates_job_and_completes_real_range_queue(tmp_path: Path) -> None:
    job, queue = _job(tmp_path)
    model = FakeModel()

    result = run_worker_job(
        job,
        device="0",
        worker_id="gpu-0",
        model_factory=lambda _job: model,
    )

    assert result == {"worker_id": "gpu-0", "tasks": 2, "rows": 4}
    assert queue.stats().complete_rows == 4
    assert len(model.prepared) == 4
    assert model.pinned is True


def test_worker_rejects_fingerprint_drift_before_loading_model(tmp_path: Path) -> None:
    job, _ = _job(tmp_path)
    payload = json.loads(job.read_text(encoding="utf-8"))
    payload["fingerprint"] = "f" * 64
    job.write_text(json.dumps(payload), encoding="utf-8")
    loaded = False

    def model_factory(_job):
        nonlocal loaded
        loaded = True
        return FakeModel()

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_worker_job(job, device="0", worker_id="gpu-0", model_factory=model_factory)

    assert loaded is False
