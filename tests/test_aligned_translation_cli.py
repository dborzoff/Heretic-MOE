from __future__ import annotations

import json
from pathlib import Path

from heretic.aligned_translation import TranslationInput
from heretic.aligned_translation_cli import (
    build_worker_jobs,
    merge_translation_sidecars,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_worker_jobs_cover_each_shard_without_embedding_text(tmp_path: Path) -> None:
    jobs = build_worker_jobs(
        dataset_manifest=tmp_path / "manifest.json",
        model=tmp_path / "model",
        output_root=tmp_path / "run",
        devices=("0", "1"),
        batch_size=16,
        max_new_tokens=256,
    )

    assert [job["shard_index"] for job in jobs] == [0, 1]
    assert all(job["shard_count"] == 2 for job in jobs)
    assert all("prompt" not in json.dumps(job) for job in jobs)
    assert jobs[0]["output_path"].endswith("translations.worker-0.jsonl")


def test_merge_sidecars_restores_canonical_order(tmp_path: Path) -> None:
    rows = [
        TranslationInput("S0001", "safe", "private-one"),
        TranslationInput("U0001", "unsafe", "private-two"),
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(
        first,
        [{"canonical_id": "U0001", "direction_class": "unsafe", "prompt": "ja-two"}],
    )
    _write(
        second,
        [{"canonical_id": "S0001", "direction_class": "safe", "prompt": "ja-one"}],
    )
    output = tmp_path / "merged.jsonl"

    summary = merge_translation_sidecars((first, second), rows, output)

    assert summary == {"rows": 2, "status": "PASS"}
    assert [json.loads(line)["canonical_id"] for line in output.read_text().splitlines()] == [
        "S0001",
        "U0001",
    ]
