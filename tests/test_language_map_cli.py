from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
import pytest
import torch

from heretic import cli
from heretic import language_map_cli
from heretic.language_map_cache import capture_residual_cache
from heretic.language_map_data import LanguageFile, load_aligned_corpus
from heretic.supervisor import GpuInfo


def _write_cell(path: Path, *, language: str, direction: str) -> None:
    rows = []
    prefix = "A" if direction == "safe" else "B"
    for number in range(1, 3):
        canonical_id = f"{prefix}{number:04d}"
        rows.append(
            {
                "canonical_id": canonical_id,
                "row_id": f"{language.upper()}-{canonical_id}",
                "language": language,
                "direction_class": direction,
                "category_id": "C01",
                "prompt": f"synthetic-{language}-{canonical_id}",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_dispatches_geometry_map_without_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(language_map_cli, "main", lambda argv: received.append(argv))
    monkeypatch.setattr(cli.sys, "argv", ["hereticMOE", "geometry-map", "analyze"])

    cli.main()

    assert received == [["analyze"]]


def test_dry_run_validates_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[tuple[str, str], Path] = {}
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = tmp_path / f"{language}-{direction}.jsonl"
            _write_cell(path, language=language, direction=direction)
            files[(language, direction)] = path

    def fail_if_loaded(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not load a model")

    monkeypatch.setattr(language_map_cli, "_capture_parallel", fail_if_loaded)
    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--group-a",
            f"en={files[('en', 'safe')]}",
            "--group-a",
            f"ru={files[('ru', 'safe')]}",
            "--group-b",
            f"en={files[('en', 'unsafe')]}",
            "--group-b",
            f"ru={files[('ru', 'unsafe')]}",
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "dry-run",
        "languages": ["en", "ru"],
        "rows": 8,
        "rows_per_cell": 2,
        "source_rows_per_cell": 2,
    }


def test_complete_geometry_cache_rejects_changed_capture_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [SimpleNamespace(prompt="row")]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        model="model-b",
        batch_size=8,
        output_dir=tmp_path,
        system_prompt="system-b",
    )
    files: list[LanguageFile] = []
    monkeypatch.setattr(
        language_map_cli,
        "_capture_metadata",
        lambda _args, _files: {"model": "model-b"},
    )
    monkeypatch.setattr(
        language_map_cli,
        "_capture_fingerprint",
        lambda *_args: "b" * 64,
    )
    monkeypatch.setattr(
        language_map_cli,
        "load_residual_cache",
        lambda _path: ([], torch.zeros(1), {"capture_fingerprint": "a" * 64}),
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        language_map_cli._capture_parallel(rows, args, files)


def test_dry_run_can_limit_each_validated_cell(tmp_path: Path) -> None:
    files: dict[tuple[str, str], Path] = {}
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = tmp_path / f"{language}-{direction}.jsonl"
            _write_cell(path, language=language, direction=direction)
            files[(language, direction)] = path

    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--limit-per-cell",
            "1",
            "--group-a",
            f"en={files[('en', 'safe')]}",
            "--group-a",
            f"ru={files[('ru', 'safe')]}",
            "--group-b",
            f"en={files[('en', 'unsafe')]}",
            "--group-b",
            f"ru={files[('ru', 'unsafe')]}",
        ]
    )

    assert result["rows"] == 4
    assert result["rows_per_cell"] == 1
    assert result["source_rows_per_cell"] == 2


def test_dry_run_discovers_frozen_layout_from_corpus_root(tmp_path: Path) -> None:
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            _write_cell(
                tmp_path / f"direction_{language}_{direction}_train.jsonl",
                language=language,
                direction=direction,
            )

    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--corpus-root",
            str(tmp_path),
        ]
    )

    assert result["status"] == "PASS"
    assert result["rows"] == 8


def test_corpus_root_can_select_test_split(tmp_path: Path) -> None:
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            _write_cell(
                tmp_path / f"direction_{language}_{direction}_test.jsonl",
                language=language,
                direction=direction,
            )

    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--corpus-root",
            str(tmp_path),
            "--split",
            "test",
        ]
    )

    assert result["status"] == "PASS"
    assert result["rows"] == 8


def test_run_rejects_single_and_multi_device_flags_together(tmp_path: Path) -> None:
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            _write_cell(
                tmp_path / f"direction_{language}_{direction}_train.jsonl",
                language=language,
                direction=direction,
            )

    with pytest.raises(ValueError, match="cannot be combined"):
        language_map_cli.main(
            [
                "run",
                "--dry-run",
                "--languages",
                "en,ru",
                "--rows-per-cell",
                "2",
                "--corpus-root",
                str(tmp_path),
                "--device",
                "0",
                "--devices",
                "0,1",
            ]
        )


def test_device_selection_defaults_to_all_eligible_and_preserves_explicit_order() -> None:
    available = [
        GpuInfo("0", "fast", 24576, 22000, 5),
        GpuInfo("1", "slow", 24576, 18000, 10),
    ]
    common = {
        "device": None,
        "max_workers": None,
        "min_free_gib": 4.0,
        "min_free_fraction": 0.35,
    }

    automatic = language_map_cli._resolve_devices(
        SimpleNamespace(devices=None, **common), available
    )
    explicit = language_map_cli._resolve_devices(
        SimpleNamespace(devices="1,0", **common), available
    )

    assert [device.index for device in automatic] == ["0", "1"]
    assert [device.index for device in explicit] == ["1", "0"]


def test_project_builds_private_anchors_and_offline_report(tmp_path: Path) -> None:
    files: list[LanguageFile] = []
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = tmp_path / f"{language}-{direction}.jsonl"
            _write_cell(path, language=language, direction=direction)
            files.append(LanguageFile(language, direction, path))
    rows = load_aligned_corpus(files, ("en", "ru"), 2)

    class FakeModel:
        def iter_residual_batches(self, prompts, batch_size):
            values = torch.arange(len(prompts) * 2 * 4, dtype=torch.float32)
            yield values.reshape(len(prompts), 2, 4)

    cache = tmp_path / "cache"
    capture_residual_cache(
        FakeModel(), rows, 8, cache, system_prompt="Synthetic system"
    )
    journal = tmp_path / "study.log"
    storage = JournalStorage(
        JournalFileBackend(
            str(journal), lock_obj=JournalFileOpenLock(str(journal))
        )
    )
    study = optuna.create_study(storage=storage, study_name="project", direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("direction_index", 0, 1), n_trials=1)

    package = tmp_path / "geometry_3d"
    result = language_map_cli.main(
        [
            "project",
            "--cache-dir",
            str(cache),
            "--journal",
            str(journal),
            "--output-dir",
            str(package),
            "--anchor-count",
            "4",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--group-a",
            f"en={tmp_path / 'en-safe.jsonl'}",
            "--group-a",
            f"ru={tmp_path / 'ru-safe.jsonl'}",
            "--group-b",
            f"en={tmp_path / 'en-unsafe.jsonl'}",
            "--group-b",
            f"ru={tmp_path / 'ru-unsafe.jsonl'}",
        ]
    )

    assert result["status"] == "PASS"
    assert result["base_rows"] == 8
    assert result["anchor_count"] == 4
    assert (package / "report.html").is_file()
    private_rows = [
        json.loads(line)
        for line in (package / "private" / "anchors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(private_rows) == 4
    assert all("prompt" in row for row in private_rows)

    rendered = language_map_cli.main(
        ["render", "--package-dir", str(package)]
    )
    assert rendered["status"] == "PASS"
    assert rendered["captured_trials"] == 0
