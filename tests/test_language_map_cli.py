from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from heretic import cli, language_map_cli
from heretic.language_map_cache import capture_residual_cache
from heretic.language_map_data import LanguageFile, load_aligned_corpus
from heretic.supervisor import GpuInfo


def _write_cell(
    path: Path,
    *,
    language: str,
    direction: str,
    count: int = 2,
) -> None:
    rows = []
    prefix = "A" if direction == "safe" else "B"
    for number in range(1, count + 1):
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_dispatches_geometry_map_without_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(language_map_cli, "main", lambda argv: received.append(argv))
    monkeypatch.setattr(cli.sys, "argv", ["hereticMOE", "geometry-map", "analyze"])

    cli.main()

    assert received == [["analyze"]]


def test_prepare_polyguard_dispatches_text_free_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"parquet-placeholder")
    output = tmp_path / "dataset"
    received: list[tuple[Path, Path]] = []

    def materialize(source_path: Path, output_path: Path) -> dict[str, object]:
        received.append((source_path, output_path))
        return {
            "status": "PASS",
            "languages": [f"l{index}" for index in range(17)],
            "directions": {"safe": 482, "unsafe": 238},
            "rows": 12_240,
        }

    monkeypatch.setattr(
        language_map_cli,
        "materialize_polyguard_language_dataset",
        materialize,
        raising=False,
    )

    result = language_map_cli.main(
        [
            "prepare-polyguard",
            "--source",
            str(source),
            "--output-dir",
            str(output),
        ]
    )

    assert received == [(source, output)]
    assert result == {
        "status": "PASS",
        "mode": "prepare-polyguard",
        "languages": 17,
        "safe": 482,
        "unsafe": 238,
        "rows": 12_240,
        "manifest": str((output / "manifest.json").resolve()),
    }


def test_compare_combines_model_language_distances(tmp_path: Path) -> None:
    languages = ["en", "fr", "ru", "zh"]
    matrix = [
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.0, 0.7, 0.8],
        [0.8, 0.7, 0.0, 0.6],
        [0.9, 0.8, 0.6, 0.0],
    ]
    inputs = []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "language_distances.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "method": "full_space_language_geometry_v1",
                    "languages": languages,
                    "distances": matrix,
                }
            ),
            encoding="utf-8",
        )
        inputs.append(directory)
    output = tmp_path / "combined"

    result = language_map_cli.main(
        [
            "compare",
            "--analysis-dir",
            str(inputs[0]),
            "--analysis-dir",
            str(inputs[1]),
            "--output-dir",
            str(output),
            "--min-k",
            "4",
            "--max-k",
            "4",
        ]
    )

    assert result["status"] == "PASS"
    assert result["models"] == 2
    assert result["recommended_languages"] == languages
    assert (output / "language_selection.json").is_file()
    assert (output / "report.html").is_file()


def test_geometry_findings_summarize_layers_without_corpus_text() -> None:
    findings = language_map_cli._geometry_findings(
        {
            "recommended_layer_bounds": [2, 4],
            "diagnostics": {
                "layers": [
                    {
                        "layer": layer,
                        "layer_reliability": reliability,
                        "cross_language_stability": stability,
                        "separation_strength": separation,
                    }
                    for layer, reliability, stability, separation in (
                        (2, 0.7, 0.8, 0.91),
                        (3, 0.9, 0.9, 0.99),
                        (4, 0.8, 0.85, 0.95),
                    )
                ]
            },
        },
        {
            "language_contributions": {
                "en": {"direction_loss": 0.01},
                "zh": {"direction_loss": 0.03},
            },
            "language_selection": {
                "recommended_k": 2,
                "recommended_languages": ["en", "zh"],
            },
        },
    )

    assert findings == {
        "usable_layers": "2-4",
        "strongest_layers": "3,4,2",
        "cross_lang_stability": "85.0%",
        "peak_separation": "0.990",
        "largest_language_loss": "zh 3.00%",
        "recommended_languages": "en,zh",
        "report": "analysis/geometry_3d/report.html",
    }


def test_base_geometry_report_is_rendered_automatically_and_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Path]] = []

    def write_package(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output_dir"])
        output.mkdir()
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        calls.append(("project", output))
        return {"status": "PASS"}

    def render(package: Path, output: Path) -> dict[str, object]:
        output.write_text("report", encoding="utf-8")
        calls.append(("render", package))
        return {"status": "PASS", "sha256": "a" * 64}

    monkeypatch.setattr(language_map_cli, "write_projection_package", write_package)
    monkeypatch.setattr(language_map_cli, "write_interactive_geometry_report", render)
    destination = tmp_path / "analysis"
    destination.mkdir()

    result = language_map_cli._write_base_geometry_report(
        index=[{"row_id": "EN-PG-1"}],
        residuals=torch.zeros((1, 1, 3)),
        output_dir=destination,
        seed=7,
        projection_device="cpu",
    )

    package = destination / "geometry_3d"
    assert result["output"] == str((package / "report.html").resolve())
    assert (package / "report.html").read_text(encoding="utf-8") == "report"
    assert [name for name, _ in calls] == ["project", "render"]


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


def test_dry_run_discovers_asymmetric_inputs_from_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = []
    for language in ("en", "ru"):
        for direction, count in (("safe", 3), ("unsafe", 1)):
            path = tmp_path / f"direction_{language}_{direction}_strict.jsonl"
            _write_cell(
                path,
                language=language,
                direction=direction,
                count=count,
            )
            files.append(
                {
                    "language": language,
                    "direction": direction,
                    "path": path.name,
                    "rows": count,
                    "sha256": _sha256(path),
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": ["en", "ru"],
                "directions": {"safe": 3, "unsafe": 1},
                "rows": 8,
                "files": files,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        language_map_cli,
        "_capture_parallel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not capture")
        ),
    )
    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--dataset-manifest",
            str(manifest),
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "dry-run",
        "languages": ["en", "ru"],
        "rows": 8,
        "rows_per_direction": {"safe": 3, "unsafe": 1},
        "source_rows_per_direction": {"safe": 3, "unsafe": 1},
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
