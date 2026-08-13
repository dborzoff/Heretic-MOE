from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from heretic.language_map_projection import write_projection_package
from heretic.language_map_report import write_interactive_geometry_report
from heretic.language_map_trajectory import (
    append_trial_projection,
    initialize_trajectory_package,
    load_text_free_trial_timeline,
)


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "geometry_3d"
    index = []
    for position, language in enumerate(("en", "ru", "zh", "en", "ru", "zh")):
        group = "A" if position % 2 == 0 else "B"
        index.append(
            {
                "index": position,
                "canonical_id": f"{group}{position:04d}",
                "row_id": f"{language.upper()}-{group}{position:04d}",
                "language": language,
                "direction_class": "safe" if group == "A" else "unsafe",
                "category_id": f"C{position % 2 + 1:02d}",
                "source_file": "PRIVATE_SENTINEL_DO_NOT_EMIT.jsonl",
                "source_line": position + 1,
            }
        )
    residuals = torch.randn((6, 2, 5), generator=torch.Generator().manual_seed(5))
    write_projection_package(index=index, residuals=residuals, output_dir=package, seed=5)

    journal_path = tmp_path / "study.log"
    storage = JournalStorage(
        JournalFileBackend(
            str(journal_path),
            lock_obj=JournalFileOpenLock(str(journal_path)),
        )
    )
    study = optuna.create_study(storage=storage, study_name="report", direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("direction_index", 0.0, 1.0)
        trial.set_user_attr("search_phase", "tpe")
        return 0.25

    study.optimize(objective, n_trials=1)
    trajectory = initialize_trajectory_package(
        package_dir=package,
        journal=journal_path,
        reference_residuals=residuals,
        anchor_count=4,
        seed=3,
    )
    candidate = residuals[trajectory["anchor_rows"]].clone()
    candidate[:, :, 0] += 0.2
    append_trial_projection(
        package,
        load_text_free_trial_timeline(journal_path)[0],
        candidate,
        evaluation_residuals=residuals[:2] + 0.1,
        evaluation_prompt_hashes=["a" * 64, "b" * 64],
    )
    return package


def test_report_is_offline_filterable_and_contains_no_private_corpus_text(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    output = package / "report.html"
    result = write_interactive_geometry_report(package, output)
    document = output.read_text(encoding="utf-8")

    assert result["status"] == "PASS"
    assert result["base_rows"] == 6
    assert result["captured_trials"] == 1
    assert "PRIVATE_SENTINEL_DO_NOT_EMIT" not in document
    assert "http://" not in document and "https://" not in document
    for control in (
        "layer-filter",
        "language-filters",
        "group-filters",
        "category-filters",
        "trial-filter",
        "phase-filters",
        "stage-filter",
        "finalist-filter",
        "verdict-filter",
        "arrow-filter",
        "animate-trials",
        "reset-original",
        "show-evaluation",
    ):
        assert f'id="{control}"' in document
    assert "Original → Trial" in document
    assert "optimizer path" in document
    assert "Evaluation points" in document
    assert "data:application/octet-stream;base64," in document
    assert "selectedFinalist" in document
    assert "verdictsFor" in document


def test_report_rejects_sensitive_verdict_payload(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package / "verdicts.jsonl").write_text(
        json.dumps(
            {
                "trial_number": 0,
                "row_id": "EN-A0000",
                "status": "success",
                "prompt": "PRIVATE_SENTINEL_DO_NOT_EMIT",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sensitive"):
        write_interactive_geometry_report(package, package / "report.html")
