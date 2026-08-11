from __future__ import annotations

import json
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
import pytest
import torch

from heretic.language_map_projection import write_projection_package
from heretic.language_map_trajectory import (
    append_trial_projection,
    initialize_trajectory_package,
    load_text_free_trial_timeline,
    select_stratified_anchors,
)


def _journal(path: Path) -> Path:
    storage = JournalStorage(
        JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))
    )
    study = optuna.create_study(
        storage=storage,
        study_name="geometry-test",
        directions=("minimize", "minimize"),
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        trial.suggest_float("direction_index", 0.0, 10.0)
        trial.suggest_categorical("direction_scope", ("global", "component"))
        trial.set_user_attr("constraints", [0.0, -0.1])
        trial.set_user_attr("search_phase", "tpe")
        trial.set_user_attr("secret_note", "must-not-leak")
        return float(trial.number), float(trial.number) / 10.0

    study.optimize(objective, n_trials=2)
    return path


def _index() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    languages = ("en", "ru", "zh")
    for index in range(18):
        language = languages[index % len(languages)]
        group = "A" if index % 2 == 0 else "B"
        rows.append(
            {
                "index": index,
                "canonical_id": f"{group}{index:04d}",
                "row_id": f"{language.upper()}-{group}{index:04d}",
                "language": language,
                "group": group,
                "category_id": f"C{index % 3 + 1:02d}",
            }
        )
    return rows


def _residuals() -> torch.Tensor:
    generator = torch.Generator().manual_seed(9)
    return torch.randn((18, 2, 5), generator=generator)


def test_journal_timeline_is_ordered_filtered_and_marks_missing_coordinates(
    tmp_path: Path,
) -> None:
    records = load_text_free_trial_timeline(_journal(tmp_path / "study.log"))

    assert [record.number for record in records] == [0, 1]
    assert [record.coordinate_status for record in records] == [
        "not_captured",
        "not_captured",
    ]
    payload = json.dumps([record.to_public_dict() for record in records])
    assert "must-not-leak" not in payload
    assert "secret_note" not in payload
    assert "direction_index" in payload
    assert "constraints" in payload


def test_anchor_selection_is_deterministic_and_balances_groups_and_languages() -> None:
    first = select_stratified_anchors(_index(), count=12, seed=41)
    second = select_stratified_anchors(_index(), count=12, seed=41)

    assert first == second
    selected = [_index()[position] for position in first]
    group_counts = {group: sum(row["group"] == group for row in selected) for group in ("A", "B")}
    language_counts = {
        language: sum(row["language"] == language for row in selected)
        for language in ("en", "ru", "zh")
    }
    assert abs(group_counts["A"] - group_counts["B"]) <= 1
    assert max(language_counts.values()) - min(language_counts.values()) <= 1
    assert len({row["row_id"] for row in selected}) == 12


def test_trial_projection_append_is_atomic_and_rejects_duplicate_trial(
    tmp_path: Path,
) -> None:
    residuals = _residuals()
    package = tmp_path / "geometry_3d"
    write_projection_package(
        index=[
            {
                **row,
                "direction_class": "safe" if row["group"] == "A" else "unsafe",
            }
            for row in _index()
        ],
        residuals=residuals,
        output_dir=package,
        seed=7,
    )
    journal = _journal(tmp_path / "study.log")
    trajectory = initialize_trajectory_package(
        package_dir=package,
        journal=journal,
        reference_residuals=residuals,
        anchor_count=6,
        seed=13,
    )
    assert trajectory["status"] == "PASS"
    assert len(trajectory["anchor_rows"]) == 6

    records = load_text_free_trial_timeline(journal)
    anchors = trajectory["anchor_rows"]
    candidate = residuals[anchors].clone()
    candidate[:, :, 0] += 0.5
    entry = append_trial_projection(package, records[0], candidate)

    assert entry["trial_number"] == 0
    assert entry["coordinate_status"] == "captured"
    assert (package / entry["file"]).stat().st_size == 6 * 2 * 3 * 4
    index_rows = [
        json.loads(line)
        for line in (package / "trial_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert index_rows == [entry]
    assert not list(package.rglob("*.tmp"))

    with pytest.raises(ValueError, match="already captured"):
        append_trial_projection(package, records[0], candidate)
