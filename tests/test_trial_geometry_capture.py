from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
import torch

from heretic.language_map_projection import write_projection_package
from heretic.language_map_trajectory import (
    TrialRecord,
    initialize_trajectory_package,
)
from heretic.plugin import Context
from heretic.trial_geometry_capture import TrialGeometrySession
from heretic.utils import Prompt


def test_context_captures_first_token_residuals_without_duplicate_generation() -> None:
    class FakeModel:
        response_calls = 0
        residual_calls = 0

        def get_responses_batched(self, prompts, skip_special_tokens=True):
            self.response_calls += 1
            return ["synthetic" for _ in prompts]

        def get_residuals_batched(self, prompts):
            self.residual_calls += 1
            return torch.ones((len(prompts), 2, 4))

    captured: list[tuple[list[Prompt], torch.Tensor]] = []
    model = FakeModel()
    context = Context(
        SimpleNamespace(save_trial_responses=False),
        model,
        response_archive_id=3,
        residual_capture=lambda prompts, values: captured.append((prompts, values)),
    )
    prompts = [Prompt(system="system", user="one"), Prompt(system="", user="two")]

    assert context.get_responses(prompts) == ["synthetic", "synthetic"]
    assert context.get_responses(prompts) == ["synthetic", "synthetic"]
    assert model.response_calls == 1
    assert model.residual_calls == 1
    assert len(captured) == 1
    assert captured[0][1].shape == (2, 2, 4)


def test_trial_session_commits_control_and_evaluation_geometry(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(4)
    reference = torch.randn((8, 2, 4), generator=generator)
    index = [
        {
            "index": position,
            "canonical_id": f"A{position:04d}",
            "row_id": f"EN-A{position:04d}",
            "language": "en" if position % 2 == 0 else "ru",
            "direction_class": "safe" if position % 2 == 0 else "unsafe",
            "category_id": f"C{position % 2 + 1:02d}",
        }
        for position in range(8)
    ]
    package = tmp_path / "geometry_3d"
    write_projection_package(index=index, residuals=reference, output_dir=package)
    journal = tmp_path / "study.log"
    storage = JournalStorage(
        JournalFileBackend(
            str(journal), lock_obj=JournalFileOpenLock(str(journal))
        )
    )
    study = optuna.create_study(storage=storage, study_name="capture", direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("direction_index", 0, 1), n_trials=1)
    trajectory = initialize_trajectory_package(
        package_dir=package,
        journal=journal,
        reference_residuals=reference,
        anchor_count=4,
    )
    private = package / "private"
    private.mkdir()
    (private / "anchors.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "base_index": position,
                    "row_id": index[position]["row_id"],
                    "system": "system",
                    "prompt": f"private-{position}",
                }
            )
            + "\n"
            for position in trajectory["anchor_rows"]
        ),
        encoding="utf-8",
    )

    class FakeModel:
        def get_residuals_batched(self, prompts):
            assert len(prompts) == 4
            values = reference[trajectory["anchor_rows"]].clone()
            values[:, :, 0] += 0.25
            return values

    session = TrialGeometrySession(package, trial_number=0)
    evaluation_prompts = [
        Prompt(system="system", user="evaluation-one"),
        Prompt(system="system", user="evaluation-two"),
    ]
    session.capture_evaluation(
        evaluation_prompts,
        torch.randn((2, 2, 4), generator=generator),
    )
    entry = session.finalize(
        FakeModel(),
        TrialRecord(
            number=0,
            state="complete",
            phase="tpe",
            parameters={"direction_index": 0.5},
            values=(0.1,),
            constraints=(0.0,),
            feasible=True,
        ),
    )

    assert entry["coordinate_status"] == "captured"
    assert entry["evaluation_count"] == 2
    assert (package / entry["file"]).is_file()
    assert (package / entry["evaluation_file"]).is_file()
    assert (package / entry["evaluation_index_file"]).is_file()
    public_eval_index = json.loads(
        (package / entry["evaluation_index_file"]).read_text(encoding="utf-8")
    )
    assert len(public_eval_index) == 2
    assert all(set(row) == {"index", "prompt_sha256"} for row in public_eval_index)
