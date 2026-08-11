"""Text-free search timelines and append-only projected trial geometry."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator, Literal

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor

from .language_map_projection import (
    ProjectionBasis,
    project_residuals,
    retained_shift_ratio,
)


CoordinateStatus = Literal["not_captured", "captured"]
_PHASES = {"random", "sobol", "exploration", "tpe", "recheck", "finalist"}


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    phase: str
    parameters: dict[str, bool | float | int | str]
    values: tuple[float, ...] | None
    constraints: tuple[float, ...]
    feasible: bool | None
    coordinate_status: CoordinateStatus = "not_captured"

    def to_public_dict(self) -> dict[str, object]:
        return {
            "trial_number": self.number,
            "state": self.state,
            "phase": self.phase,
            "parameters": self.parameters,
            "values": list(self.values) if self.values is not None else None,
            "constraints": list(self.constraints),
            "feasible": self.feasible,
            "coordinate_status": self.coordinate_status,
            "parameters_sha256": sha256(
                json.dumps(
                    self.parameters,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        }


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("trial metadata contains non-finite numeric value")
    return result


def _public_parameter(value: object) -> bool | float | int | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, str) and len(value) <= 64 and value.replace("_", "").isalnum():
        return value
    raise ValueError("unsupported or unsafe public trial parameter")


def _trial_phase(trial: optuna.trial.FrozenTrial) -> str:
    for key in ("search_phase", "phase", "sampler_phase"):
        value = trial.user_attrs.get(key)
        if isinstance(value, str) and value.lower() in _PHASES:
            return value.lower()
    if "recheck_source_trial_index" in trial.user_attrs:
        return "recheck"
    return "search"


def load_text_free_trial_timeline(journal: Path) -> list[TrialRecord]:
    """Load one Optuna study while excluding arbitrary user metadata."""

    journal = Path(journal)
    storage = JournalStorage(
        JournalFileBackend(
            str(journal),
            lock_obj=JournalFileOpenLock(str(journal)),
        )
    )
    summaries = storage.get_all_studies()
    if len(summaries) != 1:
        raise ValueError(f"expected exactly one study, found {len(summaries)}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    records: list[TrialRecord] = []
    for trial in sorted(study.get_trials(deepcopy=False), key=lambda item: item.number):
        values = (
            tuple(_finite_float(value) for value in trial.values)
            if trial.values is not None
            else None
        )
        raw_constraints = trial.user_attrs.get(
            "constraints", trial.system_attrs.get("constraints", ())
        )
        constraints = (
            tuple(_finite_float(value) for value in raw_constraints)
            if isinstance(raw_constraints, (list, tuple))
            else ()
        )
        feasible_value = trial.user_attrs.get("feasible")
        feasible = feasible_value if isinstance(feasible_value, bool) else None
        records.append(
            TrialRecord(
                number=int(trial.number),
                state=trial.state.name.lower(),
                phase=_trial_phase(trial),
                parameters={
                    str(name): _public_parameter(value)
                    for name, value in sorted(trial.params.items())
                },
                values=values,
                constraints=constraints,
                feasible=feasible,
            )
        )
    return records


def _stable_score(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return sha256(payload.encode("utf-8")).hexdigest()


def select_stratified_anchors(
    index: list[dict[str, object]], *, count: int, seed: int = 42
) -> list[int]:
    """Select deterministic anchors balanced across group/language/category."""

    if count <= 0 or count > len(index):
        raise ValueError("anchor count must be within the base-index row count")
    required = {"index", "row_id", "canonical_id", "language", "group", "category_id"}
    if any(not required.issubset(row) for row in index):
        raise ValueError("base index is missing anchor stratification metadata")

    remaining = sorted(
        range(len(index)),
        key=lambda position: _stable_score(seed, index[position]["row_id"]),
    )
    selected: list[int] = []
    group_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    category_counts: dict[tuple[str, str], int] = {}
    stratum_counts: dict[tuple[str, str, str], int] = {}

    while len(selected) < count:
        def priority(position: int) -> tuple[int, int, int, int, str]:
            row = index[position]
            group = str(row["group"])
            language = str(row["language"])
            category = str(row["category_id"])
            return (
                group_counts.get(group, 0),
                language_counts.get(language, 0),
                category_counts.get((group, category), 0),
                stratum_counts.get((group, language, category), 0),
                _stable_score(seed, row["row_id"]),
            )

        position = min(remaining, key=priority)
        remaining.remove(position)
        selected.append(position)
        row = index[position]
        group = str(row["group"])
        language = str(row["language"])
        category = str(row["category_id"])
        group_counts[group] = group_counts.get(group, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1
        category_counts[(group, category)] = category_counts.get((group, category), 0) + 1
        key = (group, language, category)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    return selected


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize_trajectory_package(
    *,
    package_dir: Path,
    journal: Path,
    reference_residuals: Tensor,
    anchor_count: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Add a frozen anchor panel and complete journal timeline to a base package."""

    package_dir = Path(package_dir)
    trajectory_manifest = package_dir / "trajectory_manifest.json"
    if trajectory_manifest.exists():
        raise FileExistsError(trajectory_manifest)
    base_manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    base_index = json.loads((package_dir / "base_index.json").read_text(encoding="utf-8"))
    expected_shape = (
        int(base_manifest["rows"]),
        int(base_manifest["layers"]),
        int(base_manifest["hidden_size"]),
    )
    if tuple(reference_residuals.shape) != expected_shape:
        raise ValueError("reference residual shape does not match base package")

    anchor_rows = select_stratified_anchors(base_index, count=anchor_count, seed=seed)
    anchors = reference_residuals[anchor_rows].detach().cpu().float().contiguous()
    anchor_reference = package_dir / "anchor_reference.safetensors"
    anchor_temporary = anchor_reference.with_suffix(".safetensors.tmp")
    save_file({"residuals": anchors}, str(anchor_temporary))
    os.replace(anchor_temporary, anchor_reference)

    anchor_index = [base_index[position] for position in anchor_rows]
    anchor_index_path = package_dir / "anchor_index.json"
    _write_json_atomic(anchor_index_path, anchor_index)
    timeline = [record.to_public_dict() for record in load_text_free_trial_timeline(journal)]
    timeline_path = package_dir / "journal_trials.jsonl"
    _write_jsonl_atomic(timeline_path, timeline)
    trial_index_path = package_dir / "trial_index.jsonl"
    _write_jsonl_atomic(trial_index_path, [])
    verdicts_path = package_dir / "verdicts.jsonl"
    _write_jsonl_atomic(verdicts_path, [])
    (package_dir / "trials").mkdir(exist_ok=True)

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (anchor_reference, anchor_index_path, timeline_path, trial_index_path, verdicts_path)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "journal_sha256": _sha256(Path(journal)),
        "trials": len(timeline),
        "captured_trials": 0,
        "anchor_count": len(anchor_rows),
        "anchor_rows": anchor_rows,
        "seed": int(seed),
        "files": files,
    }
    _write_json_atomic(trajectory_manifest, manifest)
    return manifest


@contextmanager
def _package_lock(package_dir: Path, timeout: float = 60.0) -> Iterator[None]:
    lock_path = package_dir / ".trajectory.lock"
    deadline = time.monotonic() + timeout
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"trajectory package lock timeout: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _load_basis(package_dir: Path) -> ProjectionBasis:
    tensors = load_file(str(package_dir / "projection_basis.safetensors"))
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    return ProjectionBasis(
        centers=tensors["centers"],
        components=tensors["components"],
        explained_variance=tensors["explained_variance"],
        seed=int(manifest["projection"]["seed"]),
        algorithm=str(manifest["projection"]["algorithm"]),
    )


def append_trial_projection(
    package_dir: Path,
    trial: TrialRecord,
    residuals: Tensor,
    *,
    measurement: str = "anchor_control",
    evaluation_residuals: Tensor | None = None,
    evaluation_prompt_hashes: list[str] | None = None,
) -> dict[str, object]:
    """Atomically append one real measured trial to the trajectory package."""

    package_dir = Path(package_dir)
    with _package_lock(package_dir):
        manifest_path = package_dir / "trajectory_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = package_dir / "trial_index.jsonl"
        entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if any(int(entry["trial_number"]) == trial.number for entry in entries):
            raise ValueError(f"trial {trial.number} already captured")

        reference = load_file(str(package_dir / "anchor_reference.safetensors"))[
            "residuals"
        ]
        if residuals.shape != reference.shape:
            raise ValueError("trial residual shape does not match frozen anchors")
        basis = _load_basis(package_dir)
        points = project_residuals(residuals, basis).cpu().numpy().astype("<f4")
        if not bool(np.isfinite(points).all()):
            raise ValueError("trial projection contains non-finite values")
        retained = retained_shift_ratio(reference, residuals, basis).cpu().numpy()

        evaluation_points: np.ndarray | None = None
        if evaluation_residuals is not None:
            if evaluation_prompt_hashes is None:
                raise ValueError("evaluation prompt hashes are required")
            if evaluation_residuals.ndim != 3:
                raise ValueError("evaluation residuals must be three-dimensional")
            if tuple(evaluation_residuals.shape[1:]) != tuple(reference.shape[1:]):
                raise ValueError("evaluation residual shape does not match anchors")
            if len(evaluation_prompt_hashes) != int(evaluation_residuals.shape[0]):
                raise ValueError("evaluation prompt hash count mismatch")
            if any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in evaluation_prompt_hashes
            ):
                raise ValueError("invalid evaluation prompt hash")
            evaluation_points = (
                project_residuals(evaluation_residuals, basis)
                .cpu()
                .numpy()
                .astype("<f4")
            )
        elif evaluation_prompt_hashes:
            raise ValueError("evaluation residuals are required")

        relative = Path("trials") / f"trial_{trial.number:06d}.f32"
        destination = package_dir / relative
        if destination.exists():
            raise ValueError(f"trial {trial.number} already captured")
        temporary = destination.with_suffix(".f32.tmp")
        evaluation_destination = package_dir / "trials" / f"trial_{trial.number:06d}_evaluation.f32"
        evaluation_temporary = evaluation_destination.with_suffix(".f32.tmp")
        evaluation_index_destination = (
            package_dir / "trials" / f"trial_{trial.number:06d}_evaluation_index.json"
        )
        evaluation_index_temporary = evaluation_index_destination.with_suffix(
            ".json.tmp"
        )
        try:
            points.tofile(temporary)
            os.replace(temporary, destination)
            if evaluation_points is not None and evaluation_prompt_hashes is not None:
                evaluation_points.tofile(evaluation_temporary)
                os.replace(evaluation_temporary, evaluation_destination)
                evaluation_index_temporary.write_text(
                    json.dumps(
                        [
                            {"index": index, "prompt_sha256": prompt_hash}
                            for index, prompt_hash in enumerate(evaluation_prompt_hashes)
                        ],
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(evaluation_index_temporary, evaluation_index_destination)
            entry: dict[str, object] = {
                "trial_number": trial.number,
                "phase": trial.phase,
                "state": trial.state,
                "measurement": measurement,
                "coordinate_status": "captured",
                "shape": list(points.shape),
                "file": relative.as_posix(),
                "sha256": _sha256(destination),
                "retained_shift_ratio": {
                    "mean": float(np.mean(retained)),
                    "p05": float(np.quantile(retained, 0.05)),
                    "minimum": float(np.min(retained)),
                },
            }
            if evaluation_points is not None:
                entry.update(
                    {
                        "evaluation_count": int(evaluation_points.shape[0]),
                        "evaluation_shape": list(evaluation_points.shape),
                        "evaluation_file": evaluation_destination.relative_to(
                            package_dir
                        ).as_posix(),
                        "evaluation_sha256": _sha256(evaluation_destination),
                        "evaluation_index_file": evaluation_index_destination.relative_to(
                            package_dir
                        ).as_posix(),
                        "evaluation_index_sha256": _sha256(
                            evaluation_index_destination
                        ),
                    }
                )
            _write_jsonl_atomic(index_path, [*entries, entry])
            manifest["captured_trials"] = len(entries) + 1
            manifest["files"][index_path.name] = {
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path),
            }
            _write_json_atomic(manifest_path, manifest)
            return entry
        except BaseException:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            evaluation_temporary.unlink(missing_ok=True)
            evaluation_destination.unlink(missing_ok=True)
            evaluation_index_temporary.unlink(missing_ok=True)
            evaluation_index_destination.unlink(missing_ok=True)
            raise
