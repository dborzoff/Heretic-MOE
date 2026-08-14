# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public YAML launch contract for Heretic-MOE.

Users edit one YAML file. The supervisor validates it, applies the small set
of supported command-line overrides, and materializes immutable effective
YAML/TOML artifacts for the internal controller and workers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import tomli_w
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from .config import MultilingualSearchSettings, Settings

IncompatibleContractPolicy = Literal["archive", "new_run", "replace", "fail"]
PostSearchMode = Literal["export", "recheck", "none"]
DeviceMode = Literal["auto", "include"]
GenerationBatchSize = Literal["auto"] | PositiveInt
DirectionMode = Literal["global", "per_layer"]
ExportRole = Literal["Balanced", "Max"]
GenerationBackendName = Literal["dynamic_eager", "compiled_static"]
MetricsContractVersion = Literal["multilingual_v3"]


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    incompatible_contract: IncompatibleContractPolicy = "archive"
    target_trials: PositiveInt = 600
    exploration_trials: int = Field(default=120, ge=0)
    seed: int | None = 20260811
    post_search: PostSearchMode = "export"

    @model_validator(mode="after")
    def validate_trial_counts(self) -> RunSettings:
        if self.exploration_trials > self.target_trials:
            raise ValueError("exploration_trials cannot exceed target_trials")
        return self


class DeviceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: DeviceMode = "auto"
    include: list[str] = Field(default_factory=list)
    max_workers: PositiveInt | None = None
    min_free_gib: float = Field(default=4.0, ge=0.0)
    min_free_fraction: float = Field(default=0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_include(self) -> DeviceSettings:
        normalized = [str(value).strip() for value in self.include]
        if any(not value for value in normalized):
            raise ValueError("devices.include cannot contain empty GPU indices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("devices.include cannot contain duplicate GPU indices")
        self.include = normalized
        if self.mode == "include" and not self.include:
            raise ValueError('devices.mode="include" requires devices.include')
        return self

    def selection_specification(self) -> str:
        if self.include:
            return ",".join(self.include)
        return "auto"


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: Path
    split_root: Path | None = None


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinary_max_new_tokens: PositiveInt = 100
    final_max_new_tokens: PositiveInt = 100
    batch_size: GenerationBatchSize = "auto"
    conditional_nll_batch_size: GenerationBatchSize = "auto"
    max_batch_size: PositiveInt = 4096
    backend: GenerationBackendName = "compiled_static"
    vram_headroom_fraction: float = Field(default=0.10, ge=0.0, le=1.0)


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule_capacity: PositiveInt = 1000
    schedule_seed: int = 20260811
    schedule_version: PositiveInt = 2
    direction_modes: list[DirectionMode] = Field(
        default_factory=lambda: ["global", "per_layer"]
    )
    save_responses: bool = True

    @model_validator(mode="after")
    def validate_direction_modes(self) -> SearchSettings:
        if not self.direction_modes:
            raise ValueError("search.direction_modes cannot be empty")
        if len(set(self.direction_modes)) != len(self.direction_modes):
            raise ValueError("search.direction_modes cannot contain duplicates")
        return self


class MetricsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: MetricsContractVersion = "multilingual_v3"
    max_safe_ppl_drift: float = Field(default=0.005, ge=0.0)
    max_safe_geometry_damage: float = Field(default=1.0, ge=0.0)
    max_language_instability: float = Field(default=1.0, ge=0.0)
    max_category_instability: float = Field(default=1.0, ge=0.0)
    max_empty_rate: float = Field(default=0.0, ge=0.0)
    max_truncated_rate: float = Field(default=0.0, ge=0.0)
    max_safe_d_to_r_rate: float = Field(default=0.02, ge=0.0)


class FinalistSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_n: PositiveInt = 6
    balanced_removal_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    export_roles: list[ExportRole] = Field(default_factory=lambda: ["Balanced", "Max"])

    @model_validator(mode="after")
    def validate_export_roles(self) -> FinalistSettings:
        if not self.export_roles:
            raise ValueError("finalists.export_roles cannot be empty")
        if len(set(self.export_roles)) != len(self.export_roles):
            raise ValueError("finalists.export_roles cannot contain duplicates")
        return self


class GeometrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory: bool = True
    capture_trials: bool = True
    render_html: bool = True


class RecoverySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_heartbeat_seconds: PositiveInt = 15
    worker_timeout_seconds: PositiveInt = 180
    max_restarts_per_gpu: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_timeout(self) -> RecoverySettings:
        if self.worker_timeout_seconds <= self.worker_heartbeat_seconds:
            raise ValueError(
                "worker_timeout_seconds must exceed worker_heartbeat_seconds"
            )
        return self


class LaunchConfig(BaseModel):
    """Complete public configuration accepted by the Heretic-MOE supervisor."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    model: ModelSettings
    run: RunSettings
    devices: DeviceSettings = Field(default_factory=DeviceSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    finalists: FinalistSettings = Field(default_factory=FinalistSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)


@dataclass(frozen=True)
class LaunchOverrides:
    model: str | None = None
    run_root: Path | None = None
    devices: str | None = None
    target_trials: int | None = None
    exploration_trials: int | None = None
    post_search: PostSearchMode | None = None
    incompatible_contract: IncompatibleContractPolicy | None = None


@dataclass(frozen=True)
class EffectiveConfigBundle:
    original_yaml: Path
    effective_yaml: Path
    effective_toml: Path
    manifest: Path


@dataclass(frozen=True)
class RunRootResolution:
    run_root: Path
    archived_root: Path | None = None
    replaced: bool = False
    resumed: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _path_to_text(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def _resolve_optional_path(value: Any, source: Path) -> Any:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return str(candidate)
    return str((source.parent / candidate).resolve())


def _public_payload(config: LaunchConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def effective_resume_contract_sha256(config: LaunchConfig) -> str:
    """Hash settings that must not drift when extending a shared journal."""

    payload = _public_payload(config)
    payload["run"] = {
        "exploration_trials": config.run.exploration_trials,
        "seed": config.run.seed,
    }
    # Resource availability, retry policy, target extension, and post-search role
    # selection do not alter already measured trial semantics.
    payload.pop("devices", None)
    payload.pop("recovery", None)
    payload.pop("finalists", None)
    return _canonical_sha256(payload)


def _legacy_root_hash(path: Path) -> str:
    records: list[tuple[str, int]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_file():
            records.append((child.relative_to(path).as_posix(), child.stat().st_size))
    return _canonical_sha256(records)


def _existing_resume_contract(path: Path) -> tuple[str | None, bool]:
    manifest = path / "config.manifest.json"
    if not manifest.is_file():
        return None, False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False
    if payload.get("marker") != "Heretic-MOE-run-root":
        return None, False
    value = payload.get("launch_contract_sha256")
    return (str(value) if value else None), True


def _next_run_suffix(path: Path) -> Path:
    version = 2
    while True:
        candidate = path.with_name(f"{path.name}_v{version}")
        if not candidate.exists():
            return candidate
        version += 1


def resolve_run_root(
    requested_root: Path,
    config: LaunchConfig,
    policy: IncompatibleContractPolicy,
    *,
    mutate: bool = True,
) -> RunRootResolution:
    """Resolve an incompatible run atomically at whole-root granularity."""

    root = requested_root.resolve()
    expected = effective_resume_contract_sha256(config)
    if not root.exists():
        return RunRootResolution(run_root=root)
    if not root.is_dir():
        raise RuntimeError(f"run root is not a directory: {root}")
    if not any(root.iterdir()):
        return RunRootResolution(run_root=root)

    existing, recognized = _existing_resume_contract(root)
    if recognized and existing == expected:
        return RunRootResolution(run_root=root, resumed=True)

    if policy == "fail":
        raise RuntimeError(
            f"existing run root is incompatible with the requested contract: {root}"
        )
    if policy == "new_run":
        return RunRootResolution(run_root=_next_run_suffix(root))
    if policy == "replace":
        if not recognized:
            raise RuntimeError(
                "replace requires a recognized Heretic-MOE run root with a valid marker"
            )
        if root == Path(root.anchor) or root.parent == root:
            raise RuntimeError(f"refusing to replace broad path: {root}")
        if mutate:
            shutil.rmtree(root)
        return RunRootResolution(run_root=root, replaced=True)

    old_hash = existing or _legacy_root_hash(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = root.with_name(f"{root.name}_archive_{timestamp}_{old_hash[:8]}")
    collision = 2
    while archived.exists():
        archived = root.with_name(
            f"{root.name}_archive_{timestamp}_{old_hash[:8]}_{collision}"
        )
        collision += 1
    if mutate:
        shutil.move(str(root), str(archived))
    return RunRootResolution(run_root=root, archived_root=archived)


def _resolve_config_relative_paths(config: LaunchConfig, source: Path) -> LaunchConfig:
    payload = _public_payload(config)
    root = Path(payload["run"]["root"])
    if not root.is_absolute():
        payload["run"]["root"] = str((source.parent / root).resolve())
    data = dict(payload.get("data") or {})
    for key in ("dataset_root", "split_root"):
        if key in data:
            data[key] = _resolve_optional_path(data[key], source)
    payload["data"] = data
    return LaunchConfig.model_validate(payload)


def _parse_device_override(specification: str) -> tuple[DeviceMode, list[str]]:
    value = specification.strip()
    if value.lower() == "auto":
        return "auto", []
    include = [part.strip() for part in value.split(",")]
    if any(not item for item in include):
        raise ValueError("--devices contains an empty GPU index")
    return "include", include


def load_effective_launch_config(
    source: Path,
    overrides: LaunchOverrides,
) -> LaunchConfig:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Heretic-MOE config.yaml must contain a mapping")

    payload = dict(loaded)
    model = dict(payload.get("model") or {})
    run = dict(payload.get("run") or {})
    devices = dict(payload.get("devices") or {})
    if overrides.model is not None:
        model["path"] = overrides.model
    if overrides.run_root is not None:
        run["root"] = str(overrides.run_root.resolve())
    if overrides.devices is not None:
        mode, include = _parse_device_override(overrides.devices)
        devices["mode"] = mode
        devices["include"] = include
    if overrides.target_trials is not None:
        run["target_trials"] = overrides.target_trials
    if overrides.exploration_trials is not None:
        run["exploration_trials"] = overrides.exploration_trials
    if overrides.post_search is not None:
        run["post_search"] = overrides.post_search
    if overrides.incompatible_contract is not None:
        run["incompatible_contract"] = overrides.incompatible_contract
    payload["model"] = model
    payload["run"] = run
    payload["devices"] = devices
    return _resolve_config_relative_paths(LaunchConfig.model_validate(payload), source)


def build_internal_settings(config: LaunchConfig) -> Settings:
    batch_size = 0 if config.generation.batch_size == "auto" else config.generation.batch_size
    conditional_nll_batch_size = (
        0
        if config.generation.conditional_nll_batch_size == "auto"
        else config.generation.conditional_nll_batch_size
    )
    runtime_root = config.run.root / "runtime"
    geometry_package = (
        (runtime_root / "geometry_3d").as_posix() if config.geometry.trajectory else None
    )
    multilingual = MultilingualSearchSettings(
        enabled=True,
        dataset_root=_path_to_text(config.data.dataset_root),
        split_root=_path_to_text(config.data.split_root),
        runtime_root=runtime_root.as_posix(),
        ordinary_max_new_tokens=config.generation.ordinary_max_new_tokens,
        final_max_new_tokens=config.generation.final_max_new_tokens,
        schedule_seed=config.search.schedule_seed,
        schedule_version=config.search.schedule_version,
        schedule_capacity=config.search.schedule_capacity,
        max_safe_ppl_drift=config.metrics.max_safe_ppl_drift,
        max_safe_geometry_damage=config.metrics.max_safe_geometry_damage,
        max_language_instability=config.metrics.max_language_instability,
        max_category_instability=config.metrics.max_category_instability,
        max_empty_response_rate=config.metrics.max_empty_rate,
        max_truncated_response_rate=config.metrics.max_truncated_rate,
        max_safe_d_to_r_rate=config.metrics.max_safe_d_to_r_rate,
    )
    # ``Settings`` is a BaseSettings model whose ordinary constructor also reads
    # HERETIC_* variables, .env, secrets, and a legacy config.toml.  A launch
    # through config.yaml must be reproducible from YAML + CLI overrides only,
    # so make every Settings field an explicit init value before validation.
    settings_payload = {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
        if not field.is_required()
    }
    settings_payload.update(
        {
            "model": config.model.path,
            "seed": config.run.seed,
            "n_trials": config.run.target_trials,
            "n_startup_trials": config.run.exploration_trials,
            "batch_size": batch_size,
            "conditional_nll_batch_size": conditional_nll_batch_size,
            "max_batch_size": config.generation.max_batch_size,
            "max_response_length": config.generation.ordinary_max_new_tokens,
            "generation_backend": config.generation.backend,
            "generation_batch_target_headroom_fraction": (
                config.generation.vram_headroom_fraction
            ),
            "batch_size_vram_headroom_fraction": (
                config.generation.vram_headroom_fraction
            ),
            "multilingual_search": multilingual,
            "scorers": [],
            "save_trial_responses": config.search.save_responses,
            "geometry_trajectory_package": geometry_package,
            "geometry_capture_evaluation": config.geometry.capture_trials,
            "geometry_render_html": config.geometry.render_html,
        }
    )
    return Settings(**settings_payload)


def _internal_settings_payload(config: LaunchConfig) -> dict[str, Any]:
    settings = build_internal_settings(config)
    payload = settings.model_dump(mode="json", exclude_none=True)
    multilingual = settings.multilingual_search.model_dump(
        mode="json", exclude_none=True
    )
    payload.update(
        {
            "model": settings.model,
            "seed": settings.seed,
            "n_trials": settings.n_trials,
            "n_startup_trials": settings.n_startup_trials,
            "batch_size": settings.batch_size,
            "conditional_nll_batch_size": settings.conditional_nll_batch_size,
            "max_batch_size": settings.max_batch_size,
            "max_response_length": settings.max_response_length,
            "generation_backend": (
                settings.generation_backend.value
                if hasattr(settings.generation_backend, "value")
                else str(settings.generation_backend)
            ),
            "generation_batch_target_headroom_fraction": (
                settings.generation_batch_target_headroom_fraction
            ),
            "batch_size_vram_headroom_fraction": (
                settings.batch_size_vram_headroom_fraction
            ),
            "multilingual_search": multilingual,
            "scorers": [],
            "save_trial_responses": settings.save_trial_responses,
            "geometry_trajectory_package": settings.geometry_trajectory_package,
            "geometry_capture_evaluation": settings.geometry_capture_evaluation,
            "geometry_trial_number_offset": settings.geometry_trial_number_offset,
            "geometry_render_html": settings.geometry_render_html,
        }
    )
    return {key: value for key, value in payload.items() if value is not None}


def write_effective_config_bundle(
    source: Path,
    config: LaunchConfig,
    output_directory: Path,
) -> EffectiveConfigBundle:
    output_directory.mkdir(parents=True, exist_ok=True)
    bundle = EffectiveConfigBundle(
        original_yaml=output_directory / "config.yaml",
        effective_yaml=output_directory / "config.effective.yaml",
        effective_toml=output_directory / "config.effective.toml",
        manifest=output_directory / "config.manifest.json",
    )
    shutil.copyfile(source.resolve(), bundle.original_yaml)

    effective_payload = _public_payload(config)
    bundle.effective_yaml.write_text(
        yaml.safe_dump(effective_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    bundle.effective_toml.write_text(
        tomli_w.dumps(_internal_settings_payload(config)),
        encoding="utf-8",
    )
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (
            bundle.original_yaml,
            bundle.effective_yaml,
            bundle.effective_toml,
        )
    }
    config_sha256 = _canonical_sha256(effective_payload)
    bundle.manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "marker": "Heretic-MOE-run-root",
                "config_sha256": config_sha256,
                "launch_contract_sha256": effective_resume_contract_sha256(config),
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle
