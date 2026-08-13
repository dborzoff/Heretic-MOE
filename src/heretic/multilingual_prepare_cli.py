# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-shot, resume-safe preparation of multilingual search runtime files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tomllib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hereticMOE prepare-multilingual",
        description="Freeze and verify one model's multilingual v3 runtime.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--direction-source", required=True, type=Path)
    parser.add_argument("--srg-source", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", help="Legacy single-GPU shorthand.")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--batch-size", type=int)
    return parser


def _write_or_verify(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"existing runtime manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.device is not None and args.devices != "0":
        raise ValueError("--device cannot be combined with --devices")
    devices = tuple(
        dict.fromkeys(
            part.strip()
            for part in (args.device or args.devices).split(",")
            if part.strip()
        )
    )
    if not devices:
        raise ValueError("at least one preparation GPU is required")
    if args.batch_size is not None and args.batch_size < 0:
        raise ValueError("--batch-size must be nonnegative")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    from .clean_reference_archive import merge_clean_reference_archives
    from .config import Settings, generation_runtime_contract
    from .language_map_controller import (
        GeometryWorkerSpec,
        run_worker_processes,
        worker_environment,
    )
    from .multilingual_contract import load_multilingual_dataset_bundle
    from .multilingual_prepare import (
        fingerprint_local_model,
        prepare_static_multilingual_runtime,
    )
    from .multilingual_runtime import (
        apply_multilingual_search_mode,
        load_multilingual_search_evaluator,
    )
    from .multilingual_search_evaluator import MultilingualConstraintContract

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    if args.model:
        config["model"] = args.model
    multilingual = config.get("multilingual_search")
    if not isinstance(multilingual, dict) or not multilingual.get("enabled"):
        raise ValueError("config does not enable multilingual search v3")
    runtime_root = args.runtime_root.resolve()
    multilingual["runtime_root"] = runtime_root.as_posix()
    config["device_map"] = "auto"
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    settings = Settings.model_validate(config)
    apply_multilingual_search_mode(settings)
    contract = settings.multilingual_search
    bundle = load_multilingual_dataset_bundle(
        dataset_root=str(contract.dataset_root),
        split_root=contract.split_root,
        languages=tuple(contract.languages),
        direction_rows_per_cell=contract.direction_rows_per_cell,
        trial_rows_per_cell=contract.trial_rows_per_cell,
        calibration_rows_per_language=contract.calibration_rows_per_language,
    )
    static_manifest = prepare_static_multilingual_runtime(
        bundle=bundle,
        direction_source=args.direction_source,
        srg_source=args.srg_source,
        runtime_root=runtime_root,
        languages=tuple(contract.languages),
        schedule_seed=contract.schedule_seed,
        schedule_capacity=contract.schedule_capacity,
        expected_per_direction=contract.trial_rows_per_cell,
    )
    print(
        json.dumps(
            {
                "event": "multilingual_static_ready",
                "status": "PASS",
                "rows_per_trial": static_manifest["rows_per_trial"],
                "schedule_trials": static_manifest["schedule_trials"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model_path = Path(settings.model)
    if not model_path.is_dir():
        raise ValueError(
            "strict multilingual preparation requires a local model directory"
        )
    fingerprint = fingerprint_local_model(model_path)
    _write_or_verify(runtime_root / "model" / "manifest.json", fingerprint)
    print(
        json.dumps(
            {
                "event": "model_fingerprint_ready",
                "status": "PASS",
                "files": fingerprint["files"],
                "bytes": fingerprint["bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    shards_root = runtime_root / "clean_trial_reference_shards"
    worker_count = min(len(devices), len(bundle.trial_rows))
    boundaries = [
        len(bundle.trial_rows) * index // worker_count
        for index in range(worker_count + 1)
    ]
    job_path = runtime_root / "clean_reference_job.json"
    _write_or_verify(
        job_path,
        {
            "schema_version": 1,
            "config_path": str(args.config.resolve()),
            "runtime_root": str(runtime_root),
            "shards_root": str(shards_root),
            "model": str(settings.model),
            "model_fingerprint": str(fingerprint["model_fingerprint"]),
            "batch_size": int(settings.batch_size),
            "max_response_length": int(contract.ordinary_max_new_tokens),
        },
    )
    specifications = []
    for index, device in enumerate(devices[:worker_count]):
        start, end = boundaries[index], boundaries[index + 1]
        command = (
            sys.executable,
            "-u",
            "-c",
            "from heretic.multilingual_reference_worker import main; main()",
            "--job",
            str(job_path),
            "--device",
            str(device),
            "--worker-id",
            f"gpu-{device}",
            "--start",
            str(start),
            "--end",
            str(end),
        )
        specifications.append(
            GeometryWorkerSpec(
                device=str(device),
                worker_id=f"gpu-{device}",
                command=command,
                environment=worker_environment(
                    os.environ,
                    device=str(device),
                    cpu_threads=max(1, (os.cpu_count() or 4) // worker_count),
                ),
            )
        )
    exits = run_worker_processes(specifications)
    failures = {key: value for key, value in exits.items() if value != 0}
    if failures:
        raise RuntimeError(f"clean-reference worker failure(s): {failures}")
    from .language_map_directions import load_direction_map_package

    _, direction_manifest = load_direction_map_package(
        runtime_root / "clean_map" / "directions"
    )
    shard_dirs = [
        shards_root / f"{boundaries[index]:08d}-{boundaries[index + 1]:08d}"
        for index in range(worker_count)
    ]
    clean_manifest = merge_clean_reference_archives(
        shard_dirs=shard_dirs,
        rows=bundle.trial_rows,
        output_dir=runtime_root / "clean_trial_reference",
        dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
        direction_sha256=str(direction_manifest["package_sha256"]),
        model_fingerprint=str(fingerprint["model_fingerprint"]),
        max_response_length=int(contract.ordinary_max_new_tokens),
        batch_size=int(settings.batch_size),
        generation_contract=generation_runtime_contract(settings),
    )
    job_path.unlink(missing_ok=True)
    # Contract assembly does not execute either object. This validates every
    # frozen component without loading a third redundant copy of the model.
    _, worker_runtime_manifest = load_multilingual_search_evaluator(
        bundle=bundle,
        runtime_root=runtime_root,
        model=object(),
        srg_scorer=object(),
        constraints=MultilingualConstraintContract(
            max_safe_ppl_drift=float(contract.max_safe_ppl_drift),
            max_safe_geometry_damage=float(contract.max_safe_geometry_damage),
            max_language_instability=float(contract.max_language_instability),
            max_category_instability=float(contract.max_category_instability),
            max_empty_response_rate=float(contract.max_empty_response_rate),
            max_truncated_response_rate=float(contract.max_truncated_response_rate),
            max_safe_d_to_r_rate=float(contract.max_safe_d_to_r_rate),
        ),
        expected_per_direction=contract.trial_rows_per_cell,
        expected_languages=tuple(contract.languages),
        expected_generation_contract=generation_runtime_contract(settings),
    )
    final_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_sha256": bundle.manifest["contract_sha256"],
        "model_fingerprint": fingerprint["model_fingerprint"],
        "static_runtime_sha256": static_manifest["static_runtime_sha256"],
        "clean_reference_contract_sha256": clean_manifest["archive_contract_sha256"],
        "worker_runtime_contract_sha256": worker_runtime_manifest[
            "runtime_contract_sha256"
        ],
        "direction_rows": len(bundle.direction_rows),
        "trial_pool_rows": len(bundle.trial_rows),
        "rows_per_trial": static_manifest["rows_per_trial"],
        "schedule_trials": static_manifest["schedule_trials"],
        "srg_calibration_rows": len(bundle.search_rows),
        "final_holdout_rows": len(bundle.final_rows),
    }
    _write_or_verify(runtime_root / "manifest.json", final_manifest)
    print(
        json.dumps(
            {
                "event": "multilingual_runtime_ready",
                "status": "PASS",
                "runtime_root": str(runtime_root),
                "rows": clean_manifest["rows"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return final_manifest
