# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-shot, resume-safe preparation of multilingual search runtime files."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Sequence


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
    parser.add_argument("--device", default="0")
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
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    # Select the physical preparation device before importing the ML runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    from .config import Settings
    from .model import Model
    from .multilingual_contract import load_multilingual_dataset_bundle
    from .multilingual_prepare import (
        fingerprint_local_model,
        prepare_clean_reference_runtime,
        prepare_static_multilingual_runtime,
    )
    from .multilingual_runtime import (
        apply_multilingual_search_mode,
        load_multilingual_worker_runtime,
    )

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
    if int(config.get("batch_size", 0)) <= 0:
        raise ValueError("preparation requires a fixed positive batch_size")
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

    model = Model(settings)
    clean_manifest = prepare_clean_reference_runtime(
        bundle=bundle,
        runtime_root=runtime_root,
        model=model,
        model_fingerprint=str(fingerprint["model_fingerprint"]),
        max_response_length=contract.ordinary_max_new_tokens,
        batch_size=settings.batch_size,
    )
    worker_runtime = load_multilingual_worker_runtime(settings, model)
    final_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_sha256": bundle.manifest["contract_sha256"],
        "model_fingerprint": fingerprint["model_fingerprint"],
        "static_runtime_sha256": static_manifest["static_runtime_sha256"],
        "clean_reference_contract_sha256": clean_manifest[
            "archive_contract_sha256"
        ],
        "worker_runtime_contract_sha256": worker_runtime.manifest[
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
