# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare the clean R reference only after TOP-6 has been frozen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE prepare-final-holdout")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--top-six-manifest", required=True, type=Path)
    parser.add_argument("--device", help="Legacy single-GPU shorthand.")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if not args.config.is_file() or not args.top_six_manifest.is_file():
        raise FileNotFoundError("final-holdout preparation inputs are incomplete")
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
        raise ValueError("at least one final-holdout GPU is required")
    if args.worker and len(devices) != 1:
        raise ValueError("a final-holdout worker requires exactly one GPU")
    if args.worker:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]

    from .config import Settings, generation_runtime_contract
    from .language_map_directions import load_direction_map_package
    from .model import Model
    from .multilingual_contract import load_multilingual_dataset_bundle
    from .multilingual_final_holdout import (
        build_final_holdout_archive,
        merge_final_holdout_archives,
    )
    from .multilingual_prepare import fingerprint_local_model
    from .multilingual_runtime import (
        apply_multilingual_search_mode,
        build_multilingual_srg_scorer,
    )
    from .utils import Prompt

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    multilingual = config.get("multilingual_search")
    if not isinstance(multilingual, dict) or not multilingual.get("enabled"):
        raise ValueError("config does not enable multilingual search v4")
    runtime_root = args.runtime_root.resolve()
    multilingual["runtime_root"] = runtime_root.as_posix()
    config["device_map"] = "auto"
    settings = Settings.model_validate(config)
    apply_multilingual_search_mode(settings)
    contract = settings.multilingual_search
    bundle = load_multilingual_dataset_bundle(
        dataset_root=str(contract.dataset_root),
        split_root=contract.split_root,
        languages=tuple(contract.languages),
        direction_rows_per_cell=contract.direction_rows_per_cell,
        trial_rows_per_cell=contract.trial_rows_per_cell,
        final_rows_per_cell=contract.final_rows_per_cell,
    )
    top_six = json.loads(args.top_six_manifest.read_text(encoding="utf-8"))
    top_six_sha = str(top_six.get("shortlist_contract_sha256", ""))
    if top_six.get("status") != "FROZEN" or len(top_six_sha) != 64:
        raise ValueError("TOP-6 manifest is not frozen")
    if not args.worker:
        from .language_map_controller import (
            GeometryWorkerSpec,
            run_worker_processes,
            worker_environment,
        )

        worker_count = min(len(devices), len(bundle.final_rows))
        boundaries = [
            len(bundle.final_rows) * index // worker_count
            for index in range(worker_count + 1)
        ]
        shards_root = runtime_root / "final_holdout_reference_shards"
        specifications = []
        for index, device in enumerate(devices[:worker_count]):
            start, end = boundaries[index], boundaries[index + 1]
            output_dir = shards_root / f"{start:08d}-{end:08d}"
            command = [
                sys.executable,
                "-u",
                "-c",
                "from heretic.multilingual_final_holdout_cli import main; main()",
                "--worker",
                "--config",
                str(args.config.resolve()),
                "--runtime-root",
                str(runtime_root),
                "--top-six-manifest",
                str(args.top_six_manifest.resolve()),
                "--devices",
                str(device),
                "--start",
                str(start),
                "--end",
                str(end),
                "--output-dir",
                str(output_dir),
            ]
            specifications.append(
                GeometryWorkerSpec(
                    device=str(device),
                    worker_id=f"gpu-{device}",
                    command=tuple(command),
                    environment=worker_environment(
                        os.environ,
                        device=str(device),
                        cpu_threads=max(1, (os.cpu_count() or 4) // worker_count),
                    ),
                )
            )
        exits = run_worker_processes(
            specifications,
            stage_name="Final holdout reference",
            total_rows=len(bundle.final_rows),
            next_action="TOP-6 finalist recheck",
        )
        failures = {
            worker_id: exit_code
            for worker_id, exit_code in exits.items()
            if exit_code
        }
        if failures:
            raise RuntimeError(f"final-holdout worker failure(s): {failures}")
        model_path = Path(settings.model)
        if not model_path.is_dir():
            raise ValueError("final-holdout preparation requires a local model directory")
        model_fingerprint = fingerprint_local_model(model_path)["model_fingerprint"]
        profile, _ = load_direction_map_package(
            runtime_root / "clean_map" / "directions"
        )
        srg_profile = json.loads(
            (runtime_root / "srg_profile" / "calibration_profile.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = merge_final_holdout_archives(
            shard_dirs=[
                shards_root / f"{boundaries[index]:08d}-{boundaries[index + 1]:08d}"
                for index in range(worker_count)
            ],
            rows=bundle.final_rows,
            refusal_direction=profile.consensus_refusal_direction,
            srg_profile=srg_profile,
            output_dir=runtime_root / "final_holdout_reference",
            dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
            model_fingerprint=str(model_fingerprint),
            top_six_contract_sha256=top_six_sha,
            max_response_length=contract.final_max_new_tokens,
            generation_contract=generation_runtime_contract(settings),
        )
        print(json.dumps({"event": "final_holdout_reference_ready", "status": "PASS", "rows": manifest["rows"], "shards": worker_count}, sort_keys=True), flush=True)
        return manifest
    start = int(args.start) if args.worker else 0
    end = (
        int(args.end)
        if args.worker and args.end is not None
        else len(bundle.final_rows)
    )
    selected_rows = bundle.final_rows[start:end]
    model_path = Path(settings.model)
    if not model_path.is_dir():
        raise ValueError("final-holdout preparation requires a local model directory")
    model_fingerprint = fingerprint_local_model(model_path)["model_fingerprint"]
    model = Model(settings)
    model.prepare_prompt_cache(
        [Prompt(system="", user=row.prompt) for row in selected_rows]
    )
    model.pin_prompt_cache()
    scorer = build_multilingual_srg_scorer(settings, model, runtime_root)
    profile, _ = load_direction_map_package(runtime_root / "clean_map" / "directions")
    srg_profile = json.loads(
        (runtime_root / "srg_profile" / "calibration_profile.json").read_text(
            encoding="utf-8"
        )
    )
    output_dir = args.output_dir if args.worker else runtime_root / "final_holdout_reference"

    def progress(completed: int, total: int, batch_size: int) -> None:
        print(
            json.dumps(
                {
                    "event": "final_holdout_worker_progress",
                    "worker_id": f"gpu-{devices[0]}",
                    "completed": completed,
                    "total": total,
                    "batch_size": batch_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = build_final_holdout_archive(
        model=model,
        rows=selected_rows,
        refusal_direction=profile.consensus_refusal_direction,
        srg_scorer=scorer,
        srg_profile=srg_profile,
        output_dir=output_dir,
        dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
        model_fingerprint=str(model_fingerprint),
        top_six_contract_sha256=top_six_sha,
        max_response_length=contract.final_max_new_tokens,
        generation_contract=generation_runtime_contract(settings),
        progress=progress,
    )
    print(
        json.dumps(
            {
                "event": "final_holdout_reference_ready",
                "status": "PASS",
                "rows": manifest["rows"],
                "archive_contract_sha256": manifest["archive_contract_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest
