# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal resident GPU worker for one clean-reference shard."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tomllib


def _autotune_reference_batch(model: Any, prompts: Sequence[Any], batch_size: int):
    if batch_size != 0 or not hasattr(model, "autotune_generation_batch_size"):
        return None
    return model.autotune_generation_batch_size(prompts, expected_rows=len(prompts))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE clean-reference worker")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    return parser


def run_worker_job(
    job_path: str | Path,
    *,
    device: str,
    worker_id: str,
    start: int,
    end: int,
    model_factory=None,
) -> dict[str, Any]:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    if job.get("schema_version") != 1:
        raise ValueError("unsupported clean-reference worker job schema")
    with Path(job["config_path"]).open("rb") as stream:
        config = tomllib.load(stream)
    if job.get("model"):
        config["model"] = str(job["model"])
    multilingual = config.get("multilingual_search")
    if not isinstance(multilingual, dict) or not multilingual.get("enabled"):
        raise ValueError("worker config does not enable multilingual search")
    multilingual["runtime_root"] = str(job["runtime_root"])
    config["device_map"] = "auto"
    config["batch_size"] = int(job["batch_size"])
    config["max_response_length"] = int(job["max_response_length"])

    from .clean_reference_archive import build_clean_reference_archive
    from .config import Settings, generation_runtime_contract
    from .language_map_directions import load_direction_map_package
    from .multilingual_contract import load_multilingual_dataset_bundle
    from .utils import Prompt

    settings = Settings.model_validate(config)
    contract = settings.multilingual_search
    bundle = load_multilingual_dataset_bundle(
        dataset_root=str(contract.dataset_root),
        split_root=contract.split_root,
        languages=tuple(contract.languages),
        direction_rows_per_cell=contract.direction_rows_per_cell,
        trial_rows_per_cell=contract.trial_rows_per_cell,
        final_rows_per_cell=contract.final_rows_per_cell,
    )
    if not 0 <= start < end <= len(bundle.trial_rows):
        raise ValueError("clean-reference worker range is out of bounds")
    profile, direction_manifest = load_direction_map_package(
        Path(job["runtime_root"]) / "clean_map" / "directions"
    )
    print(
        json.dumps(
            {"event": "worker_phase", "phase": "model_load", "worker_id": worker_id},
            sort_keys=True,
        ),
        flush=True,
    )
    model = model_factory(settings) if model_factory is not None else None
    if model is None:
        from .model import Model

        model = Model(settings)
    print(
        json.dumps(
            {"event": "worker_phase", "phase": "model_ready", "worker_id": worker_id},
            sort_keys=True,
        ),
        flush=True,
    )
    if hasattr(model, "set_batch_event_sink"):
        model.set_batch_event_sink(
            lambda event: print(
                json.dumps({**event, "worker_id": worker_id}, sort_keys=True),
                flush=True,
            )
        )
    shard_rows = bundle.trial_rows[start:end]
    shard_prompts = [Prompt(system="", user=row.prompt) for row in shard_rows]
    if hasattr(model, "prepare_prompt_cache"):
        prepared = model.prepare_prompt_cache(shard_prompts)
        packed = None
        if hasattr(model, "pin_prompt_cache"):
            packed = model.pin_prompt_cache()
        print(
            json.dumps(
                {
                    "event": "token_cache_ready",
                    "worker_id": worker_id,
                    "rows": int(prepared.get("rows", len(shard_rows)))
                    if isinstance(prepared, dict)
                    else len(shard_rows),
                    "tokens": int(packed.get("tokens", 0))
                    if isinstance(packed, dict)
                    else 0,
                    "pinned": bool(packed.get("pinned", False))
                    if isinstance(packed, dict)
                    else False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _autotune_reference_batch(model, shard_prompts, int(job["batch_size"]))
    output_dir = Path(job["shards_root"]) / f"{start:08d}-{end:08d}"

    def progress(completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "event": "reference_worker_progress",
                    "scope": "worker",
                    "worker_id": worker_id,
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = build_clean_reference_archive(
        model=model,
        rows=shard_rows,
        refusal_direction=profile.consensus_refusal_direction,
        output_dir=output_dir,
        dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
        direction_sha256=str(direction_manifest["package_sha256"]),
        model_fingerprint=str(job["model_fingerprint"]),
        max_response_length=int(job["max_response_length"]),
        batch_size=int(job["batch_size"]),
        generation_contract=generation_runtime_contract(settings),
        progress=progress,
    )
    return {
        "event": "reference_worker_complete",
        "worker_id": worker_id,
        "device": str(device),
        "start": start,
        "end": end,
        "rows": int(manifest["rows"]),
        "batch_size": int(manifest["batch_size"]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    result = run_worker_job(
        args.job,
        device=args.device,
        worker_id=args.worker_id,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
