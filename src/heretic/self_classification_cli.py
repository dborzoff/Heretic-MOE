# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public controller for multilingual self-classification benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .self_classification import PromptVariant
from .self_classification_data import (
    load_classification_rows,
    verify_result_coverage,
)
from .self_classification_report import (
    write_consensus_reports,
    write_model_reports,
    write_pilot_reports,
)
from .utils import get_file_sha256

LANGUAGES = ("en", "ru", "zh", "ja", "fr")
LEGACY_PILOT_VARIANTS = (
    PromptVariant.PHRASE,
    PromptVariant.NUMBER,
    PromptVariant.CODE_PERMUTED,
)
CONSENSUS_VARIANTS = (
    PromptVariant.CODE_PERMUTED,
    PromptVariant.CODE_SHIFT_1,
    PromptVariant.CODE_SHIFT_2,
    PromptVariant.CODE_SHIFT_3,
    PromptVariant.WORD_ORDER_0,
    PromptVariant.WORD_ORDER_1,
    PromptVariant.WORD_ORDER_2,
    PromptVariant.WORD_ORDER_3,
)


def _devices(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("devices must be unique comma-separated IDs")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE self-classify")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--dataset-manifest", required=True, type=Path)
    pilot.add_argument("--model", required=True, type=Path)
    pilot.add_argument("--output-dir", required=True, type=Path)
    pilot.add_argument("--devices", type=_devices, default=("0", "1"))
    pilot.add_argument("--batch-size", type=int, default=32)
    pilot.add_argument("--dtype", default="bfloat16")
    pilot.add_argument("--dry-run", action="store_true")

    models = subparsers.add_parser("run-models")
    models.add_argument("--dataset-manifest", required=True, type=Path)
    models.add_argument("--model-root", required=True, type=Path)
    models.add_argument("--model", action="append", type=Path, default=[])
    models.add_argument("--output-dir", required=True, type=Path)
    models.add_argument("--devices", type=_devices, default=("0", "1"))
    models.add_argument("--batch-size", type=int, default=32)
    models.add_argument("--dtype", default="bfloat16")
    models.add_argument("--max-models", type=int, default=10)
    models.add_argument("--max-weight-gib", type=float, default=20.0)
    models.add_argument("--dry-run", action="store_true")

    consensus = subparsers.add_parser("consensus")
    consensus.add_argument("--dataset-manifest", required=True, type=Path)
    consensus.add_argument("--model-root", required=True, type=Path)
    consensus.add_argument("--model", action="append", type=Path, default=[])
    consensus.add_argument("--output-dir", required=True, type=Path)
    consensus.add_argument("--devices", type=_devices, default=("0", "1"))
    consensus.add_argument("--batch-size", type=int, default=32)
    consensus.add_argument("--dtype", default="bfloat16")
    consensus.add_argument("--max-models", type=int, default=20)
    consensus.add_argument("--max-weight-gib", type=float, default=20.0)
    consensus.add_argument("--exclude-model-id", action="append", default=[])
    consensus.add_argument("--dry-run", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--job", required=True, type=Path)
    worker.add_argument("--device", required=True)
    worker.add_argument("--worker-id", required=True)
    return parser


def discover_candidate_models(
    model_root: str | Path,
    *,
    max_weight_bytes: int,
    max_models: int,
) -> list[Path]:
    root = Path(model_root)
    result: list[tuple[int, str, Path]] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name.lower()):
        if not path.is_dir() or not (path / "config.json").is_file():
            continue
        tokenizer_path = path / "tokenizer_config.json"
        if not tokenizer_path.is_file():
            continue
        try:
            tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not tokenizer.get("chat_template"):
            continue
        weights = sum(file.stat().st_size for file in path.rglob("*.safetensors"))
        if weights <= 0 or weights > max_weight_bytes:
            continue
        result.append((weights, path.name.lower(), path.resolve()))
    result.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in result[:max_models]]


def _read_result_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"result row in {path.name} is not an object")
            prohibited = {"prompt", "response", "raw_output", "text"} & value.keys()
            if prohibited:
                raise ValueError(f"result contains prohibited fields: {sorted(prohibited)}")
            rows.append(value)
    return rows


def merge_result_files(
    paths: Sequence[str | Path],
    output_path: str | Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys: set[tuple[str, str, str]] = set()
    for raw_path in paths:
        for row in _read_result_rows(Path(raw_path)):
            key = str(row["model_id"]), str(row["row_id"]), str(row["variant"])
            if key in keys:
                raise ValueError(f"duplicate result key: {key}")
            keys.add(key)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["model_id"]),
            str(row["row_id"]),
            str(row["variant"]),
        )
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "model"


def _write_job(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_jobs(
    jobs: Sequence[tuple[Path, str]],
    devices: Sequence[str],
) -> list[dict[str, object]]:
    pending = list(jobs)
    available = list(devices)
    active: dict[subprocess.Popen[Any], tuple[Path, str, str]] = {}
    completed: list[dict[str, object]] = []
    while pending or active:
        while pending and available:
            job_path, worker_id = pending.pop(0)
            device = available.pop(0)
            command = [
                sys.executable,
                "-u",
                "-m",
                "heretic.self_classification_worker",
                "--job",
                str(job_path),
                "--device",
                device,
                "--worker-id",
                worker_id,
            ]
            print(
                json.dumps(
                    {
                        "event": "classification_worker_start",
                        "worker_id": worker_id,
                        "device": device,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            process = subprocess.Popen(command)
            active[process] = (job_path, worker_id, device)
        finished = [process for process in active if process.poll() is not None]
        for process in finished:
            job_path, worker_id, device = active.pop(process)
            return_code = int(process.returncode or 0)
            completed.append(
                {
                    "job_path": str(job_path),
                    "worker_id": worker_id,
                    "device": device,
                    "return_code": return_code,
                }
            )
            available.append(device)
            print(
                json.dumps(
                    {
                        "event": "classification_worker_exit",
                        "worker_id": worker_id,
                        "device": device,
                        "return_code": return_code,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if active and not finished:
            time.sleep(1.0)
    return completed


def _expected_keys(
    model_ids: Sequence[str],
    rows,
    variants: Sequence[PromptVariant],
) -> set[tuple[str, str, str]]:
    return {
        (model_id, row.row_id, variant.value)
        for model_id in model_ids
        for row in rows
        for variant in variants
    }


def _pilot(args: argparse.Namespace) -> dict[str, object]:
    rows = load_classification_rows(args.dataset_manifest, LANGUAGES)
    if args.dry_run:
        return {
            "status": "PASS",
            "mode": "pilot-dry-run",
            "rows": len(rows),
            "variants": len(LEGACY_PILOT_VARIANTS),
            "tasks": len(rows) * len(LEGACY_PILOT_VARIANTS),
            "workers": len(args.devices),
        }
    output_dir = Path(args.output_dir).resolve() / "pilot"
    jobs_root = output_dir / "jobs"
    parts_root = output_dir / "parts"
    model = Path(args.model).resolve()
    model_id = model.name
    jobs: list[tuple[Path, str]] = []
    part_paths: list[Path] = []
    for index, _device in enumerate(args.devices):
        worker_id = f"pilot-{index}"
        part_path = parts_root / f"{worker_id}.jsonl"
        job_path = jobs_root / f"{worker_id}.json"
        _write_job(
            job_path,
            {
                "schema_version": 1,
                "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                "languages": list(LANGUAGES),
                "model": str(model),
                "model_id": model_id,
                "variants": [value.value for value in LEGACY_PILOT_VARIANTS],
                "output_path": str(part_path),
                "batch_size": int(args.batch_size),
                "dtype": str(args.dtype),
                "shard_index": index,
                "shard_count": len(args.devices),
            },
        )
        jobs.append((job_path, worker_id))
        part_paths.append(part_path)
    statuses = _run_jobs(jobs, args.devices)
    failed = [value for value in statuses if value["return_code"] != 0]
    if failed:
        raise RuntimeError(f"pilot workers failed: {failed}")
    merged_path = output_dir / "pilot_rows.jsonl"
    merged = merge_result_files(part_paths, merged_path)
    verify_result_coverage(
        merged_path,
        _expected_keys([model_id], rows, LEGACY_PILOT_VARIANTS),
    )
    summary = write_pilot_reports(output_dir, merged)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_sha256": get_file_sha256(Path(args.dataset_manifest)),
        "model": str(model),
        "model_id": model_id,
        "rows": len(rows),
        "tasks": len(merged),
        "winner": summary["winner"],
        "result_sha256": get_file_sha256(merged_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _model_paths(args: argparse.Namespace) -> list[Path]:
    if args.model:
        values = [Path(value).resolve() for value in args.model]
        if len(values) != len(set(values)):
            raise ValueError("model paths must be unique")
        return values
    return discover_candidate_models(
        args.model_root,
        max_weight_bytes=int(float(args.max_weight_gib) * 1024**3),
        max_models=int(args.max_models),
    )


def _run_models(args: argparse.Namespace) -> dict[str, object]:
    rows = load_classification_rows(args.dataset_manifest, LANGUAGES)
    output_root = Path(args.output_dir).resolve()
    pilot_summary_path = output_root / "pilot" / "pilot_summary.json"
    if not pilot_summary_path.is_file():
        raise FileNotFoundError("verified pilot summary is required before model run")
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
    if pilot_summary.get("status") != "PASS":
        raise ValueError("pilot summary is not PASS")
    winner = PromptVariant(str(pilot_summary["winner"]))
    models = _model_paths(args)
    if not models:
        raise ValueError("no compatible local chat models discovered")
    if args.dry_run:
        return {
            "status": "PASS",
            "mode": "models-dry-run",
            "rows": len(rows),
            "models": len(models),
            "variant": winner.value,
            "tasks": len(rows) * len(models),
            "workers": len(args.devices),
        }
    jobs_root = output_root / "models" / "jobs"
    parts_root = output_root / "models" / "parts"
    jobs: list[tuple[Path, str]] = []
    outputs: dict[str, Path] = {}
    for index, model in enumerate(models):
        model_id = model.name
        worker_id = f"model-{index:02d}-{_safe_name(model_id)}"
        output_path = parts_root / f"{_safe_name(model_id)}.jsonl"
        job_path = jobs_root / f"{worker_id}.json"
        _write_job(
            job_path,
            {
                "schema_version": 1,
                "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                "languages": list(LANGUAGES),
                "model": str(model),
                "model_id": model_id,
                "variants": [winner.value],
                "output_path": str(output_path),
                "batch_size": int(args.batch_size),
                "dtype": str(args.dtype),
                "shard_index": 0,
                "shard_count": 1,
            },
        )
        jobs.append((job_path, worker_id))
        outputs[worker_id] = output_path
    statuses = _run_jobs(jobs, args.devices)
    successful_paths = [
        outputs[str(status["worker_id"])]
        for status in statuses
        if status["return_code"] == 0
    ]
    failed = [status for status in statuses if status["return_code"] != 0]
    if not successful_paths:
        raise RuntimeError(f"all model workers failed: {failed}")
    merged_path = output_root / "models" / "model_rows.jsonl"
    merged = merge_result_files(successful_paths, merged_path)
    successful_model_ids = sorted({str(row["model_id"]) for row in merged})
    verify_result_coverage(
        merged_path,
        _expected_keys(successful_model_ids, rows, (winner,)),
    )
    summary = write_model_reports(output_root / "models", merged)
    manifest = {
        "schema_version": 1,
        "status": "PASS" if not failed else "PASS_WITH_MODEL_FAILURES",
        "dataset_sha256": get_file_sha256(Path(args.dataset_manifest)),
        "winner": winner.value,
        "models_requested": len(models),
        "models_completed": len(successful_model_ids),
        "models_failed": len(failed),
        "failed": failed,
        "rows": summary["rows"],
        "result_sha256": get_file_sha256(merged_path),
    }
    (output_root / "models" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _consensus(args: argparse.Namespace) -> dict[str, object]:
    rows = load_classification_rows(args.dataset_manifest, LANGUAGES)
    excluded = {str(value) for value in args.exclude_model_id}
    models = [path for path in _model_paths(args) if path.name not in excluded]
    if not models:
        raise ValueError("no compatible local chat models remain after exclusions")
    if args.dry_run:
        return {
            "status": "PASS",
            "mode": "consensus-dry-run",
            "rows": len(rows),
            "models": len(models),
            "variants": len(CONSENSUS_VARIANTS),
            "tasks": len(rows) * len(models) * len(CONSENSUS_VARIANTS),
            "workers": len(args.devices),
            "system_mode": "english",
            "max_new_tokens": 8,
        }

    output_root = Path(args.output_dir).resolve() / "consensus"
    jobs_root = output_root / "jobs"
    parts_root = output_root / "parts"
    jobs: list[tuple[Path, str]] = []
    outputs: dict[str, Path] = {}
    for index, model in enumerate(models):
        model_id = model.name
        worker_id = f"consensus-{index:02d}-{_safe_name(model_id)}"
        output_path = parts_root / f"{_safe_name(model_id)}.jsonl"
        job_path = jobs_root / f"{worker_id}.json"
        _write_job(
            job_path,
            {
                "schema_version": 1,
                "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                "languages": list(LANGUAGES),
                "model": str(model),
                "model_id": model_id,
                "variants": [value.value for value in CONSENSUS_VARIANTS],
                "system_mode": "english",
                "max_new_tokens": 8,
                "output_path": str(output_path),
                "batch_size": int(args.batch_size),
                "dtype": str(args.dtype),
                "shard_index": 0,
                "shard_count": 1,
            },
        )
        jobs.append((job_path, worker_id))
        outputs[worker_id] = output_path
    statuses = _run_jobs(jobs, args.devices)
    successful_paths = [
        outputs[str(status["worker_id"])]
        for status in statuses
        if status["return_code"] == 0
    ]
    failed = [status for status in statuses if status["return_code"] != 0]
    if not successful_paths:
        raise RuntimeError(f"all consensus workers failed: {failed}")
    merged_path = output_root / "consensus_rows.jsonl"
    merged = merge_result_files(successful_paths, merged_path)
    successful_model_ids = sorted({str(row["model_id"]) for row in merged})
    verify_result_coverage(
        merged_path,
        _expected_keys(successful_model_ids, rows, CONSENSUS_VARIANTS),
    )
    write_model_reports(output_root, merged)
    write_consensus_reports(output_root, merged)
    manifest = {
        "schema_version": 1,
        "status": "PASS" if not failed else "PASS_WITH_MODEL_FAILURES",
        "dataset_sha256": get_file_sha256(Path(args.dataset_manifest)),
        "variants": [value.value for value in CONSENSUS_VARIANTS],
        "system_mode": "english",
        "max_new_tokens": 8,
        "excluded_model_ids": sorted(excluded),
        "models_requested": len(models),
        "models_completed": len(successful_model_ids),
        "models_failed": len(failed),
        "failed": failed,
        "rows": len(merged),
        "result_sha256": get_file_sha256(merged_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "pilot":
        result = _pilot(args)
    elif args.command == "run-models":
        result = _run_models(args)
    elif args.command == "consensus":
        result = _consensus(args)
    else:
        from .self_classification_worker import main as worker_main

        worker_main(
            [
                "--job",
                str(args.job),
                "--device",
                str(args.device),
                "--worker-id",
                str(args.worker_id),
            ]
        )
        return {"status": "PASS", "mode": "worker"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
