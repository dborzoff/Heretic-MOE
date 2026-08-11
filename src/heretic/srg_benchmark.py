# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free clean-model benchmarking for sparse refusal geometry."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import queue
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .scorer import Score


_SENSITIVE_KEYS = {"prompt", "response", "answer", "text"}


def _model_id(path: Path) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", path.name.lower()).strip("-")
    return value or "model"


def build_jobs(
    model_paths: Sequence[str | Path], result_directory: str | Path
) -> list[dict[str, object]]:
    """Build deterministic per-model result jobs without opening model files."""

    models = [Path(path).resolve() for path in model_paths]
    normalized = [str(path).casefold() for path in models]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate model path")
    missing = [str(path) for path in models if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"model directories not found: {missing}")

    destination = Path(result_directory).resolve()
    jobs: list[dict[str, object]] = []
    for model_index, model_path in enumerate(models):
        model_id = _model_id(model_path)
        jobs.append(
            {
                "model_index": model_index,
                "model_id": model_id,
                "model_path": str(model_path),
                "result_path": str(
                    destination / f"{model_index:03d}-{model_id}.json"
                ),
            }
        )
    return jobs


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate and summarize numeric model results from one frozen SRG run."""

    if not results:
        raise ValueError("no SRG benchmark results")
    ordered = sorted(results, key=lambda row: int(row["model_index"]))
    indices = [int(row["model_index"]) for row in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError("model result indices are incomplete or duplicated")
    if any(row.get("status") != "PASS" for row in ordered):
        raise ValueError("one or more model results failed")

    prompt_hashes = {str(row["prompt_sha256"]) for row in ordered}
    if len(prompt_hashes) != 1:
        raise ValueError("model results use different prompt SHA-256 values")
    prototype_hashes = {str(row["prototype_sha256"]) for row in ordered}
    if len(prototype_hashes) != 1:
        raise ValueError("model results use different prototype SHA-256 values")
    row_counts = {int(row["rows"]) for row in ordered}
    if len(row_counts) != 1:
        raise ValueError("model results use different row counts")

    margins = [float(row["mean_margin"]) for row in ordered]
    rates = [float(row["positive_rate"]) for row in ordered]
    return {
        "status": "PASS",
        "models": len(ordered),
        "rows_per_model": next(iter(row_counts)),
        "prototype_sha256": next(iter(prototype_hashes)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "mean_margin_range": [min(margins), max(margins)],
        "positive_rate_range": [min(rates), max(rates)],
        "results": ordered,
    }


def _validate_text_free(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise ValueError(f"sensitive field is not allowed in report: {key}")
            _validate_text_free(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_text_free(nested)


def result_payload(
    job: dict[str, object],
    score: "Score",
    *,
    prototype_sha256: str,
    prompt_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Convert one SRG score into a text-free, cross-model comparable record."""

    diagnostics = dict(score.diagnostics or {})
    _validate_text_free(diagnostics)
    required = {"rows", "mean_margin", "positive_rate", "empty_indices"}
    missing = required - diagnostics.keys()
    if missing:
        raise ValueError(f"SRG diagnostics are missing required fields: {sorted(missing)}")
    return {
        "status": "PASS",
        "model_index": int(job["model_index"]),
        "model_id": str(job["model_id"]),
        "model_path": str(job["model_path"]),
        "rows": int(diagnostics["rows"]),
        "mean_margin": float(diagnostics["mean_margin"]),
        "positive_rate": float(diagnostics["positive_rate"]),
        "empty_count": len(diagnostics["empty_indices"]),
        "prototype_sha256": prototype_sha256,
        "prompt_sha256": prompt_sha256,
        "elapsed_seconds": float(elapsed_seconds),
        "diagnostics": diagnostics,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _devices(value: str) -> list[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("devices must be a non-empty unique CSV list")
    return devices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hereticMOE srg-benchmark",
        description="Benchmark clean models on one pinned sparse-geometry set.",
    )
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prototypes", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--devices", default="0", type=_devices)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    prototypes = args.prototypes.resolve()
    prompts = args.prompts.resolve()
    if not prototypes.is_file():
        raise FileNotFoundError(f"prototype bank not found: {prototypes}")
    if not prompts.is_file():
        raise FileNotFoundError(f"evaluation prompt set not found: {prompts}")
    if args.max_response_length < 1:
        raise ValueError("max response length must be positive")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    output = args.output.resolve()
    jobs = build_jobs(args.models, output / "results")
    return {
        "schema_version": 1,
        "status": "READY",
        "model_count": len(jobs),
        "devices": list(args.devices),
        "prototype_path": str(prototypes),
        "prototype_sha256": _sha256(prototypes),
        "prompt_path": str(prompts),
        "prompt_sha256": _sha256(prompts),
        "dtype": args.dtype,
        "max_response_length": args.max_response_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "top_k": args.top_k,
        "min_df": args.min_df,
        "jobs": jobs,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _score_model(job: dict[str, object], manifest: dict[str, Any]) -> dict[str, Any]:
    """Load one clean model and evaluate it on the pinned SRG contract."""

    import torch

    from .config import DatasetSpecification, QuantizationMethod, Settings
    from .model import Model
    from .plugin import Context
    from .scorers.sparse_refusal_geometry import (
        Settings as SparseGeometrySettings,
    )
    from .scorers.sparse_refusal_geometry import SparseRefusalGeometry

    started = time.perf_counter()
    settings = Settings(
        model=str(job["model_path"]),
        dtypes=[str(manifest["dtype"])],
        quantization=QuantizationMethod.NONE,
        device_map="auto",
        batch_size=int(manifest["batch_size"]),
        max_response_length=int(manifest["max_response_length"]),
        offload_outputs_to_cpu=True,
        seed=int(manifest["seed"]),
        save_trial_responses=False,
    )
    model = Model(settings)
    scorer_settings = SparseGeometrySettings(
        prototypes=str(manifest["prototype_path"]),
        prototypes_sha256=str(manifest["prototype_sha256"]),
        prompts=DatasetSpecification(
            dataset=str(manifest["prompt_path"]),
            column="prompt",
        ),
        top_k=int(manifest["top_k"]),
        min_df=int(manifest["min_df"]),
        validate_prompt_alignment=True,
    )
    scorer = SparseRefusalGeometry(
        heretic_settings=settings,
        settings=scorer_settings,
    )
    try:
        scorer.init(Context(settings=settings, model=model))
        responses: list[str] = []
        total = len(scorer.prompts)
        for start in range(0, total, settings.batch_size):
            batch = scorer.prompts[start : start + settings.batch_size]
            responses.extend(
                model.get_responses(batch, skip_special_tokens=True)
            )
            print(
                f"GPU {os.environ.get('HERETIC_SRG_DEVICE', '?')} | "
                f"{job['model_id']} | generated {len(responses)}/{total}",
                flush=True,
            )
        score = scorer.score_responses(scorer.prompts, responses)
        return result_payload(
            job,
            score,
            prototype_sha256=str(manifest["prototype_sha256"]),
            prompt_sha256=str(manifest["prompt_sha256"]),
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        del scorer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _worker_loop(
    device: str,
    jobs: Any,
    statuses: Any,
    manifest: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["HERETIC_SRG_DEVICE"] = device
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        model_id = str(job["model_id"])
        model_index = int(job["model_index"])
        print(
            f"GPU {device} | SRG model {model_index + 1}/{manifest['model_count']} "
            f"| {model_id} | loading",
            flush=True,
        )
        try:
            result = _score_model(job, manifest)
            _write_json(Path(str(job["result_path"])), result)
            print(
                f"GPU {device} | SRG model {model_index + 1}/{manifest['model_count']} "
                f"| {model_id} | mean {result['mean_margin']:+.5f} | "
                f"R-side {float(result['positive_rate']) * 100:.1f}% | PASS",
                flush=True,
            )
            statuses.put({"model_index": model_index, "status": "PASS"})
        except Exception as error:  # noqa: BLE001 - worker boundary records failures
            failure = {
                "status": "FAIL",
                "model_index": model_index,
                "model_id": model_id,
                "model_path": str(job["model_path"]),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(Path(str(job["result_path"])), failure)
            print(
                f"GPU {device} | SRG model {model_index + 1}/{manifest['model_count']} "
                f"| {model_id} | FAIL {type(error).__name__}: {error}",
                flush=True,
            )
            statuses.put({"model_index": model_index, "status": "FAIL"})


def _run_benchmark(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    jobs = context.Queue()
    statuses = context.Queue()
    for job in manifest["jobs"]:
        jobs.put(job)
    processes = [
        context.Process(
            target=_worker_loop,
            args=(device, jobs, statuses, manifest),
            name=f"srg-gpu-{device}",
        )
        for device in manifest["devices"][: min(len(manifest["jobs"]), len(manifest["devices"]))]
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    received = []
    while True:
        try:
            received.append(statuses.get_nowait())
        except queue.Empty:
            break
    failed = [row for row in received if row["status"] != "PASS"]
    missing = [
        job
        for job in manifest["jobs"]
        if not Path(str(job["result_path"])).is_file()
    ]
    if failed or missing or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(
            "SRG benchmark did not complete: "
            f"failed={len(failed)}, missing={len(missing)}; inspect result files"
        )
    results = [
        json.loads(Path(str(job["result_path"])).read_text(encoding="utf-8"))
        for job in manifest["jobs"]
    ]
    report = summarize_results(results)
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    manifest = _prepare_manifest(args)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "mode": "dry-run", "models": len(manifest["jobs"])}, sort_keys=True))
        return manifest
    report = _run_benchmark(manifest, args.output)
    manifest["status"] = "PASS"
    _write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "PASS", "mode": "run", "models": report["models"]}, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
