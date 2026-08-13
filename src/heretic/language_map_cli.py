# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free command line interface for multilingual geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .language_map_analysis import analyze_geometry, write_geometry_reports
from .language_map_cache import (
    _capture_fingerprint,
    load_residual_cache,
)
from .language_map_controller import (
    GeometryWorkerSpec,
    build_worker_command,
    run_worker_processes,
    worker_environment,
)
from .language_map_data import GeometryRow, LanguageFile, load_aligned_corpus
from .language_map_directions import (
    build_direction_map_profile,
    write_direction_map_package,
)
from .language_map_parallel import finalize_range_cache
from .language_map_projection import write_projection_package
from .language_map_report import write_interactive_geometry_report
from .language_map_trajectory import initialize_trajectory_package
from .range_work_queue import RangeWorkQueue


def _resolve_devices(args: argparse.Namespace, available):
    from .supervisor import select_devices

    specification = args.devices or args.device or "auto"
    return select_devices(
        available,
        specification,
        min_free_fraction=args.min_free_fraction,
        min_free_gib=args.min_free_gib,
        max_workers=args.max_workers,
    )


def _language_path(value: str) -> tuple[str, Path]:
    language, separator, raw_path = value.partition("=")
    if not separator or not language.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected LANG=PATH")
    return language.strip().lower(), Path(raw_path.strip())


def _languages(value: str) -> tuple[str, ...]:
    result = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("languages must be unique comma-separated codes")
    return result


def _input_files(args: argparse.Namespace) -> list[LanguageFile]:
    if args.corpus_root is not None:
        if args.group_a or args.group_b:
            raise ValueError("--corpus-root cannot be combined with explicit groups")
        root = Path(args.corpus_root)
        return [
            LanguageFile(
                language,
                direction,
                root / f"direction_{language}_{direction}_{args.split}.jsonl",
            )
            for direction in ("safe", "unsafe")
            for language in args.languages
        ]
    files: list[LanguageFile] = []
    for language, path in args.group_a:
        files.append(LanguageFile(language, "safe", path))
    for language, path in args.group_b:
        files.append(LanguageFile(language, "unsafe", path))
    return files


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="Root of the frozen aligned train layout.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Frozen corpus split to discover below --corpus-root.",
    )
    parser.add_argument(
        "--languages",
        required=True,
        type=_languages,
        help="Comma-separated language codes in canonical comparison order.",
    )
    parser.add_argument(
        "--rows-per-cell",
        required=True,
        type=int,
        help="Required row count for every language and direction group.",
    )
    parser.add_argument(
        "--group-a",
        action="append",
        default=[],
        type=_language_path,
        metavar="LANG=PATH",
        help="Input file for neutral direction group A; repeat per language.",
    )
    parser.add_argument(
        "--group-b",
        action="append",
        default=[],
        type=_language_path,
        metavar="LANG=PATH",
        help="Input file for neutral direction group B; repeat per language.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hereticMOE geometry-map",
        description="Capture and analyze a text-free multilingual geometry map.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Validate, capture once, and analyze.")
    _add_input_arguments(run)
    run.add_argument("--model", help="Hugging Face model directory or repository ID.")
    run.add_argument("--output-dir", type=Path, default=Path("geometry_map_output"))
    run.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Residual capture batch size (0 = per-worker adaptive OOM probe).",
    )
    run.add_argument(
        "--limit-per-cell",
        type=int,
        help="After full validation, capture only the first N aligned rows per cell.",
    )
    run.add_argument(
        "--device",
        help="Single CUDA device identifier (legacy shorthand for --devices).",
    )
    run.add_argument(
        "--devices",
        help="CUDA devices: auto or a comma-separated ordered list.",
    )
    run.add_argument("--max-workers", type=int)
    run.add_argument("--min-free-gib", type=float, default=4.0)
    run.add_argument("--min-free-fraction", type=float, default=0.35)
    run.add_argument("--task-rows", type=int, default=128)
    run.add_argument("--cpu-threads-per-worker", type=int, default=6)
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--system-prompt", default="You are a helpful assistant.")
    run.add_argument("--dry-run", action="store_true")

    analyze = subparsers.add_parser(
        "analyze", help="Analyze a previously verified residual cache."
    )
    analyze.add_argument("--cache-dir", required=True, type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)
    analyze.add_argument("--seed", type=int, default=42)

    project = subparsers.add_parser(
        "project",
        help="Freeze a 3D basis, import a journal, and build the offline report.",
    )
    _add_input_arguments(project)
    project.add_argument("--cache-dir", required=True, type=Path)
    project.add_argument("--journal", required=True, type=Path)
    project.add_argument("--output-dir", required=True, type=Path)
    project.add_argument("--anchor-count", type=int, default=32)
    project.add_argument("--seed", type=int, default=42)
    project.add_argument("--projection-device", default="cpu")
    project.add_argument("--system-prompt", default="You are a helpful assistant.")

    render = subparsers.add_parser(
        "render", help="Regenerate the offline HTML from a verified 3D package."
    )
    render.add_argument("--package-dir", required=True, type=Path)
    render.add_argument("--output", type=Path)
    return parser


def _capture_metadata(
    args: argparse.Namespace, files: list[LanguageFile]
) -> dict[str, object]:
    from .utils import get_file_sha256

    return {
        "model": args.model,
        "seed": args.seed,
        "languages": list(args.languages),
        "source_rows_per_cell": args.rows_per_cell,
        "rows_per_cell": args.effective_rows_per_cell,
        "split": args.split,
        "source_files": [
            {
                "language": specification.language,
                "group": "A" if specification.direction == "safe" else "B",
                "name": Path(specification.path).name,
                "sha256": get_file_sha256(specification.path),
            }
            for specification in files
        ],
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _capture_parallel(
    rows: list[GeometryRow],
    args: argparse.Namespace,
    files: list[LanguageFile],
) -> dict[str, Any]:
    if not args.model:
        raise ValueError("--model is required unless --dry-run is used")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be nonnegative")

    cache_dir = args.output_dir / "cache"
    if (cache_dir / "manifest.json").is_file():
        _, _, manifest = load_residual_cache(cache_dir)
        metadata = _capture_metadata(args, files)
        fingerprint = _capture_fingerprint(
            rows,
            args.batch_size,
            args.system_prompt,
            metadata,
        )
        if manifest.get("capture_fingerprint") != fingerprint:
            raise ValueError(
                "complete geometry cache fingerprint mismatch; use a new output dir"
            )
        print("Geometry cache already complete; verified without model reload.", flush=True)
        return manifest

    from .supervisor import detect_nvidia_gpus

    selected = _resolve_devices(args, detect_nvidia_gpus())
    print(f"Selected {len(selected)} resident GPU worker(s):", flush=True)
    for device in selected:
        print(
            f"  GPU {device.index}: {device.name} | "
            f"free {device.free_mib / 1024:.2f}/{device.total_mib / 1024:.2f} GiB | "
            f"utilization {device.utilization}%",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _capture_metadata(args, files)
    fingerprint = _capture_fingerprint(
        rows,
        args.batch_size,
        args.system_prompt,
        metadata,
    )
    queue = RangeWorkQueue(args.output_dir / "capture_queue.sqlite3")
    queue.initialize(
        row_count=len(rows),
        rows_per_task=args.task_rows,
        fingerprint=fingerprint,
    )
    parts_dir = args.output_dir / "capture_parts"
    parts_dir.mkdir(exist_ok=True)
    invalid = queue.requeue_invalid_parts(parts_dir)
    recovered = queue.recover_incomplete()
    if invalid or recovered:
        print(
            f"Geometry resume: invalid={invalid}, recovered={recovered}, "
            f"complete={queue.stats().complete_rows}/{len(rows)}",
            flush=True,
        )

    job_path = args.output_dir / "capture_job.json"
    job = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "languages": list(args.languages),
        "rows_per_cell": args.rows_per_cell,
        "limit_per_cell": args.limit_per_cell,
        "files": [
            {
                "language": specification.language,
                "direction": specification.direction,
                "path": str(Path(specification.path).resolve()),
            }
            for specification in files
        ],
        "model": args.model,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "system_prompt": args.system_prompt,
        "seed": args.seed,
        "metadata": metadata,
        "queue_path": str(queue.path),
        "parts_dir": str(parts_dir.resolve()),
        "cpu_threads": args.cpu_threads_per_worker,
    }
    _write_json_atomic(job_path, job)
    specifications = [
        GeometryWorkerSpec(
            device=device.index,
            worker_id=f"gpu-{device.index}",
            command=build_worker_command(
                job_path.resolve(), device.index, f"gpu-{device.index}"
            ),
            environment=worker_environment(
                os.environ,
                device=device.index,
                cpu_threads=args.cpu_threads_per_worker,
            ),
        )
        for device in selected
    ]
    exits = run_worker_processes(specifications)
    failed = {
        worker_id: exit_code
        for worker_id, exit_code in exits.items()
        if exit_code != 0
    }
    if failed:
        for worker_id in failed:
            queue.release_worker(worker_id)
        raise RuntimeError(f"Geometry worker failure(s): {failed}")
    stats = queue.stats()
    if stats.complete_rows != len(rows) or stats.complete != stats.total_tasks:
        raise RuntimeError(
            f"Geometry queue incomplete: {stats.complete_rows}/{len(rows)} rows"
        )
    manifest = finalize_range_cache(
        rows,
        queue,
        parts_dir,
        cache_dir,
        metadata=metadata,
        capture_fingerprint=fingerprint,
    )
    job_path.unlink(missing_ok=True)
    return manifest


def _analyze(cache_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    index, residuals, manifest = load_residual_cache(cache_dir)
    languages = tuple(
        dict.fromkeys(str(row["language"]).lower() for row in index)
    )
    profile = build_direction_map_profile(
        index,
        residuals,
        languages=languages,
    )
    direction_manifest = write_direction_map_package(profile, output_dir)
    report = analyze_geometry(index, residuals, seed=seed)
    write_geometry_reports(report, output_dir)
    return {
        "status": report["status"],
        "mode": "analyze",
        "rows": report["rows"],
        "layers": report["layers"],
        "hidden_size": report["hidden_size"],
        "cache_status": manifest["status"],
        "directions_sha256": direction_manifest["package_sha256"],
        "output_dir": str(output_dir.resolve()),
    }


def _write_private_anchors(
    output_dir: Path,
    rows: list[GeometryRow],
    anchor_rows: list[int],
    system_prompt: str,
) -> Path:
    private_dir = output_dir / "private"
    private_dir.mkdir(exist_ok=True)
    path = private_dir / "anchors.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                {
                    "base_index": position,
                    "canonical_id": rows[position].canonical_id,
                    "row_id": rows[position].row_id,
                    "language": rows[position].language,
                    "group": "A" if rows[position].direction == "safe" else "B",
                    "category_id": rows[position].category_id,
                    "system": system_prompt,
                    "prompt": rows[position].prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for position in anchor_rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _project(args: argparse.Namespace) -> dict[str, Any]:
    if args.anchor_count <= 0:
        raise ValueError("--anchor-count must be positive")
    files = _input_files(args)
    rows = load_aligned_corpus(files, args.languages, args.rows_per_cell)
    index, residuals, _ = load_residual_cache(args.cache_dir)
    expected_row_ids = [row.row_id for row in rows]
    cached_row_ids = [str(row["row_id"]) for row in index]
    if cached_row_ids != expected_row_ids:
        raise ValueError("corpus and residual cache row order differ")
    write_projection_package(
        index=index,
        residuals=residuals,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.projection_device,
    )
    trajectory = initialize_trajectory_package(
        package_dir=args.output_dir,
        journal=args.journal,
        reference_residuals=residuals,
        anchor_count=args.anchor_count,
        seed=args.seed,
    )
    _write_private_anchors(
        args.output_dir,
        rows,
        [int(value) for value in trajectory["anchor_rows"]],
        args.system_prompt,
    )
    rendered = write_interactive_geometry_report(args.output_dir)
    return {
        "status": "PASS",
        "mode": "project",
        "base_rows": len(rows),
        "anchor_count": int(trajectory["anchor_count"]),
        "journal_trials": int(trajectory["trials"]),
        "captured_trials": int(rendered["captured_trials"]),
        "output_dir": str(args.output_dir.resolve()),
        "report_sha256": rendered["sha256"],
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "analyze":
        result = _analyze(args.cache_dir, args.output_dir, args.seed)
        print(json.dumps(result, sort_keys=True))
        return result
    if args.command == "render":
        result = write_interactive_geometry_report(
            args.package_dir,
            args.output or args.package_dir / "report.html",
        )
        result["mode"] = "render"
        print(json.dumps(result, sort_keys=True))
        return result
    if args.command == "project":
        result = _project(args)
        print(json.dumps(result, sort_keys=True))
        return result

    if args.device is not None and args.devices is not None:
        raise ValueError("--device cannot be combined with --devices")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.task_rows <= 0:
        raise ValueError("--task-rows must be positive")
    if args.cpu_threads_per_worker <= 0:
        raise ValueError("--cpu-threads-per-worker must be positive")

    files = _input_files(args)
    rows = load_aligned_corpus(files, args.languages, args.rows_per_cell)
    args.effective_rows_per_cell = args.rows_per_cell
    if args.limit_per_cell is not None:
        if args.limit_per_cell <= 0 or args.limit_per_cell > args.rows_per_cell:
            raise ValueError("--limit-per-cell must be between 1 and --rows-per-cell")
        selected: list[GeometryRow] = []
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row.direction, row.language)
            count = counts.get(key, 0)
            if count < args.limit_per_cell:
                selected.append(row)
                counts[key] = count + 1
        rows = selected
        args.effective_rows_per_cell = args.limit_per_cell
    if args.dry_run:
        result = {
            "status": "PASS",
            "mode": "dry-run",
            "languages": list(args.languages),
            "rows": len(rows),
            "rows_per_cell": args.effective_rows_per_cell,
            "source_rows_per_cell": args.rows_per_cell,
        }
        print(json.dumps(result, sort_keys=True))
        return result

    _capture_parallel(rows, args, files)
    result = _analyze(args.output_dir / "cache", args.output_dir / "analysis", args.seed)
    result["mode"] = "run"
    print(json.dumps(result, sort_keys=True))
    return result
