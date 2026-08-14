# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free command line interface for multilingual geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from .language_selection import (
    combine_language_distance_reports,
    write_language_selection_report,
)
from .pipeline_ui import PipelineUI
from .polyguard_language_dataset import materialize_polyguard_language_dataset
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
    if args.dataset_manifest is not None:
        if (
            args.corpus_root is not None
            or args.group_a
            or args.group_b
            or args.languages is not None
            or args.rows_per_cell is not None
        ):
            raise ValueError(
                "--dataset-manifest cannot be combined with legacy corpus inputs"
            )
        from .utils import get_file_sha256

        manifest_path = Path(args.dataset_manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
            raise ValueError("dataset manifest is not a verified schema-v1 artifact")
        languages = tuple(str(value).lower() for value in manifest["languages"])
        if not languages or len(set(languages)) != len(languages):
            raise ValueError("dataset manifest languages are invalid")
        directions = {
            direction: int(manifest["directions"][direction])
            for direction in ("safe", "unsafe")
        }
        if any(value <= 0 for value in directions.values()):
            raise ValueError("dataset manifest direction counts are invalid")
        files: list[LanguageFile] = []
        for entry in manifest["files"]:
            path = (manifest_path.parent / str(entry["path"])).resolve()
            if int(entry["rows"]) != directions[str(entry["direction"])]:
                raise ValueError("dataset manifest file row count drift")
            if get_file_sha256(path) != str(entry["sha256"]):
                raise ValueError("dataset manifest file hash drift")
            files.append(
                LanguageFile(
                    str(entry["language"]).lower(),
                    str(entry["direction"]).lower(),
                    path,
                )
            )
        if int(manifest["rows"]) != len(languages) * sum(directions.values()):
            raise ValueError("dataset manifest aggregate row count drift")
        args.languages = languages
        args.rows_per_cell = directions
        args.dataset_manifest_data = manifest
        return files
    if args.corpus_root is not None:
        if args.group_a or args.group_b:
            raise ValueError("--corpus-root cannot be combined with explicit groups")
        if args.languages is None or args.rows_per_cell is None:
            raise ValueError("legacy corpus input requires languages and rows-per-cell")
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
    if args.languages is None or args.rows_per_cell is None:
        raise ValueError("explicit input requires languages and rows-per-cell")
    files = []
    for language, path in args.group_a:
        files.append(LanguageFile(language, "safe", path))
    for language, path in args.group_b:
        files.append(LanguageFile(language, "unsafe", path))
    return files


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Verified manifest that discovers languages, counts, files, and hashes.",
    )
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
        type=_languages,
        help="Comma-separated language codes in canonical comparison order.",
    )
    parser.add_argument(
        "--rows-per-cell",
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

    prepare_polyguard = subparsers.add_parser(
        "prepare-polyguard",
        help="Materialize the frozen strict PolyGuard language benchmark.",
    )
    prepare_polyguard.add_argument("--source", required=True, type=Path)
    prepare_polyguard.add_argument("--output-dir", required=True, type=Path)

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
    run.add_argument("--projection-device", default="auto")
    run.add_argument("--system-prompt", default="You are a helpful assistant.")
    run.add_argument("--dry-run", action="store_true")

    analyze = subparsers.add_parser(
        "analyze", help="Analyze a previously verified residual cache."
    )
    analyze.add_argument("--cache-dir", required=True, type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--projection-device", default="auto")

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
    compare = subparsers.add_parser(
        "compare", help="Combine full-space language distances from several models."
    )
    compare.add_argument("--analysis-dir", action="append", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    compare.add_argument("--min-k", type=int, default=4)
    compare.add_argument("--max-k", type=int, default=6)
    return parser


def _capture_metadata(
    args: argparse.Namespace, files: list[LanguageFile]
) -> dict[str, object]:
    from .utils import get_file_sha256

    metadata = {
        "model": args.model,
        "seed": args.seed,
        "languages": list(args.languages),
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
    if isinstance(args.rows_per_cell, dict):
        metadata["source_rows_per_direction"] = args.rows_per_cell
        metadata["rows_per_direction"] = args.effective_rows_per_cell
    else:
        metadata["source_rows_per_cell"] = args.rows_per_cell
        metadata["rows_per_cell"] = args.effective_rows_per_cell
    return metadata


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
    exits = run_worker_processes(
        specifications,
        stage_name="Direction capture",
        total_rows=len(rows),
        next_action="Merge + analysis",
    )
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
    merge_total = stats.total_tasks + 2
    merge_ui = PipelineUI()
    merge_ui.stage(
        "Direction merge",
        total=merge_total,
        workers=("cpu",),
        description=f"Verifying and merging {stats.total_tasks} residual parts",
    )
    last_phase = [""]

    def merge_progress(completed: int, total: int, phase: str) -> None:
        if phase != last_phase[0]:
            labels = {
                "verify": "Verifying part hashes and shapes...",
                "merge": "Combining residual tensors...",
                "publish": "Writing canonical cache...",
                "complete": "Canonical cache published.",
            }
            merge_ui.note(labels.get(phase, phase))
            last_phase[0] = phase
        merge_ui.update_worker("cpu", completed=completed, total=total)

    try:
        manifest = finalize_range_cache(
            rows,
            queue,
            parts_dir,
            cache_dir,
            metadata=metadata,
            capture_fingerprint=fingerprint,
            progress=merge_progress,
        )
        merge_ui.finish_stage(
            {
                "status": manifest["status"],
                "parts": stats.total_tasks,
                "rows": len(rows),
                "next": "Direction analysis",
            }
        )
    except BaseException as error:
        merge_ui.fail_stage(error)
        raise
    finally:
        merge_ui.close()
    job_path.unlink(missing_ok=True)
    return manifest


def _geometry_findings(
    direction_manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, object]:
    bounds = [int(value) for value in direction_manifest["recommended_layer_bounds"]]
    diagnostics = list(direction_manifest["diagnostics"]["layers"])
    candidates = [
        row for row in diagnostics if bounds[0] <= int(row["layer"]) <= bounds[1]
    ]
    strongest = sorted(
        candidates,
        key=lambda row: float(row["layer_reliability"]),
        reverse=True,
    )[:3]
    stability = statistics.median(
        float(row["cross_language_stability"]) for row in candidates
    )
    peak_separation = max(float(row["separation_strength"]) for row in candidates)
    language_losses = {
        str(language): float(value["direction_loss"])
        for language, value in report["language_contributions"].items()
    }
    most_sensitive = max(language_losses, key=language_losses.__getitem__)
    selection = report["language_selection"]
    return {
        "usable_layers": f"{bounds[0]}-{bounds[1]}",
        "strongest_layers": ",".join(str(row["layer"]) for row in strongest),
        "cross_lang_stability": f"{stability * 100:.1f}%",
        "peak_separation": f"{peak_separation:.3f}",
        "largest_language_loss": (
            f"{most_sensitive} {language_losses[most_sensitive] * 100:.2f}%"
        ),
        "recommended_languages": ",".join(selection["recommended_languages"]),
        "report": "analysis/geometry_3d/report.html",
    }


def _write_base_geometry_report(
    *,
    index: list[dict[str, object]],
    residuals,
    output_dir: Path,
    seed: int,
    projection_device: str,
) -> dict[str, object]:
    """Atomically replace the derived 3D package after successful rendering."""

    if projection_device == "auto":
        import torch

        projection_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    target = output_dir / "geometry_3d"
    staging = output_dir / ".geometry_3d.next"
    backup = output_dir / ".geometry_3d.previous"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    write_projection_package(
        index=index,
        residuals=residuals,
        output_dir=staging,
        seed=seed,
        device=projection_device,
    )
    rendered = write_interactive_geometry_report(staging, staging / "report.html")
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    rendered["output"] = str((target / "report.html").resolve())
    return rendered


def _analyze(
    cache_dir: Path,
    output_dir: Path,
    seed: int,
    projection_device: str = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
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
    if progress is not None:
        progress(1, 4)
    report = analyze_geometry(index, residuals, seed=seed)
    if progress is not None:
        progress(2, 4)
    write_geometry_reports(report, output_dir)
    if progress is not None:
        progress(3, 4)
    rendered = _write_base_geometry_report(
        index=index,
        residuals=residuals,
        output_dir=output_dir,
        seed=seed,
        projection_device=projection_device,
    )
    if progress is not None:
        progress(4, 4)
    findings = _geometry_findings(direction_manifest, report)
    return {
        "status": report["status"],
        "mode": "analyze",
        "rows": report["rows"],
        "layers": report["layers"],
        "hidden_size": report["hidden_size"],
        "cache_status": manifest["status"],
        "directions_sha256": direction_manifest["package_sha256"],
        "geometry_report_sha256": rendered["sha256"],
        "output_dir": str(output_dir.resolve()),
        "findings": findings,
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
    if args.command == "prepare-polyguard":
        manifest = materialize_polyguard_language_dataset(
            args.source,
            args.output_dir,
        )
        directions = manifest["directions"]
        assert isinstance(directions, dict)
        languages = manifest["languages"]
        assert isinstance(languages, list)
        result = {
            "status": manifest["status"],
            "mode": "prepare-polyguard",
            "languages": len(languages),
            "safe": int(directions["safe"]),
            "unsafe": int(directions["unsafe"]),
            "rows": int(manifest["rows"]),
            "manifest": str((args.output_dir / "manifest.json").resolve()),
        }
        print(json.dumps(result, sort_keys=True))
        return result
    if args.command == "compare":
        reports = [
            json.loads(
                (directory / "language_distances.json").read_text(encoding="utf-8")
            )
            for directory in args.analysis_dir
        ]
        if any(
            report.get("status") != "PASS"
            or report.get("method") != "full_space_language_geometry_v1"
            for report in reports
        ):
            raise ValueError("analysis contains an incompatible language distance report")
        combined = combine_language_distance_reports(
            reports, min_k=args.min_k, max_k=args.max_k
        )
        write_language_selection_report(combined, args.output_dir)
        result = {
            "status": combined["status"],
            "mode": "compare",
            "models": combined["models"],
            "recommended_k": combined["recommended_k"],
            "recommended_languages": combined["recommended_languages"],
            "output_dir": str(args.output_dir.resolve()),
        }
        print(json.dumps(result, sort_keys=True))
        return result
    if args.command == "analyze":
        result = _analyze(
            args.cache_dir,
            args.output_dir,
            args.seed,
            projection_device=args.projection_device,
        )
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
        maximum = (
            min(args.rows_per_cell.values())
            if isinstance(args.rows_per_cell, dict)
            else args.rows_per_cell
        )
        if args.limit_per_cell <= 0 or args.limit_per_cell > maximum:
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
        args.effective_rows_per_cell = (
            {direction: args.limit_per_cell for direction in ("safe", "unsafe")}
            if isinstance(args.rows_per_cell, dict)
            else args.limit_per_cell
        )
    if args.dry_run:
        result: dict[str, Any] = {
            "status": "PASS",
            "mode": "dry-run",
            "languages": list(args.languages),
            "rows": len(rows),
        }
        if isinstance(args.rows_per_cell, dict):
            result["rows_per_direction"] = args.effective_rows_per_cell
            result["source_rows_per_direction"] = args.rows_per_cell
        else:
            result["rows_per_cell"] = args.effective_rows_per_cell
            result["source_rows_per_cell"] = args.rows_per_cell
        print(json.dumps(result, sort_keys=True))
        return result

    _capture_parallel(rows, args, files)
    analysis_ui = PipelineUI()
    analysis_ui.stage(
        "Direction analysis",
        total=4,
        workers=("analysis",),
        description="Building directions, layer diagnostics and HTML report",
    )
    try:
        result = _analyze(
            args.output_dir / "cache",
            args.output_dir / "analysis",
            args.seed,
            projection_device=args.projection_device,
            progress=lambda completed, total: analysis_ui.update_worker(
                "analysis", completed=completed, total=total
            ),
        )
        findings = result["findings"]
        analysis_ui.finish_stage(
            {
                "status": result["status"],
                "rows": result["rows"],
                "layers": result["layers"],
                "next": "Clean reference",
            }
        )
        analysis_ui.result(
            "Found",
            {
                "usable": findings["usable_layers"],
                "strongest": findings["strongest_layers"],
                "cross-lang": findings["cross_lang_stability"],
                "separation": findings["peak_separation"],
                "language loss": findings["largest_language_loss"],
                "languages": findings["recommended_languages"],
            },
        )
        analysis_ui.result("Report", {"HTML": findings["report"]})
    except BaseException as error:
        analysis_ui.fail_stage(error)
        raise
    finally:
        analysis_ui.close()
    result["mode"] = "run"
    print(json.dumps(result, sort_keys=True))
    return result
