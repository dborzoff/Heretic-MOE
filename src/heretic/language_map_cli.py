# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free command line interface for multilingual geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .language_map_analysis import analyze_geometry, write_geometry_reports
from .language_map_cache import capture_residual_cache, load_residual_cache
from .language_map_data import GeometryRow, LanguageFile, load_aligned_corpus


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
    files: list[LanguageFile] = []
    for language, path in args.group_a:
        files.append(LanguageFile(language, "safe", path))
    for language, path in args.group_b:
        files.append(LanguageFile(language, "unsafe", path))
    return files


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
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
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--device", default="0", help="CUDA device identifier.")
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
    return parser


def _capture(
    rows: list[GeometryRow],
    args: argparse.Namespace,
    files: list[LanguageFile],
) -> dict[str, Any]:
    if not args.model:
        raise ValueError("--model is required unless --dry-run is used")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # Device visibility must be set before the first CUDA operation. Importing
    # the heavyweight model wrapper is deliberately delayed until this point.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from .config import QuantizationMethod, Settings
    from .model import Model
    from .utils import get_file_sha256

    settings = Settings(
        model=args.model,
        dtypes=[args.dtype],
        quantization=QuantizationMethod.NONE,
        device_map="auto",
        batch_size=args.batch_size,
        residual_batch_size=args.batch_size,
        offload_outputs_to_cpu=True,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )
    model = Model(settings)
    source_files = [
        {
            "language": specification.language,
            "group": "A" if specification.direction == "safe" else "B",
            "name": Path(specification.path).name,
            "sha256": get_file_sha256(specification.path),
        }
        for specification in files
    ]

    def progress(completed: int, total: int) -> None:
        print(f"Geometry capture: {completed}/{total}", flush=True)

    return capture_residual_cache(
        model,
        rows,
        args.batch_size,
        args.output_dir / "cache",
        system_prompt=args.system_prompt,
        metadata={
            "model": args.model,
            "seed": args.seed,
            "languages": list(args.languages),
            "rows_per_cell": args.rows_per_cell,
            "source_files": source_files,
        },
        progress=progress,
    )


def _analyze(cache_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    index, residuals, manifest = load_residual_cache(cache_dir)
    report = analyze_geometry(index, residuals, seed=seed)
    write_geometry_reports(report, output_dir)
    return {
        "status": report["status"],
        "mode": "analyze",
        "rows": report["rows"],
        "layers": report["layers"],
        "hidden_size": report["hidden_size"],
        "cache_status": manifest["status"],
        "output_dir": str(output_dir.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "analyze":
        result = _analyze(args.cache_dir, args.output_dir, args.seed)
        print(json.dumps(result, sort_keys=True))
        return result

    files = _input_files(args)
    rows = load_aligned_corpus(files, args.languages, args.rows_per_cell)
    if args.dry_run:
        result = {
            "status": "PASS",
            "mode": "dry-run",
            "languages": list(args.languages),
            "rows": len(rows),
            "rows_per_cell": args.rows_per_cell,
        }
        print(json.dumps(result, sort_keys=True))
        return result

    _capture(rows, args, files)
    result = _analyze(args.output_dir / "cache", args.output_dir / "analysis", args.seed)
    result["mode"] = "run"
    print(json.dumps(result, sort_keys=True))
    return result
