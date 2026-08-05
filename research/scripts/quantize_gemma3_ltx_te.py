#!/usr/bin/env python3
"""Run reproducible ComfyUI-native Gemma 3 TE quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("int8-convrot", "nvfp4"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    common = [
        str(args.tool),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--comfy-quant",
        "--simple",
        "--heur",
        "--device",
        args.device,
        "--save-quant-metadata",
        "--low-memory",
        "--verbose",
        "NORMAL",
        "--manual-seed",
        "20270806",
    ]
    if args.format == "int8-convrot":
        command = common + ["--int8", "--convrot", "--convrot-group-size", "256"]
    else:
        command = common + ["--nvfp4"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    subprocess.run(command, check=True)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Quantizer did not create {args.output}")
    report = {
        "status": "PASS",
        "format": args.format,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "command": command,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
