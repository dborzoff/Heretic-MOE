#!/usr/bin/env python3
"""Validate Gemma 3 ComfyUI TE safetensors structure without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--format", choices=("bf16", "int8-convrot", "nvfp4"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.file.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    metadata = header.pop("__metadata__", {})
    file_data_bytes = args.file.stat().st_size - 8 - header_size
    previous_end = 0
    for name, tensor in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        start, end = tensor["data_offsets"]
        if start < previous_end or end < start or end > file_data_bytes:
            raise RuntimeError(f"Invalid safetensors offsets at {name}")
        previous_end = end
    if "spiece_model" not in header or header["spiece_model"]["dtype"] != "U8":
        raise RuntimeError("Embedded SentencePiece model is missing")

    dtype_counts = Counter(tensor["dtype"] for tensor in header.values())
    if args.format == "bf16":
        if len(header) != 1066 or dtype_counts != Counter({"BF16": 1065, "U8": 1}):
            raise RuntimeError(f"Unexpected BF16 tensor contract: {dtype_counts}")
    elif args.format == "int8-convrot":
        if dtype_counts["I8"] == 0 or len(header) <= 1066:
            raise RuntimeError(f"INT8 tensors or quantization auxiliaries missing: {dtype_counts}")
    elif dtype_counts["U8"] <= 1 or len(header) <= 1066:
        raise RuntimeError(f"NVFP4 packed tensors or auxiliaries missing: {dtype_counts}")

    report = {
        "status": "PASS",
        "format": args.format,
        "file": str(args.file),
        "bytes": args.file.stat().st_size,
        "sha256": sha256_file(args.file),
        "tensor_count": len(header),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "metadata_keys": sorted(metadata),
        "embedded_sentencepiece": True,
        "offsets_valid": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
