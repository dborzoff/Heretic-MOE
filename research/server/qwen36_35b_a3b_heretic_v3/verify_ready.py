#!/usr/bin/env python3
"""Verify the remote Qwen3.6 Heretic-MOE search inputs without reading texts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_bundle(data_root: Path, manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version", 1) != 1:
        raise RuntimeError(f"Unsupported data manifest: {manifest_path}")

    total_bytes = 0
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Data manifest has no files: {manifest_path}")
    for record in files:
        path = data_root / str(record["name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        expected_size = int(record["bytes"])
        if size != expected_size:
            raise RuntimeError(f"Size mismatch for {path}: {size} != {expected_size}")
        digest = sha256(path)
        expected_digest = str(record["sha256"])
        if digest != expected_digest:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        total_bytes += size
    return {"files": len(files), "bytes": total_bytes}


def verify_model_bundle(model_root: Path) -> dict[str, Any]:
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise RuntimeError(f"Model architecture is missing in {config_path}")
    architecture = str(architectures[0])
    if architecture != EXPECTED_ARCHITECTURE:
        raise RuntimeError(
            f"Unexpected model architecture: {architecture} != {EXPECTED_ARCHITECTURE}"
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Weight map is missing in {index_path}")
    shards = sorted({str(name) for name in weight_map.values()})
    missing = [name for name in shards if not (model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model shards: {missing}")
    return {
        "architecture": architecture,
        "shards": len(shards),
        "tensors": len(weight_map),
        "bytes": sum((model_root / name).stat().st_size for name in shards),
    }


def verify_runtime(expected_gpus: int, min_vram_gib: float) -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} CUDA devices, found {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        total_gib = properties.total_memory / 1024**3
        if total_gib < min_vram_gib:
            raise RuntimeError(
                f"CUDA {index} has {total_gib:.2f} GiB; {min_vram_gib:.2f} required"
            )
        devices.append(
            {"index": index, "name": properties.name, "total_vram_gib": total_gib}
        )

    packages = {}
    for name in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "optuna",
        "safetensors",
        "torch",
        "torchvision",
        "transformers",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "cuda_devices": devices,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
    }


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    bundle = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=bundle / "data_manifest.json"
    )
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--min-vram-gib", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "schema_version": 1,
        "status": "PASS",
        "git_revision": git_revision(args.repository.resolve()),
        "model": verify_model_bundle(args.model.resolve()),
        "data": verify_data_bundle(args.data.resolve(), args.manifest.resolve()),
        "runtime": verify_runtime(args.expected_gpus, args.min_vram_gib),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({"status": "PASS", "report": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
