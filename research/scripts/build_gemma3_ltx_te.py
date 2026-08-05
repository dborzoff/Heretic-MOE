#!/usr/bin/env python3
"""Build a stock-ComfyUI Gemma 3 text-encoder safetensors file."""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PREFIX_MAP = (
    ("model.language_model.", "model."),
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("model.vision_tower.vision_model.", "vision_model."),
    ("vision_tower.vision_model.", "vision_model."),
    ("model.multi_modal_projector.", "multi_modal_projector."),
    ("multi_modal_projector.", "multi_modal_projector."),
)


def map_key(key: str) -> str:
    for source, target in PREFIX_MAP:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def tensor_keys(path: Path) -> set[str]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)
    return set(header)


def shard_paths(source: Path) -> list[Path]:
    index = source / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [source / name for name in sorted(set(weight_map.values()))]
    single = source / "model.safetensors"
    if single.exists():
        return [single]
    return [
        Path(item) for item in sorted(glob.glob(str(source / "model-*.safetensors")))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    shards = shard_paths(args.source)
    if not shards or any(not path.is_file() for path in shards):
        raise FileNotFoundError(f"Gemma weights not found under {args.source}")

    started = time.time()
    output: dict[str, torch.Tensor] = {}
    for path in shards:
        for key, tensor in load_file(path, device="cpu").items():
            mapped = map_key(key)
            if mapped in output:
                raise ValueError(f"duplicate mapped tensor: {mapped}")
            output[mapped] = tensor
        print(json.dumps({"stage": "load", "shard": path.name, "keys": len(output)}))

    tokenizer = args.source / "tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    output["spiece_model"] = torch.frombuffer(
        bytearray(tokenizer.read_bytes()), dtype=torch.uint8
    )

    if args.reference:
        expected = tensor_keys(args.reference) - {"spiece_model"}
        actual = set(output) - {"spiece_model"}
        if actual != expected:
            raise ValueError(
                f"reference mismatch: missing={len(expected - actual)} extra={len(actual - expected)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    save_file(output, temporary, metadata={"format": "pt", "model": "gemma3_ltx_te"})
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "tensor_count": len(output),
                "elapsed_s": round(time.time() - started, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
