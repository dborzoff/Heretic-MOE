#!/usr/bin/env python3
"""Build a native ComfyUI Gemma 3 text-encoder safetensors file."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

EXPECTED_KEY_COUNT = 1065
EXPECTED_KEYSET_SHA256 = (
    "f84af71a82b49f345f3421b2dfca2c777f889460a98b83f9bcac1dce94f5fff6"
)
PREFIX_MAP = (
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("vision_tower.vision_model.", "vision_model."),
    ("multi_modal_projector.", "multi_modal_projector."),
)


def map_key(key: str) -> str:
    for source, target in PREFIX_MAP:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyset_sha256(keys: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        help="Optional immutable base tokenizer.model when export omitted it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dst.exists():
        raise FileExistsError(f"Refusing to overwrite {args.dst}")

    shard_paths = [Path(path) for path in sorted(glob.glob(str(args.src / "model-*.safetensors")))]
    if not shard_paths and (args.src / "model.safetensors").is_file():
        shard_paths = [args.src / "model.safetensors"]
    if not shard_paths:
        raise FileNotFoundError(f"No model safetensors found in {args.src}")

    started = time.time()
    state: dict[str, torch.Tensor] = {}
    for shard in shard_paths:
        for source_key, tensor in load_file(shard).items():
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"Expected BF16: {source_key} is {tensor.dtype}")
            target_key = map_key(source_key)
            if target_key in state:
                raise ValueError(f"Mapped key collision: {target_key}")
            state[target_key] = tensor
        print(f"loaded={shard.name} mapped_tensors={len(state)}", flush=True)

    mapped_keys = set(state)
    mapped_hash = keyset_sha256(mapped_keys)
    if len(mapped_keys) != EXPECTED_KEY_COUNT or mapped_hash != EXPECTED_KEYSET_SHA256:
        raise RuntimeError(
            "Gemma TE key contract mismatch: "
            f"count={len(mapped_keys)} sha256={mapped_hash}"
        )

    tokenizer = args.tokenizer_source or (args.src / "tokenizer.model")
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    state["spiece_model"] = torch.frombuffer(
        bytearray(tokenizer.read_bytes()), dtype=torch.uint8
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.dst.with_suffix(args.dst.suffix + ".tmp")
    save_file(
        state,
        temporary,
        metadata={
            "format": "pt",
            "heretic_moe_gemma3_te_keyset_sha256": mapped_hash,
        },
    )
    os.replace(temporary, args.dst)

    header = safetensors_header(args.dst)
    output_keys = set(header) - {"__metadata__"}
    if len(output_keys) != EXPECTED_KEY_COUNT + 1 or "spiece_model" not in output_keys:
        raise RuntimeError("Saved TE header failed structural validation")

    report = {
        "status": "PASS",
        "source": str(args.src),
        "source_shards": [path.name for path in shard_paths],
        "tokenizer_source": str(tokenizer),
        "tokenizer_sha256": sha256_file(tokenizer),
        "mapped_tensor_count": len(mapped_keys),
        "mapped_keyset_sha256": mapped_hash,
        "output": str(args.dst),
        "output_bytes": args.dst.stat().st_size,
        "output_sha256": sha256_file(args.dst),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
