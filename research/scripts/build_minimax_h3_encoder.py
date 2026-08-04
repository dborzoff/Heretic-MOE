"""Build the 50-layer MiniMax-H3 ComfyUI encoder from an HF Qwen3-VL export.

The script uses the official BF16 MiniMax-H3 encoder as a layout template and
replaces its tensor payloads in a copied file.  It never loads the whole model
into RAM.  This preserves the exact ComfyUI tensor names and checkpoint header
while allowing edited HF weights to be converted deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


LANGUAGE_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
EXPECTED_LANGUAGE_LAYERS = set(range(50))
FORBIDDEN_TARGET_KEYS = {"model.norm.weight", "lm_head.weight"}


def target_to_source_key(key: str) -> str:
    if key.startswith("visual."):
        return f"model.{key}"
    if key.startswith("model."):
        return f"model.language_model.{key.removeprefix('model.')}"
    return key


def read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return 8 + header_size, header


def load_weight_map(source_model: Path) -> dict[str, str]:
    index_path = source_model / "model.safetensors.index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return data["weight_map"]


def validate_layout(
    source_model: Path,
    template: Path,
) -> tuple[int, dict[str, list[tuple[str, str]]], int, dict[str, object]]:
    data_start, header = read_header(template)
    target_keys = sorted(key for key in header if key != "__metadata__")
    weight_map = load_weight_map(source_model)
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing: list[str] = []

    for target_key in target_keys:
        source_key = target_to_source_key(target_key)
        shard = weight_map.get(source_key)
        if shard is None:
            missing.append(source_key)
            continue
        by_shard[shard].append((target_key, source_key))

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing {len(missing)} source tensors; first: {preview}")

    for shard_name, pairs in sorted(by_shard.items()):
        shard_path = source_model / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            for target_key, source_key in pairs:
                source_slice = source.get_slice(source_key)
                target_info = header[target_key]
                if source_slice.get_shape() != target_info["shape"]:
                    raise ValueError(f"Shape mismatch for {target_key}")
                if str(source_slice.get_dtype()) != target_info["dtype"]:
                    raise ValueError(f"Dtype mismatch for {target_key}")

    return data_start, by_shard, len(target_keys), header


def validate_trim_contract(header: dict[str, object]) -> dict[str, object]:
    target_keys = sorted(key for key in header if key != "__metadata__")
    language_layers = {
        int(match.group(1))
        for key in target_keys
        if (match := LANGUAGE_LAYER_RE.match(key)) is not None
    }
    if language_layers != EXPECTED_LANGUAGE_LAYERS:
        missing = sorted(EXPECTED_LANGUAGE_LAYERS - language_layers)
        extra = sorted(language_layers - EXPECTED_LANGUAGE_LAYERS)
        raise ValueError(f"Language-layer contract mismatch: missing={missing}, extra={extra}")

    forbidden_present = sorted(FORBIDDEN_TARGET_KEYS.intersection(target_keys))
    if forbidden_present:
        raise ValueError(f"Forbidden tail tensors remain: {forbidden_present}")
    if "model.embed_tokens.weight" not in target_keys:
        raise ValueError("Token embedding is missing")

    visual_keys = [key for key in target_keys if key.startswith("visual.")]
    if not visual_keys:
        raise ValueError("Vision tower is missing")
    deepstack_keys = [key for key in visual_keys if "deepstack" in key.lower()]
    if not deepstack_keys:
        raise ValueError("DeepStack tensors are missing")

    return {
        "language_layers": sorted(language_layers),
        "language_layer_count": len(language_layers),
        "visual_tensor_count": len(visual_keys),
        "deepstack_tensor_count": len(deepstack_keys),
        "tail_removed": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_template_to_source(
    source_model: Path,
    template: Path,
    by_shard: dict[str, list[tuple[str, str]]],
    tensor_count: int,
) -> dict[str, object]:
    compared = 0
    with safe_open(template, framework="pt", device="cpu") as target:
        for shard_index, (shard_name, pairs) in enumerate(sorted(by_shard.items()), start=1):
            with safe_open(source_model / shard_name, framework="pt", device="cpu") as source:
                for target_key, source_key in pairs:
                    if not torch.equal(target.get_tensor(target_key), source.get_tensor(source_key)):
                        raise ValueError(f"Template/source mismatch for {target_key}")
                    compared += 1
            print(
                json.dumps(
                    {
                        "compare_shards": f"{shard_index}/{len(by_shard)}",
                        "tensors_compared": compared,
                    }
                ),
                flush=True,
            )
    if compared != tensor_count:
        raise ValueError(f"Compared {compared} of {tensor_count} tensors")
    return {
        "status": "EXACT_MATCH",
        "tensors_compared": compared,
        "template_sha256": sha256_file(template),
    }


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    tensor = tensor.detach().cpu().contiguous()
    return memoryview(tensor.view(torch.uint8).numpy()).cast("B")


def build_encoder(
    source_model: Path,
    template: Path,
    output: Path,
    overwrite: bool,
    verify: bool,
) -> None:
    data_start, by_shard, tensor_count, header = validate_layout(source_model, template)
    trim_contract = validate_trim_contract(header)
    print(
        json.dumps(
            {
                "status": "LAYOUT_PASS",
                "tensors": tensor_count,
                "source_shards": len(by_shard),
                "template_bytes": template.stat().st_size,
                "trim_contract": trim_contract,
            }
        ),
        flush=True,
    )

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, output)

    written = 0
    with output.open("r+b", buffering=0) as destination:
        for shard_index, (shard_name, pairs) in enumerate(sorted(by_shard.items()), start=1):
            shard_path = source_model / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as source:
                for target_key, source_key in pairs:
                    tensor = source.get_tensor(source_key)
                    start, end = header[target_key]["data_offsets"]
                    payload = tensor_bytes(tensor)
                    if len(payload) != end - start:
                        raise ValueError(f"Byte-size mismatch for {target_key}")
                    destination.seek(data_start + start)
                    destination.write(payload)
                    written += 1
            print(
                json.dumps(
                    {
                        "progress_shards": f"{shard_index}/{len(by_shard)}",
                        "tensors_written": written,
                    }
                ),
                flush=True,
            )

    if verify:
        verified = 0
        with safe_open(output, framework="pt", device="cpu") as result:
            for shard_name, pairs in sorted(by_shard.items()):
                with safe_open(source_model / shard_name, framework="pt", device="cpu") as source:
                    for target_key, source_key in pairs:
                        if not torch.equal(result.get_tensor(target_key), source.get_tensor(source_key)):
                            raise ValueError(f"Verification mismatch for {target_key}")
                        verified += 1
        if verified != tensor_count:
            raise ValueError(f"Verified {verified} of {tensor_count} tensors")

    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "bytes": output.stat().st_size,
                "tensors": tensor_count,
                "verified": verify,
                "output_sha256": sha256_file(output),
                "trim_contract": trim_contract,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--compare-template-source", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        _, by_shard, tensor_count, header = validate_layout(args.source_model, args.template)
        trim_contract = validate_trim_contract(header)
        comparison = None
        if args.compare_template_source:
            comparison = compare_template_to_source(
                args.source_model,
                args.template,
                by_shard,
                tensor_count,
            )
        print(
            json.dumps(
                {
                    "status": "COMPARE_PASS" if comparison else "DRY_RUN_PASS",
                    "tensors": tensor_count,
                    "source_shards": len(by_shard),
                    "trim_contract": trim_contract,
                    "comparison": comparison,
                }
            )
        )
        return

    build_encoder(
        source_model=args.source_model,
        template=args.template,
        output=args.output,
        overwrite=args.overwrite,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
