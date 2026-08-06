#!/usr/bin/env python3
"""Validate a MiniMax-H3 Qwen3-VL TE safetensors package without loading weights."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from safetensors import safe_open


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
EXPECTED_FORMAT_COUNTS = {
    "t1-w8-hq32": {"int8": 350},
    "t2-w8w4-q23": {"int8": 312, "w4": 38},
    "t3-w8w4-q21": {"int8": 279, "w4": 71},
    "t4-w8w4-q20": {"int8": 263, "w4": 87},
    "t5-w4-balanced16": {"int8": 81, "w4": 269},
    "t6-w4-language16": {"w4": 350},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    formats: Counter[str] = Counter()
    with safe_open(args.model, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = dict(handle.metadata() or {})
        layers = {
            int(match.group(1))
            for key in keys
            if (match := LAYER_RE.match(key)) is not None
        }
        visual_count = sum(key.startswith("visual.") for key in keys)
        deepstack_count = sum(
            key.startswith("visual.") and "deepstack" in key.lower() for key in keys
        )

        if layers != set(range(50)):
            errors.append(f"language layers mismatch: {sorted(layers)}")
        if "model.norm.weight" in keys or "lm_head.weight" in keys:
            errors.append("forbidden language tail tensor present")
        if "model.embed_tokens.weight" not in keys:
            errors.append("token embedding missing")
        if visual_count == 0 or deepstack_count == 0:
            errors.append("vision or DeepStack tensors missing")
        if metadata.get("heretic_te_profile") != args.profile:
            errors.append("profile metadata mismatch")

        quant_keys = sorted(key for key in keys if key.endswith(".comfy_quant"))
        for quant_key in quant_keys:
            base = quant_key.removesuffix(".comfy_quant")
            weight_key = f"{base}.weight"
            scale_key = f"{base}.weight_scale"
            if weight_key not in keys or scale_key not in keys:
                errors.append(f"incomplete quant tuple: {base}")
                continue
            raw = bytes(handle.get_tensor(quant_key).tolist())
            config = json.loads(raw.decode("utf-8"))
            fmt = config.get("format")
            if fmt == "int8_tensorwise":
                formats["int8"] += 1
            elif fmt == "convrot_w4a4":
                formats["w4"] += 1
            elif fmt == "nvfp4":
                formats["nvfp4"] += 1
                if f"{base}.weight_scale_2" not in keys:
                    errors.append(f"NVFP4 secondary scale missing: {base}")
            else:
                errors.append(f"unknown quant format: {fmt}")

    expected = EXPECTED_FORMAT_COUNTS.get(args.profile)
    if expected is not None and dict(formats) != expected:
        errors.append(
            f"format counts mismatch: actual={dict(formats)} expected={expected}"
        )
    if args.profile == "t7-w4-all-edges12":
        if formats["w4"] <= 350 or formats["int8"] != 1:
            errors.append(f"T7 quant coverage mismatch: {dict(formats)}")
    if args.profile == "t8-nvfp4-blackwell":
        if formats["nvfp4"] != 350 or formats["int8"] != 1:
            errors.append(f"T8 quant coverage mismatch: {dict(formats)}")

    size = args.model.stat().st_size
    if not math.isfinite(float(size)) or size <= 0:
        errors.append("invalid file size")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "model": str(args.model),
        "bytes": size,
        "profile": args.profile,
        "tensor_count": len(keys),
        "language_layers": sorted(layers),
        "visual_tensor_count": visual_count,
        "deepstack_tensor_count": deepstack_count,
        "quantized": dict(sorted(formats.items())),
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
