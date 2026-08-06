#!/usr/bin/env python3
"""Build stock-Comfy mixed-precision Qwen3-VL MiniMax-H3 encoders."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LANGUAGE_RE = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def parse_layer_set(value: object) -> set[int]:
    if isinstance(value, list):
        return {int(item) for item in value}
    if isinstance(value, str) and value == "0..49":
        return set(range(50))
    raise ValueError(value)


def exact_w4_map(profile: str, matrix: dict) -> set[tuple[int, str]]:
    name = {"t2-w8w4-q23": "q23", "t3-w8w4-q21": "q21", "t4-w8w4-q20": "q20"}[profile]
    data = matrix["profiles"][name]
    if name == "q20":
        base = matrix["profiles"][data["inherits"]]
        gate = parse_layer_set(base["w4"]["gate_proj_layers"])
        up = parse_layer_set(base["w4"]["up_proj_layers"])
        gate |= parse_layer_set(data["additional_w4"]["gate_proj_layers"])
        up |= parse_layer_set(data["additional_w4"]["up_proj_layers"])
    else:
        gate = parse_layer_set(data["w4"]["gate_proj_layers"])
        up = parse_layer_set(data["w4"]["up_proj_layers"])
    return {(layer, "gate_proj") for layer in gate} | {
        (layer, "up_proj") for layer in up
    }


def exact_t5_int8_map(matrix: dict) -> set[tuple[int, str]]:
    data = matrix["profiles"]["balanced16"]["int8"]
    result: set[tuple[int, str]] = set()
    for key, layers in data.items():
        family = key.removesuffix("_layers")
        result.update((int(layer), family) for layer in layers)
    expected = int(matrix["profiles"]["balanced16"]["int8_count"])
    if len(result) != expected:
        raise ValueError(
            f"balanced16 INT8 map has {len(result)} entries, expected {expected}"
        )
    return result


def format_blob(fmt: str, groupsize: int) -> torch.Tensor:
    if fmt == "int8":
        config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": groupsize,
        }
    elif fmt == "w4":
        config = {"format": "convrot_w4a4", "convrot_groupsize": groupsize}
    elif fmt == "nvfp4":
        config = {"format": "nvfp4"}
    else:
        raise ValueError(fmt)
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode()), dtype=torch.uint8
    )


def policy_for(
    key: str,
    shape: tuple[int, ...],
    profile: str,
    w4_map: set[tuple[int, str]],
    t5_int8_map: set[tuple[int, str]],
) -> str:
    match = LANGUAGE_RE.match(key)
    if match:
        layer, family = int(match.group(1)), match.group(2)
        if profile == "t1-w8-hq32":
            return "int8"
        if profile in {"t2-w8w4-q23", "t3-w8w4-q21", "t4-w8w4-q20"}:
            return "w4" if (layer, family) in w4_map else "int8"
        if profile == "t5-w4-balanced16":
            return "int8" if (layer, family) in t5_int8_map else "w4"
        if profile in {"t6-w4-language16", "t7-w4-all-edges12"}:
            return "w4"
        if profile == "t8-nvfp4-blackwell":
            return "nvfp4"
    if profile == "t7-w4-all-edges12":
        if key == "model.embed_tokens.weight":
            return "int8"
        if key.startswith("visual.") and key.endswith(".weight") and len(shape) == 2:
            return "w4"
    if profile == "t8-nvfp4-blackwell" and key == "model.embed_tokens.weight":
        return "int8"
    return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        required=True,
        choices=(
            "t1-w8-hq32",
            "t2-w8w4-q23",
            "t3-w8w4-q21",
            "t4-w8w4-q20",
            "t5-w4-balanced16",
            "t6-w4-language16",
            "t7-w4-all-edges12",
            "t8-nvfp4-blackwell",
        ),
    )
    parser.add_argument("--profile-matrix", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groupsize", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.tools_root))
    from comfy_kitchen.tensor.convrot_w4a4 import quantize_convrot_w4a4_weight
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout
    from convert_int8_convrot import build_ratios, quantize_search

    matrix = json.loads(args.profile_matrix.read_text(encoding="utf-8"))
    w4_map = (
        exact_w4_map(args.profile, matrix)
        if args.profile.startswith(("t2-", "t3-", "t4-"))
        else set()
    )
    t5_int8_map = (
        exact_t5_int8_map(matrix) if args.profile == "t5-w4-balanced16" else set()
    )
    device = torch.device(args.device)
    ratios = build_ratios(0.80)
    outputs: dict[str, torch.Tensor] = {}
    counts: Counter[str] = Counter()
    started = time.time()

    with safe_open(args.source, framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        keys = list(source.keys())
        plans = []
        compatibility_fallbacks = []
        for key in keys:
            shape = tuple(source.get_slice(key).get_shape())
            fmt = policy_for(key, shape, args.profile, w4_map, t5_int8_map)
            if fmt == "w4" and (len(shape) != 2 or shape[-1] % args.groupsize != 0):
                compatibility_fallbacks.append(
                    {
                        "key": key,
                        "shape": shape,
                        "requested": "w4",
                        "actual": "copy",
                        "reason": f"in_features_not_divisible_by_{args.groupsize}",
                    }
                )
                fmt = "copy"
            plans.append((key, shape, fmt))
        planned = Counter(item[2] for item in plans)
        print(
            json.dumps(
                {
                    "stage": "plan",
                    "profile": args.profile,
                    "source_keys": len(keys),
                    "planned": planned,
                    "compatibility_fallbacks": compatibility_fallbacks,
                }
            )
        )

        for index, (key, shape, fmt) in enumerate(plans, start=1):
            tensor = source.get_tensor(key)
            if fmt == "copy":
                outputs[key] = tensor.contiguous()
            else:
                if len(shape) != 2:
                    raise RuntimeError(f"quantized tensor must be 2D: {key} {shape}")
                weight = tensor.to(device=device, dtype=torch.float32)
                base = key.removesuffix(".weight")
                if fmt == "int8":
                    qdata, scale = quantize_search(
                        weight, args.groupsize, ratios, clip_margin=0.05
                    )
                    outputs[key] = qdata.cpu().contiguous()
                    outputs[f"{base}.weight_scale"] = scale.cpu().contiguous()
                    del qdata, scale
                elif fmt == "w4":
                    qdata, scale = quantize_convrot_w4a4_weight(
                        weight,
                        convrot_groupsize=args.groupsize,
                        quant_group_size=64,
                        stochastic_rounding=0,
                    )
                    outputs[key] = qdata.cpu().contiguous()
                    outputs[f"{base}.weight_scale"] = scale.cpu().contiguous()
                    del qdata, scale
                elif fmt == "nvfp4":
                    # comfy-kitchen's CUDA NVFP4 kernel accepts FP16/BF16 input,
                    # while the INT8/W4 search paths above intentionally operate
                    # on FP32 weights. Cast only at this backend boundary.
                    nvfp4_weight = weight.to(dtype=torch.bfloat16)
                    qdata, params = TensorCoreNVFP4Layout.quantize(nvfp4_weight)
                    outputs[key] = qdata.cpu().contiguous()
                    outputs[f"{base}.weight_scale"] = (
                        params.block_scale.cpu().contiguous()
                    )
                    outputs[f"{base}.weight_scale_2"] = params.scale.cpu().contiguous()
                    del nvfp4_weight, qdata, params
                outputs[f"{base}.comfy_quant"] = format_blob(fmt, args.groupsize)
                counts[fmt] += 1
                del weight
            if index % 25 == 0 or index == len(plans):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "stage": "quantize",
                            "profile": args.profile,
                            "done": index,
                            "total": len(plans),
                            "counts": counts,
                        }
                    )
                )

        metadata.update(
            {
                "format": "pt",
                "heretic_te_profile": args.profile,
                "heretic_te_quantizer": "quantize_qwen_h3_te.py",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, args.output, metadata=metadata)
    print(
        json.dumps(
            {
                "status": "PASS",
                "profile": args.profile,
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "source_keys": len(plans),
                "output_keys": len(outputs),
                "counts": counts,
                "elapsed_s": round(time.time() - started, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
