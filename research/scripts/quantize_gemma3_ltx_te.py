#!/usr/bin/env python3
"""Quantize Gemma 3 LTX text encoders to stock-ComfyUI tensor layouts."""

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

LANGUAGE_WEIGHT = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def format_blob(profile: str, groupsize: int) -> torch.Tensor:
    if profile == "int8-convrot":
        config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": groupsize,
        }
    elif profile == "nvfp4":
        config = {"format": "nvfp4"}
    else:
        raise ValueError(profile)
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode()), dtype=torch.uint8
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("int8-convrot", "nvfp4"), required=True)
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
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout
    from convert_int8_convrot import build_ratios, quantize_search

    device = torch.device(args.device)
    ratios = build_ratios(0.80)
    outputs: dict[str, torch.Tensor] = {}
    counts: Counter[str] = Counter()
    started = time.time()

    with safe_open(args.source, framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        keys = list(source.keys())
        for index, key in enumerate(keys, start=1):
            tensor = source.get_tensor(key)
            if not LANGUAGE_WEIGHT.match(key):
                outputs[key] = tensor.contiguous()
                continue
            if tensor.ndim != 2:
                raise ValueError(f"expected 2D language weight: {key} {tensor.shape}")
            weight = tensor.to(device=device, dtype=torch.float32)
            base = key.removesuffix(".weight")
            if args.profile == "int8-convrot":
                qdata, scale = quantize_search(
                    weight, args.groupsize, ratios, clip_margin=0.05
                )
                outputs[key] = qdata.cpu().contiguous()
                outputs[f"{base}.weight_scale"] = scale.cpu().contiguous()
            else:
                qdata, params = TensorCoreNVFP4Layout.quantize(
                    weight.to(dtype=torch.bfloat16)
                )
                outputs[key] = qdata.cpu().contiguous()
                outputs[f"{base}.weight_scale"] = params.block_scale.cpu().contiguous()
                outputs[f"{base}.weight_scale_2"] = params.scale.cpu().contiguous()
            outputs[f"{base}.comfy_quant"] = format_blob(args.profile, args.groupsize)
            counts[args.profile] += 1
            del tensor, weight, qdata
            if index % 25 == 0:
                torch.cuda.empty_cache()
                print(
                    json.dumps({"stage": "quantize", "done": index, "total": len(keys)})
                )

        metadata.update(
            {
                "format": "pt",
                "heretic_te_profile": args.profile,
                "heretic_te_quantizer": "quantize_gemma3_ltx_te.py",
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
                "quantized": dict(counts),
                "elapsed_s": round(time.time() - started, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
