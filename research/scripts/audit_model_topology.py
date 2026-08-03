#!/usr/bin/env python3
"""Static audit of residual-stream writers in local Hugging Face checkpoints.

The script reads only config files and safetensors headers. It never loads model
weights and never reads prompt/response text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    return {
        name: value
        for name, value in header.items()
        if name != "__metadata__"
    }


def read_tensor_headers(model_dir: Path) -> dict[str, dict[str, Any]]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        headers: dict[str, dict[str, Any]] = {}
        for shard_name in shard_names:
            headers.update(read_safetensors_header(model_dir / shard_name))
        return headers

    tensor_files = sorted(model_dir.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")
    headers = {}
    for tensor_file in tensor_files:
        headers.update(read_safetensors_header(tensor_file))
    return headers


def numel(meta: dict[str, Any]) -> int:
    return math.prod(meta["shape"])


def component_for(
    suffix: str,
    model_type: str,
) -> tuple[str, bool, str] | None:
    """Return (component, currently targeted by Heretic, intervention side)."""
    if suffix == "self_attn.o_proj.weight":
        return "attention_output", True, "output"
    if suffix == "linear_attn.out_proj.weight":
        return "linear_attention_output", True, "output"
    if suffix == "mlp.down_proj.weight":
        return "dense_mlp_output", True, "output"
    if suffix == "mlp.shared_expert.down_proj.weight":
        return "shared_expert_output", True, "output"
    if suffix == "mlp.experts.down_proj":
        return "fused_routed_expert_output", True, "output"
    if suffix in {"mlp.gate.weight", "router.weight"}:
        return "router", False, "input"
    if suffix == "mlp.shared_expert_gate.weight":
        return "shared_expert_gate", False, "nonlinear_gate"
    if suffix == "linear_attn.in_proj_z.weight":
        return "linear_attention_z_gate", False, "latent_gate"
    if suffix == "per_layer_input_gate.weight":
        return "per_layer_input_gate", False, "latent_gate"
    if suffix == "per_layer_projection.weight":
        return "per_layer_output_projection", False, "output"
    if suffix == "experts.down_proj":
        return "direct_fused_expert_output", False, "output"
    if suffix == "self_attn.q_proj.weight" and model_type.startswith("qwen3_5"):
        return "packed_full_attention_query_and_gate", False, "packed_latent_gate"
    return None


def audit_model(model_dir: Path) -> dict[str, Any]:
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = raw_config.get("text_config") or raw_config
    model_type = str(config.get("model_type", raw_config.get("model_type", "unknown")))
    layer_types = config.get("layer_types") or []
    headers = read_tensor_headers(model_dir)

    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "layers": [],
            "tensor_count": 0,
            "parameters": 0,
            "currently_targeted": False,
            "intervention_side": None,
        }
    )
    layer_tensor_counts: dict[int, int] = defaultdict(int)

    for name, meta in headers.items():
        if name.startswith("mtp.") or ".mtp." in name:
            continue
        match = LAYER_RE.search(name)
        if not match:
            continue
        layer_index = int(match.group(1))
        suffix = match.group(2)
        layer_tensor_counts[layer_index] += 1
        classification = component_for(suffix, model_type)
        if classification is None:
            continue
        component, targeted, side = classification
        record = components[component]
        record["layers"].append(layer_index)
        record["tensor_count"] += 1
        parameters = numel(meta)
        if component == "packed_full_attention_query_and_gate":
            parameters //= 2
        record["parameters"] += parameters
        record["currently_targeted"] = targeted
        record["intervention_side"] = side

    for record in components.values():
        record["layers"] = sorted(set(record["layers"]))

    mtp_headers = {
        name: meta
        for name, meta in headers.items()
        if name.startswith("mtp.") or ".mtp." in name
    }

    return {
        "model": model_dir.name,
        "model_type": model_type,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "layer_types": layer_types,
        "layer_type_counts": {
            layer_type: layer_types.count(layer_type)
            for layer_type in sorted(set(layer_types))
        },
        "num_experts": config.get("num_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "enable_moe_block": config.get("enable_moe_block"),
        "hidden_size_per_layer_input": config.get("hidden_size_per_layer_input"),
        "num_kv_shared_layers": config.get("num_kv_shared_layers"),
        "mtp_num_hidden_layers": config.get("mtp_num_hidden_layers"),
        "checkpoint_tensor_count": len(headers),
        "detected_layer_count": len(layer_tensor_counts),
        "components": dict(sorted(components.items())),
        "mtp": {
            "tensor_count": len(mtp_headers),
            "parameters": sum(numel(meta) for meta in mtp_headers.values()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = [audit_model(path.resolve()) for path in args.model_dirs]
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
