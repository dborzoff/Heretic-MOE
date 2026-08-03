#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare exact perplexity for local models on one frozen token stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def model_loader(path: Path) -> type[AutoModelForCausalLM] | type[AutoModelForImageTextToText]:
    configs = PretrainedConfig.get_config_dict(path, local_files_only=True)
    if any("vision_config" in config for config in configs):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=400)
    parser.add_argument(
        "--wikitext-arrow",
        type=Path,
        required=True,
        help="Frozen Hugging Face Arrow file containing a text column.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model(spec: str) -> tuple[str, Path]:
    label, separator, path_text = spec.partition("=")
    if not separator or not label or not path_text:
        raise ValueError(f"Invalid --model {spec!r}; expected LABEL=PATH")
    path = Path(path_text).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return label, path


def load_frozen_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = Dataset.from_file(str(path))
    if "text" not in dataset.column_names:
        raise ValueError(f"Frozen dataset has no text column: {dataset.column_names}")
    return "\n\n".join(text for text in dataset["text"] if text.strip())


def score_model(
    model_path: Path,
    windows: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, int]:
    model = model_loader(model_path).from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    total_nll = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for index, window in enumerate(windows, start=1):
            ids = window.unsqueeze(0).to(device)
            output = model(ids, labels=ids, use_cache=False)
            count = ids.shape[1] - 1
            total_nll += float(output.loss) * count
            target_tokens += count
            if index % 25 == 0 or index == len(windows):
                print(f"windows={index}/{len(windows)}", flush=True)
    perplexity = math.exp(total_nll / max(target_tokens, 1))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return perplexity, target_tokens


def main() -> None:
    args = parse_args()
    if args.window <= 1 or args.chunks <= 0:
        raise ValueError("Window must exceed 1 and chunks must be positive")
    models = [parse_model(spec) for spec in args.model]
    if len({label for label, _ in models}) != len(models):
        raise ValueError("Model labels must be unique")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    text = load_frozen_text(args.wikitext_arrow)
    tokenizer = AutoTokenizer.from_pretrained(models[0][1], local_files_only=True)
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    available = len(token_ids) // args.window
    if available < args.chunks:
        raise RuntimeError(f"Only {available} complete windows available, need {args.chunks}")
    windows = [
        token_ids[index * args.window : (index + 1) * args.window]
        for index in range(args.chunks)
    ]

    results: list[dict[str, object]] = []
    baseline: float | None = None
    for label, model_path in models:
        print(f"model={label} windows={len(windows)}", flush=True)
        perplexity, target_tokens = score_model(
            model_path,
            windows,
            device=device,
            dtype=dtype,
        )
        if baseline is None:
            baseline = perplexity
        result = {
            "label": label,
            "model": str(model_path),
            "perplexity": perplexity,
            "target_tokens": target_tokens,
            "relative_to_baseline": perplexity / baseline - 1.0,
        }
        results.append(result)
        print(
            f"result={label} perplexity={perplexity:.6f} "
            f"relative={result['relative_to_baseline']:.8f}",
            flush=True,
        )

    report = {
        "schema_version": 2,
        "device": str(device),
        "dtype": args.dtype,
        "window": args.window,
        "chunks": args.chunks,
        "dataset": str(args.wikitext_arrow.resolve()),
        "dataset_sha256": sha256(args.wikitext_arrow),
        "models": results,
        "text_free": True,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
