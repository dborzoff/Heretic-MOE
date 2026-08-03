#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate resume-safe response archives without printing corpus text."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control chat-template thinking mode when the template supports it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (intended for text-free smoke tests).",
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


def model_loader(path: Path) -> type[AutoModelForCausalLM] | type[AutoModelForImageTextToText]:
    configs = PretrainedConfig.get_config_dict(path, local_files_only=True)
    if any("vision_config" in config for config in configs):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def load_completed(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path)
    result = {int(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate ids in resume archive {path}")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)
    models = [parse_model(spec) for spec in args.model]
    prompts = read_jsonl(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts or any("prompt" not in row for row in prompts):
        raise ValueError("Prompt JSONL must contain non-empty prompt fields")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    reports: list[dict[str, object]] = []

    for label, model_path in models:
        output_path = args.output_dir / f"{label}.responses.jsonl"
        completed = load_completed(output_path)
        expected_ids = set(range(len(prompts)))
        if not set(completed).issubset(expected_ids):
            raise RuntimeError(f"Resume archive {output_path} contains out-of-range ids")

        print(f"model={label} completed={len(completed)}/{len(prompts)}", flush=True)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = model_loader(model_path).from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        model.generation_config.max_length = None
        load_elapsed = time.perf_counter() - load_started
        loaded_memory = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            loaded_memory = {
                "allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
                "reserved_mib": torch.cuda.memory_reserved(device) / (1024**2),
            }
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()

        pending = [index for index in range(len(prompts)) if index not in completed]
        with output_path.open("a", encoding="utf-8", newline="\n") as stream:
            for offset in range(0, len(pending), args.batch_size):
                ids = pending[offset : offset + args.batch_size]
                template_options: dict[str, object] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if args.thinking != "auto":
                    template_options["enable_thinking"] = args.thinking == "on"
                rendered = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": str(prompts[index]["prompt"])}],
                        **template_options,
                    )
                    for index in ids
                ]
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
                input_length = encoded["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                tails = generated[:, input_length:]
                answers = tokenizer.batch_decode(tails, skip_special_tokens=True)
                for index, answer, token_row in zip(ids, answers, tails):
                    prompt_row = prompts[index]
                    generated_tokens = int(
                        (token_row != tokenizer.pad_token_id).sum().item()
                    )
                    record = {
                        "id": index,
                        "base_id": prompt_row.get("base_id"),
                        "language": prompt_row.get("language"),
                        "subcategory": prompt_row.get("subcategory"),
                        "prompt": prompt_row["prompt"],
                        "answer": answer,
                        "generated_tokens": generated_tokens,
                        "hit_token_cap": generated_tokens >= args.max_new_tokens,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[index] = record
                done = len(completed)
                elapsed = time.perf_counter() - started
                rate = (done - (len(prompts) - len(pending))) / max(elapsed, 1e-9)
                print(
                    f"model={label} progress={done}/{len(prompts)} "
                    f"rate={rate:.3f}/s",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_elapsed = time.perf_counter() - started
        ordered = [completed[index] for index in range(len(prompts))]
        token_counts = [int(row["generated_tokens"]) for row in ordered]
        total_generated_tokens = sum(token_counts)
        generation_memory = None
        if device.type == "cuda":
            generation_memory = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                / (1024**2),
            }
        report = {
            "label": label,
            "model": str(model_path),
            "config_sha256": sha256(model_path / "config.json"),
            "responses": str(output_path.resolve()),
            "responses_sha256": sha256(output_path),
            "rows": len(ordered),
            "unique_ids": len(completed),
            "empty_answers": sum(not str(row["answer"]).strip() for row in ordered),
            "hit_token_cap": sum(bool(row["hit_token_cap"]) for row in ordered),
            "timing": {
                "load_seconds": load_elapsed,
                "generation_seconds": generation_elapsed,
                "rows_per_second": len(ordered) / max(generation_elapsed, 1e-9),
                "generated_tokens_per_second": total_generated_tokens
                / max(generation_elapsed, 1e-9),
            },
            "memory_after_load": loaded_memory,
            "memory_during_generation": generation_memory,
            "generated_tokens": {
                "min": min(token_counts),
                "median": statistics.median(token_counts),
                "max": max(token_counts),
                "sum": total_generated_tokens,
            },
        }
        reports.append(report)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "prompts": str(args.prompts.resolve()),
        "prompts_sha256": sha256(args.prompts),
        "prompt_rows": len(prompts),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "models": reports,
        "text_free_report": True,
        "status": "PASS",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": len(reports),
                "rows_per_model": len(prompts),
                "manifest": str(manifest_path.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
