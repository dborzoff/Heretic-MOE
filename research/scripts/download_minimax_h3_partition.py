"""Download one MiniMax-H3 modular partition without duplicating Qwen weights."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


COMMON_PATTERNS = [
    "README.md",
    "model_index.json",
    "assets/**",
    "scripts/readme/**",
    "processor/**",
    "tokenizer/**",
    "scheduler/**",
    "audio_scheduler/**",
    "vae/**",
    "audio_vae/**",
    "text_encoder/*.json",
    "text_encoder/*.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--repo", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--token-file", type=Path)
    return parser.parse_args()


def read_token(path: Path | None) -> str | None:
    if path is not None:
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    return os.environ.get("HF_TOKEN") or None


def main() -> None:
    args = parse_args()
    component = "transformer" if args.partition == "fl2va" else "transformer_ref"
    patterns = [*COMMON_PATTERNS, f"{component}/**"]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"MiniMax-H3 partition download: {args.partition}", flush=True)
    print(f"Generator component: {component}", flush=True)
    print("Qwen text-encoder weights: intentionally excluded", flush=True)
    result = snapshot_download(
        repo_id=args.repo,
        local_dir=args.output,
        allow_patterns=patterns,
        token=read_token(args.token_file),
    )
    print(f"Download complete: {result}", flush=True)


if __name__ == "__main__":
    main()
