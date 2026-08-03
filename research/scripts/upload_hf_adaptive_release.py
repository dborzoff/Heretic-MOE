#!/usr/bin/env python3
"""Create and upload the public Heretic Adaptive model repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Hugging Face token file is empty")
    if not (args.folder / "README.md").is_file():
        raise RuntimeError("Release README.md is missing")
    for variant in ("max", "balanced"):
        if not (args.folder / variant / "config.json").is_file():
            raise RuntimeError(f"Variant {variant!r} is incomplete")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    print(f"Uploading {args.repo_id}", flush=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.folder,
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )
    print(f"Upload complete: https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
