#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen multilingual Heretic splits without printing text."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest = source / "manifest_v3.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for split in ("direction", "search", "micro_search", "final_holdout"):
        split_directory = source / "texts" / split
        for label in ("safe", "unsafe"):
            inputs = sorted(split_directory.glob(f"{split}_*_{label}.jsonl"))
            if not inputs:
                raise ValueError(f"No files found for {split}/{label}")

            destination = output / f"{split}_{label}.jsonl"
            seen: set[tuple[int, str]] = set()
            row_count = 0
            with destination.open("w", encoding="utf-8", newline="\n") as target:
                for input_path in inputs:
                    with input_path.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            row = json.loads(line)
                            if row.get("label") != label:
                                raise ValueError(
                                    f"Label mismatch in {input_path}:{line_number}"
                                )
                            prompt = row.get("prompt")
                            if not isinstance(prompt, str) or not prompt.strip():
                                raise ValueError(
                                    f"Empty prompt in {input_path}:{line_number}"
                                )
                            key = (row["base_id"], row["language"])
                            if key in seen:
                                raise ValueError(
                                    f"Duplicate base_id/language in {split}/{label}: {key}"
                                )
                            seen.add(key)
                            target.write(line.rstrip("\r\n") + "\n")
                            row_count += 1

            records[destination.name] = {
                "rows": row_count,
                "sha256": file_sha256(destination),
                "source_files": len(inputs),
            }

    manifest = {
        "schema_version": 1,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "files": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "files": len(records),
                "rows": sum(record["rows"] for record in records.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
