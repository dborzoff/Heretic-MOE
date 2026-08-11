#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_corrections(audit_path: Path) -> dict[str, str]:
    corrections: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        key = (str(row.get("case_id", "")), str(row.get("canonical_id", "")))
        if not all(key) or key in seen:
            raise ValueError(f"invalid or duplicate audit key at line {line_number}")
        seen.add(key)
        verdict = row.get("verdict")
        correction = row.get("corrected_ru_prompt")
        if verdict in {"translation_collapse", "hash_uniqueness_rephrase"}:
            if not isinstance(correction, str) or not correction.strip():
                raise ValueError(f"missing correction at line {line_number}")
            if key[1] in corrections:
                raise ValueError(f"duplicate corrected canonical_id {key[1]}")
            corrections[key[1]] = correction.strip()
        elif correction is not None:
            raise ValueError(f"unexpected correction at line {line_number}")
    if not corrections:
        raise ValueError("audit contains no corrections")
    return corrections


def _repair_jsonl(path: Path, corrections: dict[str, str]) -> list[str]:
    output: list[bytes] = []
    changed: list[str] = []
    for raw_line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(raw_line.decode("utf-8"))
        canonical_id = row.get("canonical_id")
        if (
            canonical_id in corrections
            and row.get("language") == "ru"
            and row.get("direction_class", "unsafe") == "unsafe"
        ):
            row["prompt"] = corrections[str(canonical_id)]
            output.append(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            changed.append(str(canonical_id))
        else:
            output.append(raw_line)
    if changed:
        _atomic_write(path, b"".join(output))
    return changed


def _refresh_manifest(corpus_root: Path, audit_path: Path, changed: dict[str, list[str]]) -> None:
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_records = manifest.get("files")
    if not isinstance(file_records, dict):
        raise ValueError("manifest.files must be an object")
    for name, record in file_records.items():
        path = corpus_root / name
        if not path.is_file() or not isinstance(record, dict):
            continue
        record["rows"] = sum(1 for _ in path.open("rb"))
        record["sha256"] = _sha256(path)
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    repair_record = {
        "audit": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "corrected_rows": sum(len(ids) for ids in changed.values()),
        "corrected_unique_ids": len({item for ids in changed.values() for item in ids}),
        "files": {name: len(ids) for name, ids in sorted(changed.items())},
        "status": "PASS",
    }
    repair_history = manifest.get("translation_repairs", [])
    if not isinstance(repair_history, list):
        raise ValueError("manifest.translation_repairs must be an array")
    legacy_record = manifest.pop("translation_repair", None)
    if isinstance(legacy_record, dict) and legacy_record not in repair_history:
        repair_history.append(legacy_record)
    repair_history.append(repair_record)
    manifest["translation_repairs"] = repair_history
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a validated blind translation audit.")
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    corpus_root = args.corpus_root.resolve()
    audit_path = args.audit.resolve()
    corrections = _load_corrections(audit_path)

    changed: dict[str, list[str]] = {}
    for path in sorted(corpus_root.glob("*.jsonl")):
        changed_ids = _repair_jsonl(path, corrections)
        if changed_ids:
            changed[path.name] = changed_ids

    primary = changed.get("direction_ru_unsafe.jsonl", [])
    if sorted(primary) != sorted(corrections):
        raise ValueError("primary frozen RU corpus did not contain every correction exactly once")
    if len(primary) != len(set(primary)):
        raise ValueError("duplicate corrected IDs in primary frozen RU corpus")
    _refresh_manifest(corpus_root, audit_path, changed)
    print(
        json.dumps(
            {
                "audit_sha256": _sha256(audit_path),
                "corrected_unique_ids": len(corrections),
                "files_changed": len(changed),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
