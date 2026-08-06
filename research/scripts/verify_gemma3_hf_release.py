#!/usr/bin/env python3
"""Verify a Gemma Heretic-MOE HF release from immutable upload reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional artifact as path|bytes|sha256",
    )
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    api = HfApi(token=token)
    info = api.model_info(args.repo, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}

    expected: dict[str, dict[str, object]] = {}
    source_reports: dict[str, str] = {}
    for report_path in sorted(args.reports_dir.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        target = report["path_in_repo"]
        item = {"bytes": int(report["bytes"]), "sha256": report["sha256"]}
        previous = expected.get(target)
        if previous is not None and previous != item:
            raise ValueError(f"Conflicting reports for {target}")
        expected[target] = item
        source_reports[target] = report_path.name

    for raw in args.extra:
        target, size, digest = raw.split("|", 2)
        expected[target] = {"bytes": int(size), "sha256": digest}
        source_reports[target] = "explicit-extra"

    rows = []
    for target, wanted in sorted(expected.items()):
        sibling = remote.get(target)
        remote_size = None if sibling is None else sibling.size
        remote_sha = None
        method = "missing"
        if sibling is not None and sibling.lfs is not None:
            if isinstance(sibling.lfs, dict):
                remote_sha = sibling.lfs.get("sha256")
            else:
                remote_sha = getattr(sibling.lfs, "sha256", None)
            method = "lfs_metadata"
        elif sibling is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded = hf_hub_download(
                    repo_id=args.repo,
                    filename=target,
                    repo_type="model",
                    local_dir=temp_dir,
                    force_download=True,
                    token=token,
                )
                remote_sha = sha256(Path(downloaded))
            method = "download_sha256"

        size_ok = remote_size == wanted["bytes"]
        sha_ok = remote_sha == wanted["sha256"]
        rows.append(
            {
                "path": target,
                "expected_bytes": wanted["bytes"],
                "remote_bytes": remote_size,
                "expected_sha256": wanted["sha256"],
                "remote_sha256": remote_sha,
                "verification_method": method,
                "source_report": source_reports[target],
                "size_ok": size_ok,
                "sha256_ok": sha_ok,
                "status": "PASS" if size_ok and sha_ok else "FAIL",
            }
        )

    status = "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL"
    payload = {
        "status": status,
        "repo": args.repo,
        "repo_commit": info.sha,
        "expected_artifacts": len(rows),
        "passed_artifacts": sum(row["status"] == "PASS" for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}))
    if status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
