#!/usr/bin/env python3
"""Upload one release artifact and verify its remote LFS digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remote_metadata(
    api: HfApi, repo: str, target: str
) -> tuple[int | None, str | None]:
    info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename != target:
            continue
        lfs = sibling.lfs
        if lfs is None:
            remote_sha = None
        elif isinstance(lfs, dict):
            remote_sha = lfs.get("sha256")
        else:
            remote_sha = getattr(lfs, "sha256", None)
        return sibling.size, remote_sha
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(args.file)
    local_size = args.file.stat().st_size
    local_sha = sha256(args.file)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)

    remote_size, remote_sha = remote_metadata(api, args.repo, args.path_in_repo)
    action = "skip-existing"
    if remote_size != local_size or remote_sha != local_sha:
        action = "upload"
        api.upload_file(
            repo_id=args.repo,
            repo_type="model",
            path_or_fileobj=args.file,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
        )
        for _ in range(30):
            remote_size, remote_sha = remote_metadata(api, args.repo, args.path_in_repo)
            if remote_size == local_size and remote_sha == local_sha:
                break
            time.sleep(10)

    status = "PASS" if remote_size == local_size and remote_sha == local_sha else "FAIL"
    payload = {
        "status": status,
        "action": action,
        "repo": args.repo,
        "path_in_repo": args.path_in_repo,
        "local_path": str(args.file),
        "bytes": local_size,
        "sha256": local_sha,
        "remote_bytes": remote_size,
        "remote_sha256": remote_sha,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    if status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
