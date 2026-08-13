# SPDX-License-Identifier: AGPL-3.0-or-later

"""Portable SRG assets used by every multilingual Heretic-MOE run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

_PROFILE_SHA256 = "3fa45586e90fe0af53a50b1c454e282d10f551b9383417098bd49988a169df28"
_PROTOTYPE_SHA256 = "a12725291235cedbd61a1d8c900feac7c5107a10c60233ad8b91776340f945b4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(name: str) -> Path:
    resource = files("heretic").joinpath("assets", "srg", name)
    with as_file(resource) as path:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved


def load_builtin_srg_contract() -> dict[str, Any]:
    """Resolve and validate the immutable cross-model SRG assets."""

    profile_path = _asset("calibration_profile.json")
    prototype_path = _asset("prototypes.jsonl")
    if _sha256(profile_path) != _PROFILE_SHA256:
        raise ValueError("built-in SRG profile hash mismatch")
    if _sha256(prototype_path) != _PROTOTYPE_SHA256:
        raise ValueError("built-in SRG prototype hash mismatch")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("schema_version") != 2
        or profile.get("status") != "PASS"
        or int(profile.get("model_count", 0)) < 2
        or not isinstance(profile.get("group_weight"), dict)
    ):
        raise ValueError("built-in SRG profile is invalid")
    return {
        "schema_version": 1,
        "status": "PASS",
        "external_only": True,
        "model_count": int(profile["model_count"]),
        "profile_path": profile_path,
        "profile_sha256": _PROFILE_SHA256,
        "prototype_path": prototype_path,
        "prototype_sha256": _PROTOTYPE_SHA256,
        "top_k": 5,
        "min_df": 2,
        "max_response_length": 100,
        "validate_prompt_alignment": False,
    }


def materialize_builtin_srg_runtime(destination_dir: str | Path) -> dict[str, Any]:
    """Copy built-in SRG assets into one immutable run directory."""

    source = load_builtin_srg_contract()
    destination = Path(destination_dir).resolve()
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        from .multilingual_runtime import resolve_srg_runtime_contract

        resolve_srg_runtime_contract(destination)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        shutil.copy2(source["profile_path"], temporary / "calibration_profile.json")
        shutil.copy2(source["prototype_path"], temporary / "prototypes.jsonl")
        manifest = {
            key: value
            for key, value in source.items()
            if key not in {"profile_path", "prototype_path"}
        }
        manifest.update(
            {
                "profile_path": str(destination / "calibration_profile.json"),
                "prototype_path": str(destination / "prototypes.jsonl"),
                "portable_package": True,
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

