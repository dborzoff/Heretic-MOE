import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from research.scripts.run_adaptive_search import validate_adaptive_cost_contract

BUNDLE = (
    Path(__file__).parents[1]
    / "research"
    / "server"
    / "qwen36_35b_a3b_heretic_v3"
)


def load_verify_module():
    path = BUNDLE / "verify_ready.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("qwen36_verify_ready", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen36_profile_uses_the_frozen_search_contract() -> None:
    profile = BUNDLE / "qwen36_sparse_geometry.toml"
    with profile.open("rb") as stream:
        config = tomllib.load(stream)

    validate_adaptive_cost_contract(config, source=profile)
    assert config["model"] == "Qwen/Qwen3.6-35B-A3B"
    perplexity = config["scorer"]["Perplexity"]
    assert perplexity["window"] == 512
    assert perplexity["chunks"] == 24
    assert perplexity["text"]["dataset"] == "builtin://perplexity-reference-v1"
    assert config["save_trial_responses"] is True


def test_server_shell_entrypoints_are_valid_bash() -> None:
    for name in ("prepare_and_run.sh", "run_search.sh"):
        result = subprocess.run(
            ["bash", "-n", str(BUNDLE / name)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_upload_script_can_validate_a_local_bundle_without_ssh(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is not installed")

    data = tmp_path / "data"
    data.mkdir()
    payload = data / "sample.jsonl"
    payload.write_bytes(b"opaque payload\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "name": payload.name,
                        "bytes": 15,
                        "sha256": (
                            "6be3e62a1ac8e624341686291d334ddc302b7586b36f21fa"
                            "98fbfc069af3181c"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-File",
            str(BUNDLE / "upload_data.ps1"),
            "-ValidateOnly",
            "-ManifestPath",
            str(manifest),
            "-LocalDataRoot",
            str(data),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"status":"PASS"' in result.stdout.replace(" ", "")
    assert "ssh" not in result.stdout.lower()


def test_data_bundle_validation_uses_size_and_sha256(tmp_path: Path) -> None:
    verifier = load_verify_module()
    data = tmp_path / "data"
    data.mkdir()
    payload = data / "sample.jsonl"
    payload.write_bytes(b"opaque payload\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": payload.name,
                        "bytes": 15,
                        "sha256": (
                            "6be3e62a1ac8e624341686291d334ddc302b7586b36f21fa"
                            "98fbfc069af3181c"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = verifier.verify_data_bundle(data, manifest)

    assert report == {"files": 1, "bytes": 15}


def test_model_validation_requires_every_indexed_shard(tmp_path: Path) -> None:
    verifier = load_verify_module()
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.a": "model-00001-of-00002.safetensors",
                    "model.b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")

    report = verifier.verify_model_bundle(tmp_path)

    assert report["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert report["shards"] == 2


def test_committed_data_manifest_matches_the_frozen_local_bundle() -> None:
    manifest = json.loads((BUNDLE / "data_manifest.json").read_text(encoding="utf-8"))

    assert manifest == {
        "schema_version": 1,
        "files": [
            {
                "name": "direction_safe.jsonl",
                "bytes": 180337,
                "sha256": "99e421b77df30f4c293120ef48d2872bae8f28a0d30525a506eacf4b58fed115",
            },
            {
                "name": "direction_unsafe.jsonl",
                "bytes": 194975,
                "sha256": "13f0965fea984ac306037fdddd18c8979d8accdec210eec67da487ee41f8227e",
            },
            {
                "name": "search_unsafe.jsonl",
                "bytes": 88015,
                "sha256": "755ef3a9eb81c04016b7f4df3bcd51b72605a235e4c510db42ad224358bf8629",
            },
            {
                "name": "prototypes.jsonl",
                "bytes": 3030578,
                "sha256": "a12725291235cedbd61a1d8c900feac7c5107a10c60233ad8b91776340f945b4",
            },
        ],
    }
