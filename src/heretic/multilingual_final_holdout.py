# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent 660-row multilingual final-holdout measurement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .language_map_data import GeometryRow
from .multilingual_contract import CalibrationRow
from .srg_calibration import relative_group_summary, relative_score
from .trial_geometry_metrics import evaluate_trial_geometry
from .utils import Prompt

_FORBIDDEN_PUBLIC_KEYS = {"prompt", "response", "answer", "text"}


def _generation_contract(
    value: Mapping[str, object] | None,
) -> dict[str, str | int]:
    raw = value or {
        "backend": "dynamic_eager",
        "prompt_bucket_multiple": 0,
        "compile_mode": "default",
    }
    backend = str(raw.get("backend", "")).strip()
    compile_mode = str(raw.get("compile_mode", "")).strip()
    prompt_bucket_multiple = int(raw.get("prompt_bucket_multiple", -1))
    if (
        backend not in {"dynamic_eager", "compiled_static"}
        or not compile_mode
        or prompt_bucket_multiple < 0
    ):
        raise ValueError("generation contract is invalid")
    return {
        "backend": backend,
        "prompt_bucket_multiple": prompt_bucket_multiple,
        "compile_mode": compile_mode,
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _assert_text_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public final-holdout record contains forbidden field {key}")
            _assert_text_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_text_free(nested)


def _margins(score: object, expected: int) -> list[float]:
    diagnostics = getattr(score, "diagnostics", None)
    values = diagnostics.get("margins") if isinstance(diagnostics, Mapping) else None
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError("final-holdout SRG margins are not aligned")
    output = [float(value) for value in values]
    if any(not math.isfinite(value) for value in output):
        raise ValueError("final-holdout SRG margins are non-finite")
    return output


@contextmanager
def _generation_length(model: Any, length: int) -> Iterator[None]:
    if length <= 0:
        raise ValueError("max_response_length must be positive")
    settings = getattr(model, "settings", None)
    if settings is None or not hasattr(settings, "max_response_length"):
        yield
        return
    previous = settings.max_response_length
    settings.max_response_length = length
    try:
        yield
    finally:
        settings.max_response_length = previous


def _row_contract(rows: Sequence[CalibrationRow]) -> list[dict[str, str]]:
    return [
        {
            "base_id": row.base_id,
            "row_id": row.row_id,
            "language": row.language.lower(),
            "category_id": row.category_id,
        }
        for row in rows
    ]


def _geometry_rows(rows: Sequence[CalibrationRow]) -> list[GeometryRow]:
    return [
        GeometryRow(
            canonical_id=row.base_id,
            row_id=row.row_id,
            language=row.language.lower(),
            direction="unsafe",
            category_id=row.category_id,
            prompt=row.prompt,
            source_path=row.source_path,
            source_line=row.source_line,
        )
        for row in rows
    ]


def build_final_holdout_archive(
    *,
    model: Any,
    rows: Sequence[CalibrationRow],
    refusal_direction: Tensor,
    srg_scorer: Any,
    srg_profile: Mapping[str, object],
    output_dir: str | Path,
    dataset_contract_sha256: str,
    model_fingerprint: str,
    top_six_contract_sha256: str,
    max_response_length: int,
    generation_contract: Mapping[str, object] | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate the clean R reference once, after TOP-6 membership is frozen."""

    ordered = tuple(rows)
    if not ordered or len({row.row_id for row in ordered}) != len(ordered):
        raise ValueError("final-holdout rows must be non-empty and unique")
    direction = refusal_direction.detach().to(torch.float32).cpu().contiguous()
    if direction.ndim != 2 or not bool(torch.isfinite(direction).all()):
        raise ValueError("final-holdout refusal direction must be finite")
    contract = {
        "schema_version": 2,
        "dataset_contract_sha256": dataset_contract_sha256,
        "model_fingerprint": model_fingerprint,
        "top_six_contract_sha256": top_six_contract_sha256,
        "max_response_length": int(max_response_length),
        "generation_contract": _generation_contract(generation_contract),
        "row_contract_sha256": _canonical_sha256(_row_contract(ordered)),
        "direction_sha256": _tensor_sha256(direction),
    }
    contract_sha256 = _canonical_sha256(contract)
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.json"
    private_path = destination / "private" / "clean_records.jsonl"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "PASS"
            or manifest.get("archive_contract_sha256") != contract_sha256
            or not private_path.is_file()
            or manifest.get("private_records_sha256") != _sha256(private_path)
        ):
            raise ValueError("existing final-holdout archive contract differs")
        return manifest

    prompts = [Prompt(system="", user=row.prompt) for row in ordered]
    generation_options: dict[str, object] = {"skip_special_tokens": True}
    if progress is not None:
        generation_options["progress"] = progress
    with _generation_length(model, max_response_length):
        responses, token_ids, residuals = (
            model.get_response_artifacts_with_prefill_residuals_batched(
                prompts,
                **generation_options,
            )
        )
    if (
        len(responses) != len(ordered)
        or len(token_ids) != len(ordered)
        or residuals.shape != (len(ordered), *direction.shape)
    ):
        raise ValueError("clean final-holdout generation artifacts are not aligned")
    margins = _margins(srg_scorer.score_responses(prompts, responses), len(ordered))
    projections = torch.einsum(
        "blh,lh->bl", residuals.to(torch.float32).cpu(), direction
    )
    serialized = []
    for index, row in enumerate(ordered):
        record = {
            "base_id": row.base_id,
            "row_id": row.row_id,
            "language": row.language,
            "category_id": row.category_id,
            "prompt": row.prompt,
            "clean_response": responses[index],
            "clean_response_token_ids": [int(value) for value in token_ids[index]],
            "clean_margin": margins[index],
            "clean_prompt_residual_projection": [
                float(value) for value in projections[index]
            ],
        }
        serialized.append(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    _atomic_write(private_path, "".join(serialized).encode("utf-8"))
    groups = [(row.language, row.category_id) for row in ordered]
    baseline = relative_score(margins, margins, dict(srg_profile), groups=groups)
    manifest: dict[str, Any] = {
        **contract,
        "status": "PASS",
        "archive_contract_sha256": contract_sha256,
        "rows": len(ordered),
        "layers": int(direction.shape[0]),
        "hidden_size": int(direction.shape[1]),
        "clean_baseline_srg_gain": float(baseline["srg_gain"]),
        "clean_baseline_r_gain": float(baseline["r_gain"]),
        "private_records_sha256": _sha256(private_path),
    }
    _assert_text_free(manifest)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def load_final_holdout_archive(
    input_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(input_dir).resolve()
    manifest_path = source / "manifest.json"
    private_path = source / "private" / "clean_records.jsonl"
    if not manifest_path.is_file() or not private_path.is_file():
        raise FileNotFoundError("final-holdout archive is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_keys = [
        "schema_version",
        "dataset_contract_sha256",
        "model_fingerprint",
        "top_six_contract_sha256",
        "max_response_length",
        "row_contract_sha256",
        "direction_sha256",
    ]
    if int(manifest.get("schema_version", -1)) >= 2:
        contract_keys.append("generation_contract")
    contract = {key: manifest[key] for key in contract_keys}
    if (
        manifest.get("status") != "PASS"
        or manifest.get("archive_contract_sha256") != _canonical_sha256(contract)
        or manifest.get("private_records_sha256") != _sha256(private_path)
    ):
        raise ValueError("final-holdout archive hash mismatch")
    records = [
        json.loads(line)
        for line in private_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != int(manifest.get("rows", -1)):
        raise ValueError("final-holdout archive row count mismatch")
    return manifest, records


def merge_final_holdout_archives(
    *,
    shard_dirs: Sequence[str | Path],
    rows: Sequence[CalibrationRow],
    refusal_direction: Tensor,
    srg_profile: Mapping[str, object],
    output_dir: str | Path,
    dataset_contract_sha256: str,
    model_fingerprint: str,
    top_six_contract_sha256: str,
    max_response_length: int,
    generation_contract: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Merge contiguous GPU shards into one canonical final-holdout archive."""

    ordered = tuple(rows)
    if not ordered or not shard_dirs:
        raise ValueError("final-holdout rows and shards must be non-empty")
    direction = refusal_direction.detach().to(torch.float32).cpu().contiguous()
    expected_generation = _generation_contract(generation_contract)
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        manifest, shard_records = load_final_holdout_archive(shard_dir)
        manifests.append(manifest)
        records.extend(shard_records)
    if [record.get("row_id") for record in records] != [
        row.row_id for row in ordered
    ]:
        raise ValueError("final-holdout shards do not reconstruct canonical order")
    expected_common = {
        "dataset_contract_sha256": dataset_contract_sha256,
        "model_fingerprint": model_fingerprint,
        "top_six_contract_sha256": top_six_contract_sha256,
        "max_response_length": int(max_response_length),
        "direction_sha256": _tensor_sha256(direction),
        "generation_contract": expected_generation,
    }
    for manifest in manifests:
        if any(manifest.get(key) != value for key, value in expected_common.items()):
            raise ValueError("final-holdout shard contract differs")
    contract = {
        "schema_version": 2,
        **expected_common,
        "row_contract_sha256": _canonical_sha256(_row_contract(ordered)),
    }
    contract_sha256 = _canonical_sha256(contract)
    destination = Path(output_dir).resolve()
    manifest_path = destination / "manifest.json"
    private_path = destination / "private" / "clean_records.jsonl"
    if manifest_path.is_file():
        existing, _ = load_final_holdout_archive(destination)
        if existing.get("archive_contract_sha256") != contract_sha256:
            raise ValueError("existing final-holdout archive contract differs")
        return existing
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    _atomic_write(private_path, payload)
    margins = [float(record["clean_margin"]) for record in records]
    groups = [(row.language, row.category_id) for row in ordered]
    baseline = relative_score(margins, margins, dict(srg_profile), groups=groups)
    manifest = {
        **contract,
        "status": "PASS",
        "archive_contract_sha256": contract_sha256,
        "rows": len(records),
        "layers": int(direction.shape[0]),
        "hidden_size": int(direction.shape[1]),
        "shards": len(manifests),
        "clean_baseline_srg_gain": float(baseline["srg_gain"]),
        "clean_baseline_r_gain": float(baseline["r_gain"]),
        "private_records_sha256": _sha256(private_path),
    }
    _assert_text_free(manifest)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def evaluate_final_holdout(
    *,
    trial_number: int,
    model: Any,
    rows: Sequence[CalibrationRow],
    clean_records: Sequence[Mapping[str, object]],
    refusal_direction: Tensor,
    layer_reliability: Tensor,
    srg_scorer: Any,
    srg_profile: Mapping[str, object],
    private_records_path: str | Path,
    max_response_length: int,
) -> dict[str, Any]:
    """Remeasure a frozen finalist against the exact clean R reference."""

    ordered = tuple(rows)
    if [record.get("row_id") for record in clean_records] != [
        row.row_id for row in ordered
    ]:
        raise ValueError("final-holdout clean record order differs from rows")
    direction = refusal_direction.detach().to(torch.float32).cpu().contiguous()
    prompts = [Prompt(system="", user=row.prompt) for row in ordered]
    with _generation_length(model, max_response_length):
        responses, token_ids, residuals = (
            model.get_response_artifacts_with_prefill_residuals_batched(
                prompts,
                skip_special_tokens=True,
            )
        )
    if (
        len(responses) != len(ordered)
        or len(token_ids) != len(ordered)
        or residuals.shape != (len(ordered), *direction.shape)
    ):
        raise ValueError("candidate final-holdout generation artifacts are not aligned")
    candidate_margins = _margins(
        srg_scorer.score_responses(prompts, responses), len(ordered)
    )
    baseline_margins = [float(record["clean_margin"]) for record in clean_records]
    groups = [(row.language, row.category_id) for row in ordered]
    srg = relative_score(
        baseline_margins,
        candidate_margins,
        dict(srg_profile),
        groups=groups,
    )
    group_summary = relative_group_summary(
        baseline_margins,
        candidate_margins,
        dict(srg_profile),
        groups=groups,
    )
    clean_projection = torch.tensor(
        [record["clean_prompt_residual_projection"] for record in clean_records],
        dtype=torch.float32,
    )
    candidate_projection = torch.einsum(
        "blh,lh->bl", residuals.to(torch.float32).cpu(), direction
    )
    geometry = evaluate_trial_geometry(
        _geometry_rows(ordered),
        clean_projection,
        candidate_projection,
        layer_reliability=layer_reliability,
    )
    removal = (
        0.50 * float(srg["srg_gain"])
        + 0.25 * float(srg["r_gain"])
        + 0.25 * float(geometry["unsafe_geometry_gain"])
    )
    private_lines = []
    for index, row in enumerate(ordered):
        private_lines.append(
            json.dumps(
                {
                    "trial_number": int(trial_number),
                    "base_id": row.base_id,
                    "row_id": row.row_id,
                    "language": row.language,
                    "category_id": row.category_id,
                    "prompt": row.prompt,
                    "response": responses[index],
                    "response_token_ids": [int(value) for value in token_ids[index]],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    payload = "".join(private_lines).encode("utf-8")
    _atomic_write(Path(private_records_path).resolve(), payload)
    public: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "trial_number": int(trial_number),
        "rows": len(ordered),
        "srg_gain": float(srg["srg_gain"]),
        "r_gain": float(srg["r_gain"]),
        "unsafe_geometry_gain": float(geometry["unsafe_geometry_gain"]),
        "removal": removal,
        "groups": group_summary,
        "srg": {
            key: value
            for key, value in srg.items()
            if key != "standardized_gain"
        },
        "geometry": geometry,
        "private_records_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _assert_text_free(public)
    return public
