# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resume-safe private clean references for multilingual trial evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from .language_map_data import GeometryRow
from .utils import Prompt


_FORBIDDEN_PUBLIC_KEYS = {"prompt", "response", "answer", "text"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return result


def _assert_public_text_free(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public archive manifest contains forbidden field {key}")
            _assert_public_text_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public_text_free(child)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _read_private_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _existing_manifest(output_dir: Path, contract_sha256: str) -> dict[str, Any] | None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("existing clean reference manifest is not PASS")
    if manifest.get("archive_contract_sha256") != contract_sha256:
        raise ValueError("existing clean reference archive uses a different contract")
    records = output_dir / "private" / "records.jsonl"
    if not records.is_file() or _sha256(records) != manifest.get("private_records_sha256"):
        raise ValueError("existing clean reference archive hash mismatch")
    return manifest


def build_clean_reference_archive(
    *,
    model: Any,
    rows: Sequence[GeometryRow],
    refusal_direction: Tensor,
    output_dir: str | Path,
    dataset_contract_sha256: str,
    direction_sha256: str,
    model_fingerprint: str,
    max_response_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Generate each frozen trial reference once and materialize a private archive."""

    ordered = tuple(rows)
    if not ordered or len({row.row_id for row in ordered}) != len(ordered):
        raise ValueError("reference rows must be non-empty with unique row IDs")
    if refusal_direction.ndim != 2 or not bool(torch.isfinite(refusal_direction).all()):
        raise ValueError("refusal direction must be a finite [layers,hidden] tensor")
    if batch_size <= 0 or max_response_length <= 0:
        raise ValueError("batch size and response length must be positive")
    dataset_sha = _validate_sha256(dataset_contract_sha256, "dataset contract")
    direction_sha = _validate_sha256(direction_sha256, "direction")
    if not model_fingerprint.strip():
        raise ValueError("model fingerprint must be non-empty")
    contract = {
        "schema_version": 1,
        "dataset_contract_sha256": dataset_sha,
        "direction_sha256": direction_sha,
        "model_fingerprint": model_fingerprint,
        "max_response_length": max_response_length,
        "batch_size": batch_size,
        "row_id_order_sha256": _canonical_sha256([row.row_id for row in ordered]),
    }
    contract_sha = _canonical_sha256(contract)
    destination = Path(output_dir).resolve()
    existing = _existing_manifest(destination, contract_sha)
    if existing is not None:
        return existing

    parts = destination / "private" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    direction = refusal_direction.detach().to(torch.float32).cpu().contiguous()
    part_paths: list[Path] = []
    for start in range(0, len(ordered), batch_size):
        stop = min(start + batch_size, len(ordered))
        batch_rows = ordered[start:stop]
        part = parts / f"{start:08d}-{stop:08d}.jsonl"
        metadata_path = part.with_suffix(".meta.json")
        row_ids = [row.row_id for row in batch_rows]
        row_ids_sha = _canonical_sha256(row_ids)
        if part.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("archive_contract_sha256") != contract_sha
                or metadata.get("row_id_order_sha256") != row_ids_sha
                or metadata.get("sha256") != _sha256(part)
            ):
                raise ValueError(f"clean reference part contract mismatch: {part.name}")
            part_paths.append(part)
            continue

        prompts = [Prompt(system="", user=row.prompt) for row in batch_rows]
        responses, token_ids, residuals = (
            model.get_response_artifacts_with_prefill_residuals(
                prompts,
                skip_special_tokens=True,
            )
        )
        if (
            len(responses) != len(batch_rows)
            or len(token_ids) != len(batch_rows)
            or residuals.shape[0] != len(batch_rows)
            or residuals.shape[1:] != direction.shape
        ):
            raise ValueError("generation/residual direction coverage mismatch")
        projections = torch.einsum(
            "blh,lh->bl",
            residuals.to(torch.float32).cpu(),
            direction,
        )
        safe_positions = [
            index for index, row in enumerate(batch_rows) if row.direction == "safe"
        ]
        safe_nll = model.get_conditional_nll(
            [prompts[index] for index in safe_positions],
            [token_ids[index] for index in safe_positions],
        ) if safe_positions else []
        nll_by_position = dict(zip(safe_positions, safe_nll, strict=True))

        serialized = []
        for index, row in enumerate(batch_rows):
            record = {
                "canonical_id": row.canonical_id,
                "row_id": row.row_id,
                "language": row.language,
                "direction_class": row.direction,
                "category_id": row.category_id,
                "prompt": row.prompt,
                "clean_response": responses[index],
                "clean_response_token_ids": [int(value) for value in token_ids[index]],
                "clean_prompt_residual_projection": [
                    float(value) for value in projections[index]
                ],
                "clean_response_length": len(token_ids[index]),
                "clean_conditional_nll": (
                    float(nll_by_position[index]) if index in nll_by_position else None
                ),
            }
            serialized.append(
                (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
        _atomic_write(part, b"".join(serialized))
        _write_json(
            metadata_path,
            {
                "status": "PASS",
                "archive_contract_sha256": contract_sha,
                "rows": len(batch_rows),
                "row_id_order_sha256": row_ids_sha,
                "sha256": _sha256(part),
            },
        )
        part_paths.append(part)

    final_records = destination / "private" / "records.jsonl"
    _atomic_write(final_records, b"".join(path.read_bytes() for path in part_paths))
    records = _read_private_records(final_records)
    if [record["row_id"] for record in records] != [row.row_id for row in ordered]:
        raise ValueError("assembled clean reference row order mismatch")
    lengths = [int(record["clean_response_length"]) for record in records]
    safe_nll_count = sum(
        record["direction_class"] == "safe"
        and record["clean_conditional_nll"] is not None
        for record in records
    )
    manifest: dict[str, Any] = {
        **contract,
        "status": "PASS",
        "archive_contract_sha256": contract_sha,
        "rows": len(records),
        "safe_rows_with_nll": safe_nll_count,
        "layers": int(direction.shape[0]),
        "hidden_size": int(direction.shape[1]),
        "parts": len(part_paths),
        "token_length": {
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "max": max(lengths),
        },
        "private_records_sha256": _sha256(final_records),
    }
    _assert_public_text_free(manifest)
    _write_json(destination / "manifest.json", manifest)
    return manifest
