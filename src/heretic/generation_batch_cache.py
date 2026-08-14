# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-free persistent cache for resident generation batch selection."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FORBIDDEN_PUBLIC_KEYS = {"prompt", "response", "answer", "text"}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_text_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"generation batch cache contains forbidden field {key}")
            _assert_text_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_text_free(nested)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_shape_sha256(widths: Sequence[int], *, bucket_multiple: int) -> str:
    """Hash only the bucketed prompt-width shape, never prompt text."""

    if bucket_multiple < 0:
        raise ValueError("bucket_multiple must be nonnegative")
    if not widths:
        raise ValueError("row shape requires at least one width")
    counts: dict[int, int] = {}
    for raw in widths:
        width = int(raw)
        if width <= 0:
            raise ValueError("prompt widths must be positive")
        if bucket_multiple > 0:
            width = ((width + bucket_multiple - 1) // bucket_multiple) * bucket_multiple
        counts[width] = counts.get(width, 0) + 1
    return _canonical_sha256(
        {
            "bucket_multiple": int(bucket_multiple),
            "buckets": sorted(counts.items()),
            "rows": len(widths),
        }
    )


def tokenizer_fingerprint(tokenizer: object) -> str:
    """Fingerprint the tokenizer contract without reading corpus text."""

    chat_template = getattr(tokenizer, "chat_template", None)
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
        "chat_template_sha256": (
            None if chat_template is None else _sha256_text(str(chat_template))
        ),
    }
    return _canonical_sha256(payload)


def current_gpu_identity() -> dict[str, Any]:
    """Return the physical worker GPU identity visible inside one worker."""

    import torch

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    worker_label = os.environ.get("HERETIC_WORKER_LABEL", "")
    if not torch.cuda.is_available():
        raise RuntimeError("generation batch cache requires a CUDA worker")
    device_index = int(torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    identity = {
        "available": True,
        "visible_devices": visible_devices,
        "worker_label": worker_label,
        "device_index": device_index,
        "name": str(torch.cuda.get_device_name(device_index)),
        "capability": [int(capability[0]), int(capability[1])],
        "multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    for field in ("uuid", "pci_bus_id", "pci_device_id", "pci_domain_id"):
        value = getattr(properties, field, None)
        if value is not None:
            identity[field] = str(value)
    return identity


def build_generation_batch_key(
    *,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    gpu: Mapping[str, Any],
    dtype: str,
    generation_backend: str,
    generation_compile_mode: str,
    prompt_bucket_multiple: int,
    mode: str,
    expected_rows: int,
    row_shape_sha256: str,
    max_response_length: int,
) -> dict[str, Any]:
    if not model_fingerprint.strip() or not tokenizer_fingerprint.strip():
        raise ValueError("model and tokenizer fingerprints must be non-empty")
    if len(row_shape_sha256) != 64:
        raise ValueError("row_shape_sha256 must be a SHA-256 hex digest")
    if expected_rows <= 0 or max_response_length <= 0:
        raise ValueError("expected_rows and max_response_length must be positive")
    if prompt_bucket_multiple < 0:
        raise ValueError("prompt_bucket_multiple must be nonnegative")
    if not mode.strip():
        raise ValueError("cache mode must be non-empty")
    gpu_payload = dict(gpu)
    # Current free VRAM is checked at reuse time; it is not part of identity.
    gpu_payload.pop("free_bytes", None)
    if int(gpu_payload.get("total_bytes", 0)) <= 0:
        raise ValueError("GPU identity must include total VRAM")
    key = {
        # Version 2 records are selected by measured 100-token throughput,
        # rather than by maximum VRAM fit alone.  The key bump prevents an old
        # memory-only choice from silently bypassing the speed sweep.
        "schema_version": 2,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "gpu": gpu_payload,
        "dtype": str(dtype),
        "generation_backend": str(generation_backend),
        "generation_compile_mode": str(generation_compile_mode),
        "prompt_bucket_multiple": int(prompt_bucket_multiple),
        "mode": mode,
        "expected_rows": int(expected_rows),
        "row_shape_sha256": row_shape_sha256,
        "max_response_length": int(max_response_length),
    }
    _assert_text_free(key)
    return key


def generation_batch_key_sha256(key: Mapping[str, Any]) -> str:
    _assert_text_free(key)
    return _canonical_sha256(dict(key))


@dataclass(frozen=True)
class GenerationBatchCacheContext:
    runtime_root: Path
    key: dict[str, Any]
    required_free_bytes: int
    current_free_bytes: int
    current_total_bytes: int
    maximum_batch_size: int

    def cache_path(self) -> Path:
        return (
            Path(self.runtime_root).resolve()
            / "generation_batch"
            / f"{generation_batch_key_sha256(self.key)}.json"
        )


def _validate_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    contract = payload.get("contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64:
        raise ValueError("generation batch cache contract hash is invalid")
    without = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if _canonical_sha256(without) != contract:
        raise ValueError("generation batch cache contract mismatch")
    _assert_text_free(without)
    return payload


def build_cache_record(
    key: Mapping[str, Any], tuning: Mapping[str, Any]
) -> dict[str, Any]:
    if tuning.get("status") != "PASS":
        raise ValueError("only PASS tuning results can be cached")
    validation = tuning.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "PASS":
        raise ValueError("cached batch requires a PASS real-generation validation")
    batch_size = int(tuning.get("batch_size", 0))
    if batch_size <= 0 or int(validation.get("batch_size", -1)) != batch_size:
        raise ValueError("cache batch size does not match its validation")
    baseline_free = int(validation["baseline_free_bytes"])
    min_free = int(validation["min_free_bytes"])
    working_set = int(validation["working_set_bytes"])
    if working_set < 0 or working_set != max(0, baseline_free - min_free):
        raise ValueError("validation working set is inconsistent")
    int(validation["required_free_bytes"])
    int(validation["recovered_free_bytes"])
    int(validation["peak_allocated_bytes"])
    record = {
        "schema_version": 1,
        "status": "PASS",
        "key": dict(key),
        "batch_size": batch_size,
        "validation": dict(validation),
    }
    _assert_text_free(record)
    record["contract_sha256"] = _canonical_sha256(record)
    return _validate_contract(record)


def store_generation_batch_cache(
    context: GenerationBatchCacheContext, record: Mapping[str, Any]
) -> Path:
    payload = _validate_contract(record)
    if generation_batch_key_sha256(payload["key"]) != generation_batch_key_sha256(
        context.key
    ):
        raise ValueError("generation batch cache key mismatch")
    batch_size = int(payload["batch_size"])
    if not 1 <= batch_size <= int(context.maximum_batch_size):
        raise ValueError("cached batch size is outside the current maximum")
    path = context.cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def load_generation_batch_cache(
    context: GenerationBatchCacheContext,
) -> dict[str, Any] | None:
    """Return a compatible record, or None for missing/broken/stale cache."""

    try:
        path = context.cache_path()
        if not path.is_file():
            return None
        payload = _validate_contract(json.loads(path.read_text(encoding="utf-8")))
        if payload.get("status") != "PASS":
            return None
        if generation_batch_key_sha256(payload["key"]) != generation_batch_key_sha256(
            context.key
        ):
            return None
        gpu = payload["key"].get("gpu", {})
        if int(gpu.get("total_bytes", -1)) != int(context.current_total_bytes):
            return None
        batch_size = int(payload["batch_size"])
        if not 1 <= batch_size <= int(context.maximum_batch_size):
            return None
        validation = payload["validation"]
        if validation.get("status") != "PASS":
            return None
        if int(validation.get("batch_size", -1)) != batch_size:
            return None
        baseline_free_bytes = int(validation["baseline_free_bytes"])
        min_free_bytes = int(validation["min_free_bytes"])
        working_set_bytes = int(validation["working_set_bytes"])
        if working_set_bytes != max(0, baseline_free_bytes - min_free_bytes):
            return None
        int(validation["peak_allocated_bytes"])
        projected_free = int(context.current_free_bytes) - working_set_bytes
        if projected_free < int(context.required_free_bytes):
            return None
        return payload
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
