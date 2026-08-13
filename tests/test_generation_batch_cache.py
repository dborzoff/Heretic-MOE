# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

from heretic.generation_batch_cache import (
    GenerationBatchCacheContext,
    build_cache_record,
    build_generation_batch_key,
    generation_batch_key_sha256,
    load_generation_batch_cache,
    row_shape_sha256,
    store_generation_batch_cache,
    tokenizer_fingerprint,
)


def _gpu(**overrides):
    gpu = {
        "available": True,
        "visible_devices": "3",
        "worker_label": "GPU 3",
        "device_index": 0,
        "name": "Test GPU",
        "capability": [9, 0],
        "uuid": "GPU-uuid-3",
        "pci_bus_id": "0000:03:00.0",
        "total_bytes": 24 * 1024**3,
    }
    gpu.update(overrides)
    return gpu


def _key(**overrides):
    key = build_generation_batch_key(
        model_fingerprint="model-v1",
        tokenizer_fingerprint="tokenizer-v1",
        gpu=_gpu(),
        dtype="torch.bfloat16",
        generation_backend="compiled_static",
        generation_compile_mode="default",
        prompt_bucket_multiple=32,
        mode="resident_search",
        expected_rows=800,
        row_shape_sha256=row_shape_sha256([64, 64, 96], bucket_multiple=32),
        max_response_length=100,
    )
    key.update(overrides)
    return key


def _context(tmp_path: Path, **overrides) -> GenerationBatchCacheContext:
    values = {
        "runtime_root": tmp_path / "runtime",
        "key": _key(),
        "required_free_bytes": 2 * 1024**3,
        "current_free_bytes": 20 * 1024**3,
        "current_total_bytes": 24 * 1024**3,
        "maximum_batch_size": 256,
    }
    values.update(overrides)
    return GenerationBatchCacheContext(**values)


def _record(batch_size: int = 40) -> dict:
    return build_cache_record(
        _key(),
        {
            "status": "PASS",
            "batch_size": batch_size,
            "validation": {
                "status": "PASS",
                "batch_size": batch_size,
                "rows": batch_size,
                "max_new_tokens": 100,
                "baseline_free_bytes": 20 * 1024**3,
                "min_free_bytes": 8 * 1024**3,
                "working_set_bytes": 12 * 1024**3,
                "required_free_bytes": 2 * 1024**3,
                "recovered_free_bytes": 20 * 1024**3,
                "peak_allocated_bytes": 16 * 1024**3,
            },
        },
    )


def test_cache_key_is_strict_for_model_tokenizer_gpu_backend_and_shape() -> None:
    base = generation_batch_key_sha256(_key())
    mutations = [
        {"model_fingerprint": "model-v2"},
        {"tokenizer_fingerprint": "tokenizer-v2"},
        {"gpu": _gpu(visible_devices="4")},
        {"gpu": _gpu(uuid="GPU-uuid-4")},
        {"gpu": _gpu(pci_bus_id="0000:04:00.0")},
        {"gpu": _gpu(total_bytes=32 * 1024**3)},
        {"dtype": "torch.float16"},
        {"generation_backend": "dynamic_eager"},
        {"generation_compile_mode": "reduce-overhead"},
        {"mode": "final_holdout"},
        {"expected_rows": 801},
        {"row_shape_sha256": row_shape_sha256([64, 64, 128], bucket_multiple=32)},
        {"max_response_length": 101},
    ]
    for mutation in mutations:
        assert generation_batch_key_sha256(_key(**mutation)) != base


def test_store_load_reuse_and_reject_low_vram_or_corrupt_cache(tmp_path: Path) -> None:
    context = _context(tmp_path)
    record = _record(batch_size=40)
    path = store_generation_batch_cache(context, record)
    assert path.parent == tmp_path / "runtime" / "generation_batch"
    assert load_generation_batch_cache(context)["batch_size"] == 40

    low_vram = _context(tmp_path, current_free_bytes=4 * 1024**3)
    assert load_generation_batch_cache(low_vram) is None

    # Old-style `current_free >= prior min_free` would accept 10 GiB, but the
    # persisted working set leaves no required headroom for a new run.
    working_set_rejected = _context(tmp_path, current_free_bytes=10 * 1024**3)
    assert load_generation_batch_cache(working_set_rejected) is None

    below_validated = _context(tmp_path, current_free_bytes=7 * 1024**3)
    assert load_generation_batch_cache(below_validated) is None

    path.write_text("{not-json", encoding="utf-8")
    assert load_generation_batch_cache(context) is None


def test_cache_record_is_text_free_and_contract_checked(tmp_path: Path) -> None:
    context = _context(tmp_path)
    record = _record(batch_size=16)
    path = store_generation_batch_cache(context, record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contract_sha256"]
    assert "PRIVATE_SENTINEL" not in path.read_text(encoding="utf-8")
    payload["batch_size"] = 17
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_generation_batch_cache(context) is None


def test_tokenizer_fingerprint_uses_public_tokenizer_contract_only() -> None:
    class Tokenizer:
        name_or_path = "tokenizer-a"
        chat_template = "template"
        vocab_size = 128
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2
        unk_token_id = 3
        padding_side = "left"
        model_max_length = 4096

    first = tokenizer_fingerprint(Tokenizer())
    assert len(first) == 64
    changed = Tokenizer()
    changed.chat_template = "other"
    assert tokenizer_fingerprint(changed) != first
