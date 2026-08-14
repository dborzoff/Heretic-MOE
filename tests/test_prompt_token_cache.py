from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from heretic.generation_batch_selection import GenerationBatchProbe
from heretic.model import Model
from heretic.utils import Prompt


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls = 0
        self.name_or_path = "tokenizer-a"

    def __call__(self, values, **kwargs):
        self.calls += 1
        assert kwargs["padding"] is False
        return {
            "input_ids": [
                [index + 1] * (index + 1) for index, _value in enumerate(values)
            ]
        }


class _GenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def generate(self, *, input_ids, attention_mask, **kwargs):
        assert torch.equal(attention_mask, input_ids != 0)
        generated = torch.full(
            (input_ids.shape[0], 1), 7, dtype=torch.long, device=input_ids.device
        )
        return torch.cat((input_ids, generated), dim=1)


def _wrapper() -> Model:
    wrapper = object.__new__(Model)
    wrapper.model = _GenerateModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = SimpleNamespace(
        response_prefix="",
        generation_backend="dynamic_eager",
        generation_prompt_bucket_multiple=0,
        generation_compile_mode="default",
        batch_size=0,
        max_batch_size=4,
        max_response_length=100,
    )
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    return wrapper


def test_prepare_prompt_cache_tokenizes_once_and_generate_reuses_it() -> None:
    wrapper = _wrapper()
    prompts = [Prompt(system="", user="short"), Prompt(system="", user="long")]

    stats = wrapper.prepare_prompt_cache(prompts)
    inputs, outputs = wrapper.generate(list(reversed(prompts)), max_new_tokens=1)

    assert stats == {"rows": 2, "unique": 2, "new": 2, "tokens": 3}
    assert wrapper.tokenizer.calls == 1
    assert inputs["input_ids"].tolist() == [[2, 2], [0, 1]]
    assert outputs[:, -1].tolist() == [7, 7]


def test_prompt_cache_invalidates_when_response_prefix_changes() -> None:
    wrapper = _wrapper()
    prompts = [Prompt(system="", user="same")]

    wrapper.prepare_prompt_cache(prompts)
    wrapper.settings.response_prefix = "changed"
    wrapper.prepare_prompt_cache(prompts)

    assert wrapper.tokenizer.calls == 2


def test_prompt_cache_invalidates_when_tokenizer_changes() -> None:
    wrapper = _wrapper()
    prompts = [Prompt(system="", user="same")]

    wrapper.prepare_prompt_cache(prompts)
    replacement = _Tokenizer()
    replacement.name_or_path = "tokenizer-b"
    wrapper.tokenizer = replacement
    wrapper.prepare_prompt_cache(prompts)

    assert replacement.calls == 1


def test_full_model_reload_clears_prompt_runtime_cache_and_role_probe() -> None:
    wrapper = _wrapper()
    wrapper._prompt_token_cache = {("old", "prompt"): torch.tensor([1])}
    wrapper._prompt_token_arena = torch.tensor([1])
    wrapper._prompt_token_cache_signature = ("old",)
    wrapper._no_system_role = True

    wrapper._clear_prompt_runtime_cache(reset_role_probe=True)

    assert wrapper._prompt_token_cache == {}
    assert wrapper._prompt_token_arena is None
    assert wrapper._prompt_token_cache_signature is None
    assert wrapper._no_system_role is False


def test_failed_cuda_batch_releases_retained_static_cache(monkeypatch) -> None:
    wrapper = _wrapper()
    wrapper.model._cache = object()
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))

    wrapper._release_failed_cuda_batch()

    assert wrapper.model._cache is None
    assert calls == ["empty", "sync"]


def test_prompt_cache_can_be_packed_into_one_cpu_arena() -> None:
    wrapper = _wrapper()
    prompts = [Prompt(system="", user="a"), Prompt(system="", user="b")]

    wrapper.prepare_prompt_cache(prompts)
    stats = wrapper.pin_prompt_cache()
    rows = wrapper._cached_prompt_token_ids(list(reversed(prompts)))

    assert stats["rows"] == 2
    assert stats["tokens"] == 3
    assert rows[0].tolist() == [2, 2]
    assert rows[1].tolist() == [1]
    assert rows[0].untyped_storage().data_ptr() == rows[1].untyped_storage().data_ptr()


def test_prompt_collation_rounds_width_up_to_configured_multiple() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_prompt_bucket_multiple = 32
    prompts = [Prompt(system="", user="a"), Prompt(system="", user="b")]

    inputs = wrapper._collate_cached_prompts(prompts)

    assert inputs["input_ids"].shape == (2, 32)
    assert inputs["attention_mask"].sum(dim=1).tolist() == [1, 2]


def test_compiled_static_generation_is_used_only_for_multitoken_decode() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    calls: list[dict[str, object]] = []

    def record_generate(**kwargs):
        calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        generated = torch.full(
            (input_ids.shape[0], 1), 7, dtype=torch.long, device=input_ids.device
        )
        return torch.cat((input_ids, generated), dim=1)

    wrapper.model.generate = record_generate
    prompts = [Prompt(system="", user="a")]

    wrapper.generate(prompts, max_new_tokens=1)
    wrapper.generate(prompts, max_new_tokens=100)

    assert "cache_implementation" not in calls[0]
    assert "compile_config" not in calls[0]
    assert calls[1]["cache_implementation"] == "static"
    assert calls[1]["compile_config"].mode == "default"


def test_compiled_backend_prewarm_covers_full_and_tail_batch_shapes() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.generation_prompt_bucket_multiple = 32
    prompts = [Prompt(system="", user=f"row-{index}") for index in range(10)]
    calls: list[int] = []

    def capture(prompts, **_kwargs):
        calls.append(len(prompts))
        return (
            ["ok"] * len(prompts),
            [[7]] * len(prompts),
            torch.zeros((len(prompts), 1, 1)),
        )

    wrapper.get_response_artifacts_with_prefill_residuals = capture

    result = wrapper.prewarm_generation_backend(prompts)

    assert result["status"] == "PASS"
    assert result["batch_size"] == 4
    assert result["shapes"] == [[4, 32], [2, 32]]
    assert calls == [4, 2]


def test_compiled_backend_prewarm_covers_search_and_finalist_tail_shapes() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.generation_prompt_bucket_multiple = 32
    wrapper.settings.batch_size = 40
    prompts = [Prompt(system="", user=f"row-{index}") for index in range(10)]
    calls: list[int] = []

    def capture(prompts, **_kwargs):
        calls.append(len(prompts))
        return (
            ["ok"] * len(prompts),
            [[7]] * len(prompts),
            torch.zeros((len(prompts), 1, 1)),
        )

    wrapper.get_response_artifacts_with_prefill_residuals = capture

    result = wrapper.prewarm_generation_backend(
        prompts,
        expected_rows=(800, 4000, 660),
    )

    assert result["status"] == "PASS"
    assert result["shapes"] == [[40, 32], [20, 32]]
    assert calls == [40, 20]


def test_batched_artifact_progress_reports_completed_rows() -> None:
    wrapper = _wrapper()
    wrapper.settings.batch_size = 2
    prompts = [Prompt(system="", user=f"row-{index}") for index in range(5)]
    progress: list[tuple[int, int, int]] = []

    def capture(prompts, **_kwargs):
        return (
            ["ok"] * len(prompts),
            [[7]] * len(prompts),
            torch.zeros((len(prompts), 1, 1)),
        )

    wrapper.get_response_artifacts_with_prefill_residuals = capture

    wrapper.get_response_artifacts_with_prefill_residuals_batched(
        prompts,
        progress=lambda completed, total, batch: progress.append(
            (completed, total, batch)
        ),
    )

    assert progress == [(2, 5, 2), (4, 5, 2), (5, 5, 2)]


def test_compiled_batch_autotune_measures_every_eight_rows_to_40() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_batch_size = 256
    wrapper.settings.generation_batch_probe_start = 8
    wrapper.settings.generation_batch_granularity = 8
    wrapper.settings.batch_size_vram_headroom_fraction = 0.05
    wrapper.settings.batch_size_vram_headroom_gib = 1.0
    wrapper.settings.generation_batch_target_headroom_fraction = 0.10
    prompts = [Prompt(system="", user=f"row-{index}") for index in range(80)]
    gib = 1024**3
    probes = {
        8: GenerationBatchProbe(8, int(9.5 * gib), 24 * gib, 15 * gib),
        16: GenerationBatchProbe(16, 8 * gib, 24 * gib, 16 * gib),
        24: GenerationBatchProbe(24, 6 * gib, 24 * gib, 18 * gib),
        32: GenerationBatchProbe(32, 4 * gib, 24 * gib, 20 * gib),
        40: GenerationBatchProbe(40, int(1.7 * gib), 24 * gib, int(22.3 * gib)),
    }
    calls: list[int] = []

    def probe(_prompts, batch_size):
        calls.append(batch_size)
        return probes[batch_size]

    wrapper._probe_generation_batch = probe
    validated: list[int] = []

    def validation(_prompts, *, batch_size, expected_rows):
        del expected_rows
        validated.append(batch_size)
        return {
            "status": "PASS",
            "batch_size": batch_size,
            "rows": batch_size,
            "max_new_tokens": 100,
            "min_free_bytes": 0,
            "required_free_bytes": 0,
            "recovered_free_bytes": 0,
        }

    wrapper.validate_generation_batch_size = validation
    result = wrapper.autotune_generation_batch_size(prompts, expected_rows=800)

    assert calls == [8, 16, 24, 32, 40]
    assert validated == [16, 32, 40]
    assert result["status"] == "PASS"
    assert result["batch_size"] == 40
    assert wrapper._adaptive_generation_batch_size == 40


def test_batch_autotune_selects_fastest_safe_real_generation_batch() -> None:
    """Catch regressions that choose the largest VRAM-fitting batch over throughput."""

    wrapper = _wrapper()
    wrapper.settings.generation_backend = "dynamic_eager"
    wrapper.settings.max_batch_size = 128
    wrapper.settings.generation_batch_probe_start = 128
    wrapper.settings.generation_batch_granularity = 32
    wrapper.settings.batch_size_vram_headroom_fraction = 0.10
    wrapper.settings.batch_size_vram_headroom_gib = 2.0
    wrapper.settings.generation_batch_target_headroom_fraction = 0.10
    gib = 1024**3
    wrapper._probe_generation_batch = lambda _prompts, batch: GenerationBatchProbe(
        batch,
        4 * gib,
        24 * gib,
        20 * gib,
    )
    measured = {
        64: 8_000.0,
        96: 9_500.0,
        128: 8_700.0,
    }
    calls: list[int] = []

    def validate(_prompts, *, batch_size, expected_rows):
        calls.append(batch_size)
        return {
            "status": "PASS",
            "batch_size": batch_size,
            "rows": batch_size,
            "expected_rows": expected_rows,
            "max_new_tokens": 100,
            "min_free_bytes": 4 * gib,
            "required_free_bytes": 2 * gib,
            "recovered_free_bytes": 20 * gib,
            "tokens_per_second": measured[batch_size],
        }

    wrapper.validate_generation_batch_size = validate
    result = wrapper.autotune_generation_batch_size(
        [Prompt(system="", user="row") for _index in range(128)],
        expected_rows=800,
    )

    assert calls == [64, 96, 128]
    assert result["batch_size"] == 96
    assert result["throughput"]["tokens_per_second"] == 9_500.0
    assert wrapper._adaptive_generation_batch_size == 96


def test_batch_autotune_runs_exactly_one_short_validation_batch(monkeypatch) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_batch_size = 8
    wrapper.settings.generation_batch_probe_start = 8
    wrapper.settings.generation_batch_granularity = 8
    wrapper.settings.batch_size_vram_headroom_fraction = 0.05
    wrapper.settings.batch_size_vram_headroom_gib = 1.0
    wrapper.settings.generation_batch_target_headroom_fraction = 0.10
    wrapper.settings.max_response_length = 100
    gib = 1024**3
    wrapper._probe_generation_batch = lambda _prompts, _batch: GenerationBatchProbe(
        8,
        8 * gib,
        24 * gib,
        16 * gib,
    )
    calls: list[dict[str, object]] = []

    def generate(batch, **kwargs):
        calls.append({"rows": len(batch), **kwargs})
        inputs = {"input_ids": torch.zeros((len(batch), 1), dtype=torch.long)}
        outputs = torch.ones(
            (len(batch), 1 + int(kwargs["max_new_tokens"])),
            dtype=torch.long,
        )
        return inputs, outputs

    wrapper.generate = generate
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (8 * gib, 24 * gib, 16 * gib),
            (20 * gib, 24 * gib, 0),
        ]
    )
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = wrapper.autotune_generation_batch_size(
        [Prompt(system="", user="row") for _index in range(80)],
        expected_rows=803,
    )

    assert result["batch_size"] == 8
    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["max_new_tokens"] == 100
    assert result["validation"]["working_set_bytes"] == 12 * gib
    assert result["validation"]["baseline_free_bytes"] == 20 * gib
    assert result["validation"]["generated_tokens"] == 800
    assert result["validation"]["elapsed_seconds"] > 0.0
    assert result["validation"]["tokens_per_second"] > 0.0
    assert calls == [
        {"rows": 8, "max_new_tokens": 100, "min_new_tokens": 100}
    ]


def test_batch_autotune_backs_off_below_probe_start_when_reserve_is_too_low() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_batch_size = 256
    wrapper.settings.generation_batch_probe_start = 8
    wrapper.settings.generation_batch_granularity = 8
    wrapper.settings.batch_size_vram_headroom_fraction = 0.05
    wrapper.settings.batch_size_vram_headroom_gib = 1.0
    wrapper.settings.generation_batch_target_headroom_fraction = 0.10
    gib = 1024**3
    probes = {
        8: GenerationBatchProbe(8, int(0.8 * gib), 24 * gib, 23 * gib),
        4: GenerationBatchProbe(4, int(0.9 * gib), 24 * gib, 22 * gib),
        2: GenerationBatchProbe(2, 3 * gib, 24 * gib, 21 * gib),
    }
    calls: list[int] = []

    def probe(_prompts, batch_size):
        calls.append(batch_size)
        return probes[batch_size]

    wrapper._probe_generation_batch = probe
    wrapper.validate_generation_batch_size = lambda _prompts, *, batch_size, expected_rows: {
        "status": "PASS",
        "batch_size": batch_size,
        "rows": batch_size,
        "max_new_tokens": 100,
        "min_free_bytes": 0,
        "required_free_bytes": 0,
        "recovered_free_bytes": 0,
    }

    result = wrapper.autotune_generation_batch_size(
        [Prompt(system="", user="row")],
        expected_rows=800,
    )

    assert calls == [8, 4, 2]
    assert result["batch_size"] == 2




def test_validation_checks_batch_shape_without_decoding_generated_text(monkeypatch) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 100
    gib = 1024**3
    prompts = [Prompt(system="", user="row") for _index in range(8)]

    def bad_generate(batch, **kwargs):
        del kwargs
        inputs = {"input_ids": torch.zeros((len(batch), 1), dtype=torch.long)}
        outputs = torch.ones((len(batch) + 1, 2), dtype=torch.long)
        return inputs, outputs

    wrapper.generate = bad_generate
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (20 * gib, 24 * gib, 0),
        ]
    )
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(ValueError, match="invalid batch shape"):
        wrapper.validate_generation_batch_size(prompts, batch_size=8, expected_rows=800)


def test_validation_rejects_early_eos_that_undermeasures_memory(monkeypatch) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 100
    gib = 1024**3
    prompts = [Prompt(system="", user="row") for _index in range(8)]

    def early_eos_generate(batch, **kwargs):
        assert kwargs == {"max_new_tokens": 100, "min_new_tokens": 100}
        inputs = {"input_ids": torch.zeros((len(batch), 1), dtype=torch.long)}
        outputs = torch.ones((len(batch), 2), dtype=torch.long)
        return inputs, outputs

    wrapper.generate = early_eos_generate
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (20 * gib, 24 * gib, 0),
        ]
    )
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(ValueError, match="expected 100 new tokens"):
        wrapper.validate_generation_batch_size(prompts, batch_size=8, expected_rows=800)


def test_validation_always_measures_100_tokens_even_with_a_lower_runtime_limit(
    monkeypatch,
) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 99
    wrapper.settings.batch_size_vram_headroom_fraction = 0.05
    wrapper.settings.batch_size_vram_headroom_gib = 1.0
    gib = 1024**3
    prompts = [Prompt(system="", user="row") for _index in range(8)]
    calls: list[dict[str, int]] = []

    def generate(batch, **kwargs):
        calls.append(kwargs)
        inputs = {"input_ids": torch.zeros((len(batch), 1), dtype=torch.long)}
        outputs = torch.ones((len(batch), 101), dtype=torch.long)
        return inputs, outputs

    wrapper.generate = generate
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (8 * gib, 24 * gib, 16 * gib),
            (20 * gib, 24 * gib, 0),
        ]
    )
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = wrapper.validate_generation_batch_size(
        prompts,
        batch_size=8,
        expected_rows=800,
    )

    assert result["max_new_tokens"] == 100
    assert calls == [{"max_new_tokens": 100, "min_new_tokens": 100}]

def test_batch_autotune_validation_oom_reduces_batch_and_retries() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_batch_size = 256
    wrapper.settings.generation_batch_probe_start = 8
    wrapper.settings.generation_batch_granularity = 8
    wrapper.settings.batch_size_vram_headroom_fraction = 0.05
    wrapper.settings.batch_size_vram_headroom_gib = 1.0
    wrapper.settings.generation_batch_target_headroom_fraction = 0.10
    gib = 1024**3
    wrapper._probe_generation_batch = lambda _prompts, _batch: GenerationBatchProbe(
        8,
        8 * gib,
        24 * gib,
        16 * gib,
    )
    attempts: list[int] = []

    def validation(_prompts, *, batch_size, expected_rows):
        del expected_rows
        attempts.append(batch_size)
        if batch_size == 8:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return {
            "status": "PASS",
            "batch_size": batch_size,
            "rows": batch_size,
            "max_new_tokens": 100,
            "min_free_bytes": 8 * gib,
            "required_free_bytes": 2 * gib,
            "recovered_free_bytes": 20 * gib,
        }

    wrapper.validate_generation_batch_size = validation
    result = wrapper.autotune_generation_batch_size(
        [Prompt(system="", user="row")],
        expected_rows=800,
    )

    assert attempts == [8, 4]
    assert result["batch_size"] == 4
    assert result["validation_attempts"][0]["status"] == "OOM"
    assert wrapper._adaptive_generation_batch_size == 4


def test_memory_probe_presizes_cache_for_full_trial_but_decodes_one_token(
    monkeypatch,
) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.generation_prompt_bucket_multiple = 64
    wrapper.settings.max_response_length = 100
    wrapper.settings.generation_batch_recovery_tolerance_mib = 256
    wrapper.model.generation_config = SimpleNamespace(max_length=262144)
    prompts = [Prompt(system="", user="short") for _index in range(8)]
    calls: list[dict[str, object]] = []

    def generate(prompts, **kwargs):
        calls.append(
            {
                "rows": len(prompts),
                "model_max_length": wrapper.model.generation_config.max_length,
                **kwargs,
            }
        )
        inputs = {"input_ids": torch.zeros((len(prompts), 64), dtype=torch.long)}
        outputs = torch.ones((len(prompts), 65), dtype=torch.long)
        return inputs, outputs

    wrapper.generate = generate
    gib = 1024**3
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (8 * gib, 24 * gib, 16 * gib),
            (20 * gib, 24 * gib, 0),
        ]
    )
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    wrapper.model._cache = object()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    probe = wrapper._probe_generation_batch(prompts, 8)

    assert calls == [
        {
            "rows": 8,
            "model_max_length": None,
            "max_new_tokens": 1,
            "cache_implementation": "static",
            "max_cache_len": 164,
            "disable_compile": True,
        }
    ]
    assert probe.batch_size == 8
    assert probe.baseline_free_bytes == 20 * gib
    assert probe.recovered_free_bytes == 20 * gib
    assert wrapper.model._cache is None
    assert wrapper.model.generation_config.max_length == 262144


def test_memory_probe_uses_longest_cached_prompts() -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 100
    wrapper.settings.generation_batch_recovery_tolerance_mib = 256
    prompts = [
        Prompt(system="", user="short"),
        Prompt(system="", user="long"),
    ]
    wrapper.prepare_prompt_cache(prompts)
    calls: list[int] = []

    def generate(_prompts, **kwargs):
        calls.append(int(kwargs["max_cache_len"]))
        return {"input_ids": torch.zeros((1, 2), dtype=torch.long)}, torch.ones(
            (1, 3), dtype=torch.long
        )

    wrapper.generate = generate
    wrapper._cuda_memory_snapshot = lambda: (20 * 1024**3, 24 * 1024**3, 0)
    wrapper._release_generation_probe_cache = lambda: None
    original_reset = torch.cuda.reset_peak_memory_stats
    original_synchronize = torch.cuda.synchronize
    try:
        torch.cuda.reset_peak_memory_stats = lambda: None
        torch.cuda.synchronize = lambda: None
        wrapper._probe_generation_batch(prompts, 1)
    finally:
        torch.cuda.reset_peak_memory_stats = original_reset
        torch.cuda.synchronize = original_synchronize

    assert calls == [102]


def test_memory_probe_rechecks_recovery_before_reraising_oom(monkeypatch) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 100
    wrapper.settings.generation_batch_recovery_tolerance_mib = 256
    prompts = [Prompt(system="", user="row")]
    gib = 1024**3
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (20 * gib, 24 * gib, 0),
        ]
    )
    releases: list[bool] = []

    def oom(*_args, **_kwargs):
        raise torch.OutOfMemoryError("probe oom")

    wrapper.generate = oom
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    wrapper._release_generation_probe_cache = lambda: releases.append(True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    with pytest.raises(torch.OutOfMemoryError, match="probe oom"):
        wrapper._probe_generation_batch(prompts, 8)

    assert releases == [True, True]


def test_memory_probe_reports_leaked_cache_instead_of_hiding_it_as_oom(
    monkeypatch,
) -> None:
    wrapper = _wrapper()
    wrapper.settings.generation_backend = "compiled_static"
    wrapper.settings.max_response_length = 100
    wrapper.settings.generation_batch_recovery_tolerance_mib = 256
    prompts = [Prompt(system="", user="row")]
    gib = 1024**3
    snapshots = iter(
        [
            (20 * gib, 24 * gib, 0),
            (19 * gib, 24 * gib, 0),
        ]
    )

    def oom(*_args, **_kwargs):
        raise torch.OutOfMemoryError("probe oom")

    wrapper.generate = oom
    wrapper._cuda_memory_snapshot = lambda: next(snapshots)
    wrapper._release_generation_probe_cache = lambda: None
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    with pytest.raises(RuntimeError, match="did not release its CUDA cache"):
        wrapper._probe_generation_batch(prompts, 8)
