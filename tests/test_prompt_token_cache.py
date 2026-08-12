from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from heretic.model import Model
from heretic.utils import Prompt


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

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
