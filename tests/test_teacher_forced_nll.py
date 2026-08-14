from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import heretic.model as model_module
from heretic.model import Model
from heretic.teacher_forced import per_row_conditional_nll
from heretic.utils import Prompt


def test_per_row_nll_ignores_prompt_and_padding_labels() -> None:
    # For each label position make the expected token have a known probability.
    logits = torch.zeros((2, 4, 3), dtype=torch.float64)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2],
            [-100, 0, -100, -100],
        ]
    )
    # Causal shift: label[2] uses logits[1], label[3] uses logits[2].
    logits[0, 1, 1] = math.log(4.0)
    logits[0, 2, 2] = math.log(9.0)
    logits[1, 0, 0] = math.log(5.0)

    values = per_row_conditional_nll(logits, labels)

    expected_0 = -0.5 * (
        torch.log_softmax(logits[0, 1], dim=-1)[1]
        + torch.log_softmax(logits[0, 2], dim=-1)[2]
    )
    expected_1 = -torch.log_softmax(logits[1, 0], dim=-1)[0]
    assert values.tolist() == pytest.approx([float(expected_0), float(expected_1)])


def test_per_row_nll_rejects_rows_without_target_tokens() -> None:
    logits = torch.zeros((1, 3, 4))
    labels = torch.full((1, 3), -100)

    with pytest.raises(ValueError, match="no target tokens"):
        per_row_conditional_nll(logits, labels)


def test_per_row_nll_validates_shapes_and_finiteness() -> None:
    with pytest.raises(ValueError, match="shape"):
        per_row_conditional_nll(torch.zeros((2, 3)), torch.zeros((2, 3)))
    logits = torch.zeros((1, 3, 4))
    logits[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        per_row_conditional_nll(logits, torch.tensor([[-100, 1, -100]]))


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, values, **kwargs):
        assert kwargs["padding"] is False
        return {"input_ids": [[1, 2] for _ in values]}


class _CountingTokenizer(_Tokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, values, **kwargs):
        self.calls += 1
        return super().__call__(values, **kwargs)


class _CausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.batch_sizes.append(int(input_ids.shape[0]))
        assert use_cache is False
        assert torch.equal(attention_mask, input_ids != 0)
        batch, sequence = input_ids.shape
        logits = torch.zeros((batch, sequence, 8), device=input_ids.device)
        for row in range(batch):
            # prompt length is two; target labels are [3, 4]
            logits[row, 1, 3] = 5.0
            logits[row, 2, 4] = 5.0
        return type("Output", (), {"logits": logits})()


def test_model_conditional_nll_uses_fixed_target_tokens_without_generation() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type("Settings", (), {"batch_size": 2})()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user="a"), Prompt(system="", user="b")],
        [[3, 4], [3, 4]],
    )

    expected = -torch.log_softmax(torch.tensor([0.0] * 3 + [5.0] + [0.0] * 4), -1)[
        3
    ]
    assert values == pytest.approx([float(expected), float(expected)])


def test_fixed_prompt_tokenization_is_cached_across_nll_trials() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _CountingTokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 2,
            "conditional_nll_batch_size": 2,
            "response_prefix": None,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    prompts = [Prompt(system="", user="a"), Prompt(system="", user="b")]

    first = wrapper.get_conditional_nll(prompts, [[3, 4], [3, 4]])
    second = wrapper.get_conditional_nll(prompts, [[3, 4], [3, 4]])

    assert first == pytest.approx(second)
    assert wrapper.tokenizer.calls == 1


def test_fixed_conditional_nll_releases_transient_cuda_cache_once_per_call() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 2,
            "conditional_nll_batch_size": 2,
            "response_prefix": None,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    releases: list[None] = []
    wrapper._release_failed_cuda_batch = lambda: releases.append(None)

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(4)],
        [[3, 4] for _ in range(4)],
    )

    assert len(values) == 4
    assert len(releases) == 1


class _LengthCausalModel(_CausalModel):
    def __init__(self):
        super().__init__()
        self.sequence_lengths: list[int] = []

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.sequence_lengths.append(int(input_ids.shape[1]))
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )


def test_conditional_nll_processes_longest_sequences_first_for_allocator_reuse() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _LengthCausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 1,
            "conditional_nll_batch_size": 1,
            "response_prefix": None,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    wrapper._release_failed_cuda_batch = lambda: None

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(3)],
        [[3], [3, 4, 3, 4], [3, 4]],
    )

    assert len(values) == 3
    assert wrapper.model.sequence_lengths == [6, 4, 3]


class _OomCausalModel(_CausalModel):
    def forward(self, *, input_ids, attention_mask, use_cache):
        if input_ids.shape[0] > 2:
            self.batch_sizes.append(int(input_ids.shape[0]))
            raise torch.OutOfMemoryError("synthetic OOM")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )


def test_conditional_nll_has_independent_adaptive_batch_with_oom_backoff() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _OomCausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 1,
            "conditional_nll_batch_size": 0,
            "max_batch_size": 8,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(5)],
        [[3, 4] for _ in range(5)],
    )

    assert len(values) == 5
    assert wrapper._adaptive_nll_batch_size == 2
    assert 5 in wrapper.model.batch_sizes
    assert max(size for size in wrapper.model.batch_sizes if size <= 2) == 2


def test_conditional_nll_retries_oom_raised_by_loss_materialization(
    monkeypatch,
) -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 1,
            "conditional_nll_batch_size": 0,
            "max_batch_size": 4,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    original = model_module.per_row_conditional_nll

    def score(logits, labels):
        if logits.shape[0] > 2:
            raise torch.OutOfMemoryError("synthetic loss OOM")
        return original(logits, labels)

    monkeypatch.setattr(model_module, "per_row_conditional_nll", score)

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(4)],
        [[3, 4] for _ in range(4)],
    )

    assert len(values) == 4
    assert wrapper._adaptive_nll_batch_size == 2
    assert wrapper.model.batch_sizes[:2] == [4, 2]


def test_conditional_nll_reduces_batch_until_vram_headroom_is_ten_percent(
    monkeypatch,
) -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 1,
            "conditional_nll_batch_size": 0,
            "max_batch_size": 4,
            "batch_size_vram_headroom_fraction": 0.10,
            "batch_size_vram_headroom_gib": 0.0,
        },
    )()
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    snapshots = iter(
        [
            (50, 1000, 0),
            (200, 1000, 0),
            (150, 1000, 0),
            (200, 1000, 0),
        ]
    )
    monkeypatch.setattr(wrapper, "_cuda_memory_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(wrapper, "_release_failed_cuda_batch", lambda: None)

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(4)],
        [[3, 4] for _ in range(4)],
    )

    assert len(values) == 4
    assert wrapper._adaptive_nll_batch_size == 3
    assert wrapper.model.batch_sizes[:3] == [4, 2, 3]


def test_automatic_conditional_nll_starts_from_quarter_generation_batch() -> None:
    wrapper = object.__new__(Model)
    wrapper.model = _CausalModel()
    wrapper.tokenizer = _Tokenizer()
    wrapper.settings = type(
        "Settings",
        (),
        {
            "batch_size": 1,
            "conditional_nll_batch_size": 0,
            "max_batch_size": 400,
        },
    )()
    wrapper._adaptive_generation_batch_size = 176
    wrapper._render_chat_prompts = lambda prompts: [prompt.user for prompt in prompts]

    values = wrapper.get_conditional_nll(
        [Prompt(system="", user=str(index)) for index in range(400)],
        [[3, 4] for _ in range(400)],
    )

    assert len(values) == 400
    assert wrapper.model.batch_sizes[0] == 44
    assert wrapper._adaptive_nll_batch_size == 44
