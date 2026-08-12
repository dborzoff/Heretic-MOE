from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch
from torch import nn

from heretic.model import Model
from heretic.utils import Prompt


class _ShiftLayer(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
        return (values + self.shift,)


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 3)
        self.layers = nn.ModuleList([_ShiftLayer(1.0), _ShiftLayer(2.0)])
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


def test_one_generation_captures_prefill_without_retaining_decode_steps() -> None:
    wrapper = object.__new__(Model)
    fake = _FakeLanguageModel()
    wrapper.model = fake
    wrapper.settings = SimpleNamespace(
        max_response_length=3,
        offload_outputs_to_cpu=True,
        winsorization_quantile=1.0,
        batch_size=2,
    )
    wrapper.tokenizer = SimpleNamespace(
        batch_decode=lambda values, skip_special_tokens: [
            "decoded-" + "-".join(str(int(item)) for item in row) for row in values
        ]
    )
    wrapper.get_layers = MethodType(lambda self: fake.layers, wrapper)

    def generate(self, prompts, **kwargs):
        assert kwargs == {"max_new_tokens": 3}
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        hidden = fake.embedding(input_ids)
        for layer in fake.layers:
            hidden = layer(hidden)[0]
        # Simulate one decode call with very different values. The capture must
        # retain the first (prefill) invocation only.
        decode = fake.embedding(torch.tensor([[20], [21]]))
        for layer in fake.layers:
            decode = layer(decode)[0]
        output_ids = torch.tensor([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])
        return {"input_ids": input_ids}, output_ids

    wrapper.generate = MethodType(generate, wrapper)
    prompts = [Prompt(system="", user="one"), Prompt(system="", user="two")]

    responses, token_ids, residuals = wrapper.get_response_artifacts_with_prefill_residuals(
        prompts,
        skip_special_tokens=True,
    )

    expected_embedding = fake.embedding(torch.tensor([3, 6]))
    assert responses == ["decoded-7-8-9", "decoded-10-11-12"]
    assert token_ids == [[7, 8, 9], [10, 11, 12]]
    assert residuals.shape == (2, 3, 3)
    assert residuals.device.type == "cpu"
    assert torch.equal(residuals[:, 0], expected_embedding)
    assert torch.equal(residuals[:, 1], expected_embedding + 1.0)
    assert torch.equal(residuals[:, 2], expected_embedding + 3.0)
    assert all(not module._forward_hooks for module in [fake.embedding, *fake.layers])


def test_batched_prefill_capture_preserves_prompt_order() -> None:
    wrapper = object.__new__(Model)
    wrapper.settings = SimpleNamespace(batch_size=2)

    def capture(self, prompts, skip_special_tokens=False):
        values = [int(prompt.user) for prompt in prompts]
        return (
            [f"r-{value}" for value in values],
            torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1),
        )

    wrapper.get_responses_with_prefill_residuals = MethodType(capture, wrapper)
    prompts = [Prompt(system="", user=str(index)) for index in range(5)]

    responses, residuals = wrapper.get_responses_with_prefill_residuals_batched(prompts)

    assert responses == ["r-0", "r-1", "r-2", "r-3", "r-4"]
    assert residuals.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_batched_artifact_capture_preserves_text_tokens_and_residual_order() -> None:
    wrapper = object.__new__(Model)
    wrapper.settings = SimpleNamespace(batch_size=2)

    def capture(self, prompts, skip_special_tokens=False):
        values = [int(prompt.user) for prompt in prompts]
        return (
            [f"r-{value}" for value in values],
            [[value, value + 10] for value in values],
            torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1),
        )

    wrapper.get_response_artifacts_with_prefill_residuals = MethodType(
        capture, wrapper
    )
    prompts = [Prompt(system="", user=str(index)) for index in range(5)]

    responses, token_ids, residuals = (
        wrapper.get_response_artifacts_with_prefill_residuals_batched(prompts)
    )

    assert responses == ["r-0", "r-1", "r-2", "r-3", "r-4"]
    assert token_ids == [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]]
    assert residuals.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_adaptive_artifact_batch_buckets_lengths_and_recovers_from_oom() -> None:
    wrapper = object.__new__(Model)
    wrapper.settings = SimpleNamespace(batch_size=0, max_batch_size=8)
    attempted: list[list[str]] = []

    def capture(self, prompts, skip_special_tokens=False):
        attempted.append([prompt.user for prompt in prompts])
        if len(prompts) > 2:
            raise torch.OutOfMemoryError("synthetic OOM")
        values = [int(prompt.user.split("-")[0]) for prompt in prompts]
        return (
            [f"r-{value}" for value in values],
            [[value] for value in values],
            torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1),
        )

    wrapper.get_response_artifacts_with_prefill_residuals = MethodType(
        capture, wrapper
    )
    prompts = [
        Prompt(system="", user="0-xxxxxxxx"),
        Prompt(system="", user="1-x"),
        Prompt(system="", user="2-xxxx"),
        Prompt(system="", user="3-xx"),
    ]

    responses, token_ids, residuals = (
        wrapper.get_response_artifacts_with_prefill_residuals_batched(prompts)
    )

    assert responses == ["r-0", "r-1", "r-2", "r-3"]
    assert token_ids == [[0], [1], [2], [3]]
    assert residuals.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert wrapper._adaptive_generation_batch_size == 2
    assert any(len(batch) == 4 for batch in attempted)
    assert [len(value) for value in attempted[-1]] == sorted(
        len(value) for value in attempted[-1]
    )
