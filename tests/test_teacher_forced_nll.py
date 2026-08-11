from __future__ import annotations

import math

import pytest
import torch
from torch import nn

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


class _CausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, *, input_ids, attention_mask, use_cache):
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
