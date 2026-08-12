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
    wrapper.settings = SimpleNamespace(response_prefix="")
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
