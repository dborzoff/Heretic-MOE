# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import torch

from heretic.model import Model
from heretic.utils import Prompt


class _IdentityModule(torch.nn.Module):
    def forward(self, value):
        return value


class _FakeRoot:
    def __init__(self, embedding):
        self._embedding = embedding

    def get_input_embeddings(self):
        return self._embedding


def test_residual_hooks_copy_final_position_without_retaining_full_layer_storage(
    monkeypatch,
) -> None:
    embedding = _IdentityModule()
    layer = _IdentityModule()
    wrapper = object.__new__(Model)
    wrapper.model = _FakeRoot(embedding)
    wrapper.settings = SimpleNamespace(
        winsorization_quantile=1.0,
        offload_outputs_to_cpu=False,
    )
    wrapper.get_layers = lambda: [layer]

    def generate(prompts, **_kwargs):
        value = torch.zeros((len(prompts), 32, 16), dtype=torch.bfloat16)
        embedding(value)
        layer(value + 1)
        return {"input_ids": torch.zeros((len(prompts), 1), dtype=torch.long)}, torch.zeros(
            (len(prompts), 2), dtype=torch.long
        )

    wrapper.generate = generate
    original_stack = torch.stack
    captured_storage: list[tuple[int, int]] = []

    def inspect_stack(tensors, *args, **kwargs):
        captured_storage.extend(
            (tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
            for tensor in tensors
        )
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", inspect_stack)

    result = wrapper.get_residuals([Prompt(system="", user="synthetic")])

    assert result.shape == (1, 2, 16)
    assert captured_storage == [(32, 32), (32, 32)]
