from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from heretic.config import RowNormalization
from heretic.model import AbliterationParameters, Model


def _parameter() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros((2, 4, 8)))


def test_fused_expert_discovery_covers_supported_container_and_weight_names() -> None:
    expected = []
    layers = []
    for block_name, weight_name in (
        ("mlp", "down_proj"),
        ("block_sparse_moe", "w2"),
        ("feed_forward", "w2"),
        ("moe", "output_linear"),
    ):
        parameter = _parameter()
        expected.append(parameter)
        experts = SimpleNamespace(**{weight_name: parameter})
        layers.append(SimpleNamespace(**{block_name: SimpleNamespace(experts=experts)}))
    wrapper = Model.__new__(Model)
    wrapper.get_layers = lambda: layers

    found = [parameter for _index, parameter in wrapper._iter_fused_expert_parameters()]

    assert found == expected
    assert wrapper._has_fused_experts()


def _fallback_wrapper(config: object) -> Model:
    wrapper = Model.__new__(Model)
    wrapper.model = SimpleNamespace(config=config)
    wrapper._fused_experts_cache = {}
    wrapper._iter_fused_expert_parameters = lambda: [(0, _parameter())]
    wrapper.settings = SimpleNamespace(
        fused_expert_chunk_size=2,
        record_edit_telemetry=False,
        row_normalization=RowNormalization.NONE,
    )
    return wrapper


def _skip_parameters() -> dict[str, AbliterationParameters]:
    return {
        "mlp.experts.down_proj": AbliterationParameters(
            max_weight=0.0,
            max_weight_position=0.0,
            min_weight=0.0,
            min_weight_distance=1.0,
        )
    }


@pytest.mark.parametrize("error_type", (AttributeError, TypeError))
def test_text_config_fallback_is_limited_to_expected_errors(error_type) -> None:
    class Config:
        hidden_size = 999

        def get_text_config(self):
            raise error_type("expected fallback")

    wrapper = _fallback_wrapper(Config())

    wrapper._abliterate_fused_experts(
        torch.zeros((2, 4)),
        None,
        _skip_parameters(),
    )

    assert wrapper._fused_experts_cache == {}


def test_text_config_fallback_does_not_swallow_unexpected_errors() -> None:
    class Config:
        hidden_size = 999

        def get_text_config(self):
            raise ValueError("unexpected")

    wrapper = _fallback_wrapper(Config())

    with pytest.raises(ValueError, match="unexpected"):
        wrapper._abliterate_fused_experts(
            torch.zeros((2, 4)),
            None,
            _skip_parameters(),
        )
