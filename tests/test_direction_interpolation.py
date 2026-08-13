from __future__ import annotations

import torch

from heretic.model import Model


def test_last_residual_direction_does_not_read_past_tensor() -> None:
    directions = torch.eye(3)

    selected = Model._interpolate_residual_direction(directions, 1.0)

    assert torch.equal(selected, directions[2])


def test_fractional_residual_direction_interpolates_adjacent_rows() -> None:
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    selected = Model._interpolate_residual_direction(directions, 0.5)

    expected = torch.nn.functional.normalize(
        directions[1].lerp(directions[2], 0.5),
        p=2,
        dim=0,
    )
    assert torch.allclose(selected, expected)
