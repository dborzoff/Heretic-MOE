# -*- coding: utf-8 -*-
"""Поворот Адамара из comfy_kitchen, вынесенный без зависимостей.

Функции скопированы дословно из
comfy_kitchen/tensor/int8_utils.py, чтобы формат совпадал с тем, что
читает ComfyUI. Переписывать по памяти нельзя: разойдётся хоть в знаке -
и файл станет несовместимым.
"""
import math

import torch

_HADAMARD_CACHE: dict = {}


def _build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def _rotate_weight(
    weight: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)
