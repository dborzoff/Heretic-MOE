# SPDX-License-Identifier: AGPL-3.0-or-later

"""Teacher-forced preservation metrics shared by clean and trial evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def per_row_conditional_nll(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Return one causal mean NLL per row while ignoring prompt/padding labels."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels shape mismatch")
    if logits.shape[1] < 2:
        raise ValueError("sequence must contain at least two positions")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain non-finite values")

    shifted_logits = logits[:, :-1, :].to(torch.float32)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    counts = mask.sum(dim=1)
    if bool(torch.any(counts == 0)):
        bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"rows have no target tokens: {bad}")

    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(shifted_labels.shape)
    return ((losses * mask).sum(dim=1) / counts).detach().cpu()
