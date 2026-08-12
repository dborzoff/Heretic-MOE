# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure selection rules for resident generation batch tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationBatchProbe:
    batch_size: int
    free_bytes: int
    total_bytes: int
    peak_allocated_bytes: int
    baseline_free_bytes: int = 0
    recovered_free_bytes: int = 0


def next_batch_candidate(
    *,
    current_batch_size: int,
    current_free_bytes: int,
    required_free_bytes: int,
    maximum_batch_size: int,
    granularity: int = 8,
) -> int | None:
    """Advance one measured step without skipping intermediate VRAM probes."""

    if current_batch_size <= 0 or maximum_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if current_free_bytes < 0 or required_free_bytes < 0:
        raise ValueError("memory byte counts must be nonnegative")
    if granularity <= 0:
        raise ValueError("batch granularity must be positive")
    if current_free_bytes <= required_free_bytes:
        return None
    candidate = min(maximum_batch_size, current_batch_size + granularity)
    return candidate if candidate > current_batch_size else None
