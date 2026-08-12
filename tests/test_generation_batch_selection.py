from __future__ import annotations

from heretic.generation_batch_selection import next_batch_candidate

GIB = 1024**3


def test_next_batch_candidate_advances_by_one_granularity_step() -> None:
    assert (
        next_batch_candidate(
            current_batch_size=32,
            current_free_bytes=3 * GIB,
            required_free_bytes=int(1.5 * GIB),
            maximum_batch_size=256,
            granularity=8,
        )
        == 40
    )


def test_next_batch_candidate_does_not_skip_intermediate_memory_probes() -> None:
    assert (
        next_batch_candidate(
            current_batch_size=32,
            current_free_bytes=12 * GIB,
            required_free_bytes=2 * GIB,
            maximum_batch_size=256,
            granularity=8,
        )
        == 40
    )


def test_next_batch_candidate_stops_when_current_probe_reaches_reserve() -> None:
    assert (
        next_batch_candidate(
            current_batch_size=32,
            current_free_bytes=2 * GIB,
            required_free_bytes=2 * GIB,
            maximum_batch_size=256,
            granularity=8,
        )
        is None
    )


def test_first_successful_probe_advances_by_one_step() -> None:
    assert (
        next_batch_candidate(
            current_batch_size=8,
            current_free_bytes=20 * GIB,
            required_free_bytes=2 * GIB,
            maximum_batch_size=128,
            granularity=8,
        )
        == 16
    )
