# Automatic Batch Selection Design

## Goal

Restore measured batch-size selection for Heretic-MOE searches instead of forcing every model to use batch size 8.

## Chosen approach

Use the existing startup benchmark. It tests powers of two, keeps the fastest batch that satisfies the configured VRAM reserve, and retains the last safe result when a larger batch runs out of memory. Raise the default search ceiling to 4096 so small models can use the available GPUs effectively, while reserving at least 10% of total VRAM and the existing 2 GiB absolute minimum.

The three maintained adaptive-search profiles use `batch_size = 0`, which is the existing public value for automatic selection. No model-specific heuristic or new dependency is introduced.

## Runtime behavior

- New searches benchmark batch sizes `1, 2, 4, ... 4096` at startup.
- The selected batch is fixed for the rest of that worker process.
- A candidate is eligible only while at least `max(10% total VRAM, 2 GiB)` remains free.
- Before doubling again, the selector extrapolates the latest VRAM-growth step and skips a candidate whose predicted free memory is below that reserve.
- OOM handling remains unchanged.
- A search that was started with fixed batch size 8 is preserved and replaced by a new run root; its journal is not mixed with the new batch regime.
- A later process start or continuation performs selection again because the runtime profile requests `batch_size = 0`.

## Validation

- A focused test asserts the public defaults (`batch_size = 0`, `max_batch_size = 4096`, `batch_size_vram_headroom_fraction = 0.10`) and all maintained adaptive profiles.
- The focused test must fail before the change and pass afterward.
- The existing test suite is run without opening prompt/answer fixtures.
