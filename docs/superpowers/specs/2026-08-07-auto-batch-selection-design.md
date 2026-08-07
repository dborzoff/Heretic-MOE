# Automatic Batch Selection Design

## Goal

Restore measured batch-size selection for Heretic-MOE searches instead of forcing every model to use batch size 8.

## Chosen approach

Use the existing startup benchmark. It tests powers of two, keeps the fastest batch that satisfies the configured VRAM reserve, and retains the last safe result when a larger batch runs out of memory. Raise the default search ceiling from 128 to 512 so small models can use the available GPUs effectively.

The three maintained adaptive-search profiles use `batch_size = 0`, which is the existing public value for automatic selection. No model-specific heuristic or new dependency is introduced.

## Runtime behavior

- New searches benchmark batch sizes `1, 2, 4, ... 512` at startup.
- The selected batch is fixed for the rest of that worker process.
- VRAM headroom and OOM handling remain unchanged.
- A search that is already running keeps the batch size loaded at its startup; its journal is not mixed with a new batch regime.
- A later process start or continuation performs selection again because the runtime profile requests `batch_size = 0`.

## Validation

- A focused test asserts the public defaults (`batch_size = 0`, `max_batch_size = 512`) and all maintained adaptive profiles.
- The focused test must fail before the change and pass afterward.
- The existing test suite is run without opening prompt/answer fixtures.

