# Qwen3-VL-32B rental runbook

Status: prepared only. Do not rent a machine, create a Hugging Face repository,
delete a repository, or upload model weights until the user gives an explicit
go-ahead.

## Release target

- Base: `Qwen/Qwen3-VL-32B-Instruct`
- Frozen revision: `0cfaf48183f594c314753d30a4c4974bc75f3ccb`
- New repository: `DmitryDB/Qwen3-VL-32B-Instruct-Heretic-Adaptive-v1`
- Layout: `balanced/` and `max-removal/`
- Publish BF16 Transformers checkpoints only. Quantization is deferred to the
  local machine.
- Keep the existing Qwen3.6-35B-A3B repositories unchanged because they use a
  different base architecture.

## Required inputs

The Git checkout supplies the Heretic code, regression tests, built-in
perplexity reference, and run configuration. Transfer these private runtime
inputs separately without printing their contents:

- `direction_safe.jsonl`
- `direction_unsafe.jsonl`
- `search_unsafe.jsonl`
- `sparse_geometry_bank_v1/prototypes.jsonl`

The four files are small and must not be committed to the public repository.
Verify their SHA-256 values after transfer.

## Machine policy

Use one full-power RTX PRO 6000 96GB for the smoke run and retain it for the
full run if the measured projection is at most eight hours. Require at least
580 W, 300 GB disk, verified hosting, reliability at least 0.99, inexpensive
traffic, and fast inbound/outbound networking. Use a two-GPU host only if the
single-GPU timing projection exceeds eight hours.

## Execution gates

1. Verify GPU model, power limit, driver, free disk, and network.
2. Clone the exact Heretic source commit and verify it.
3. Transfer and hash-check the four private runtime inputs.
4. Download the frozen Qwen revision directly from Hugging Face with Xet high
   performance mode.
5. Run architecture/load/generation smoke checks before the search.
6. Run a short timing sample and project the 600-trial duration.
7. Continue on one GPU only when the projection is at most eight hours.
8. Save every trial response and all Optuna state needed to reconstruct any
   trial from the journal.
9. Select several finalists, remeasure them with the full PPL protocol, and
   export the balanced and max-removal winners sequentially.
10. Verify export/reload before uploading.

## Publication safeguards

- Create a new repository; never overwrite the existing Qwen3.6 repositories.
- Keep the model card in English.
- Do not upload raw prompts or raw answers.
- Include only text-free evaluation aggregates, selected trial parameters,
  hashes, source commit, base revision, and reproduction notes.
- For MiniMax-H3 integration, replace weight shards only and retain MiniMax's
  tokenizer/processor configuration files.
