# Recheck-Only Post-Search Mode

## Goal

Allow Heretic-MOE to finish an optimization journal, select six diverse finalists, and run the existing high-fidelity 64x1024 recheck without assembling or exporting model weights.

## User-visible contract

- `--recheck-only` is a post-search mode separate from `--finalize` and `--search-only`.
- The default finalist count is six.
- A completed journal may be reopened with the same target trial count; search is skipped and only the missing recheck stage runs.
- The recheck writes the finalist manifest, measurements, winner selection, hashes, and Balanced/Max decision.
- No model directory, assembled weights, GGUF, or release export is created in this mode.
- A later normal finalization reuses the preserved recheck artifacts and performs only the export work that is still missing.

## Modes

| Mode | Search | Recheck | Export |
|---|---:|---:|---:|
| default / `--finalize` | yes | yes | yes |
| `--recheck-only` | yes | yes | no |
| `--search-only` / `--no-finalize` | yes | no | no |

The three modes are mutually exclusive. Existing command lines remain valid.

## Implementation boundary

The existing finalization path already owns candidate preparation, high-fidelity measurement, winner selection, and export. It will receive an explicit `export_models` decision and return immediately after a validated `winners.json` when export is disabled. The recheck contract itself remains identical so that later export can reuse it.

## State and recovery

The run manifest records the post-search mode and ends with `recheck_complete` after a successful recheck-only run. Existing immutable finalist artifacts are validated before reuse. Search journals are never rewritten merely to perform a recheck.

## Verification

- CLI tests cover all three modes, mutual exclusion, and the default TOP 6.
- Controller tests cover recheck-only dispatch and final status.
- A dry-run demonstrates that recheck is planned while export is disabled.
- The current 600-trial run is allowed to finish unchanged, then is resumed at target 600 with `--recheck-only --finalist-top-n 6 --recheck-ppl-chunks 64 --recheck-ppl-window 1024`.
