# Heretic-MOE Four-Language Corpus v4 Design

## Goal

Freeze one text-private, four-language corpus for Heretic-MOE with three
canonical-ID-disjoint pools per direction:

- map: 1,000 SAFE and 1,000 UNSAFE rows per language;
- trial: 400 SAFE and 400 UNSAFE rows per language;
- final: 200 SAFE and 200 UNSAFE rows per language.

The frozen language order is en, ru, zh, ja. Every translation family has
identical canonical-ID membership and order in all four languages.

## Source selection

SAFE trial and final come from the completed 912-row PolyGuard SAFE consensus.
The first 400 rows by weakest-language DIRECT rate form trial; the next 200
form final. Their honest source category is polyguard_benign.

SAFE map uses 1,000 rows from the older 1,400-row frozen SAFE corpus. Selection
is deterministic and category-stratified across its 19 native
GEN/XSTest/benign categories. The remaining 400 rows stay in reserve.

UNSAFE uses the completed 1,664-row four-language consensus. Within each
C01-C14 category, rows are ranked by weakest-language HARD_REFUSE rate, global
HARD_REFUSE rate, SOFT/PARTIAL diagnostics, invalid count, and canonical ID.
Category-proportional allocation produces trial 400, final 200, map 1,000, and
reserve 64. Map deliberately retains partial, soft, and language-disagreement
cases to maximize geometric coverage.

No external rows are required: the available unique counts cover all requested
cells.

## Files

Root:
F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4/

Each pool contains:

- {pool}_{language}_{direction}_{count}.jsonl;
- categories/{pool}/{direction}/{category}.jsonl;
- a text-free membership manifest with counts, order hashes, source hashes,
  category allocations, quality summaries, and overlap checks.

The root manifest proves:

- exactly 6,400 SAFE cells and 6,400 UNSAFE cells;
- zero canonical-ID overlap among map/trial/final within each direction;
- exact four-language order agreement;
- exact source-row preservation;
- no prompt or response text in public reports.

## Heretic integration

The public YAML data contract gains:

~~~yaml
data:
  dataset_root: F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4
  split_root: F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4
  languages: [en, ru, zh, ja]
  direction_rows_per_cell: 1000
  trial_rows_per_cell: 400
  final_rows_per_cell: 200
~~~

Heretic loads:

- map files as the direction/residual geometry pool;
- trial files as the rotating per-trial evaluation pool;
- final files as the independent finalist holdout.

The final holdout becomes bidirectional. UNSAFE rows measure SRG/R-side and
unsafe geometry gain; SAFE rows measure preservation geometry and are reported
separately. Public output remains text-free, while private response archives
retain the same access and hashing rules as existing finalists.

Dataset and run-contract schema versions are bumped so old 5-language/132-row
journals cannot resume under the v4 corpus.

## Verification

Before a real search:

1. mechanically validate every file, hash, count, category allocation, and
   disjointness rule;
2. run focused loader/config/final-holdout tests;
3. run the complete CPU test suite;
4. run a two-GPU preparation/smoke pipeline on an available 8B or 9B model;
5. confirm map, trial, and final phases consume 8,000, 3,200, and 1,600 rows
   respectively and emit only text-free public artifacts.

Balanced and Max exports will later be re-evaluated against the exact same
frozen final 200+200 x 4 holdout for direct comparison.
