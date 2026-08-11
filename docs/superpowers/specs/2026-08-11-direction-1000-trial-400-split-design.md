# Canonical 1000/400 multilingual split

## Goal

Materialize deterministic operative subsets from each frozen 1,400-row
multilingual direction corpus without changing prompt text, canonical IDs,
source order, or any existing source artifact:

- 1,000 canonical rows per direction for direction construction;
- 400 canonical rows per direction for trial evaluation;
- identical canonical membership and order in EN, RU, ZH, ES, and FR;
- 400 rows equal to exactly `400 / 1400 = 2 / 7` of the corpus, with the
  closest possible integer allocation inside every category.

The source JSONL files and their manifest remain immutable. Generated files
are derivative artifacts in a separate versioned output directory.

## Inputs

The splitter reads the ten frozen full files under the selected corpus root:

```text
direction_{en,ru,zh,es,fr}_{safe,unsafe}.jsonl
```

Each direction must contain exactly 1,400 unique canonical IDs. Every language
must have the same canonical-ID sequence, category ID, direction class, and
source-family metadata for a corresponding canonical row. Prompt text is
treated as an opaque field: it is copied byte-for-byte and is never printed,
logged, hashed into a public per-row report, or otherwise inspected by the
split-selection logic.

## Allocation

SAFE and UNSAFE are allocated independently because their category taxonomies
are not crossed or interchangeable.

For category `c` with `n_c` canonical rows, the ideal trial quota is:

```text
q_c = n_c * 400 / 1400 = n_c * 2 / 7
```

The integer category quotas use Hamilton largest remainder:

1. assign `floor(q_c)` rows to every category;
2. distribute the remaining seats by descending fractional remainder;
3. break equal remainders by the bytewise category ID;
4. require the final category quotas to sum to exactly 400.

Within a category quota, source-family representation is preserved as closely
as integers allow using a second largest-remainder allocation. The public
source families are `stock` and `authored`; all non-stock provenance values are
grouped into `authored` without modifying their original row metadata.

Specific canonical IDs within each category/source-family cell are selected by
a stable SHA-256 order over:

```text
schema_version | seed | direction | category | source_family | canonical_id
```

The seed is recorded in the manifest. Selected IDs form the 400-row trial
subset. The 1,000-row direction-fit subset is the exact complement. Both output
files retain the source corpus order rather than hash-selection order.

## Multilingual alignment

Selection is performed once on canonical metadata and then propagated to all
five languages. A canonical ID cannot be assigned differently by language.

For every direction and language:

- direction-fit contains exactly 1,000 rows;
- trial contains exactly 400 rows;
- overlap is zero;
- union is the original 1,400 canonical sequence;
- each output row remains byte-identical to the corresponding source JSONL
  line;
- category and source-family counts match the frozen allocation manifest.

This split defines membership only. A later trial-panel materializer may choose
one or two aligned language realizations for a canonical ID, but it must consume
these frozen memberships and cannot silently resample them.

## Outputs

The default output is a new directory below the corpus root, for example:

```text
operative_split_1000_400_v1/
  direction_fit_{lang}_{safe,unsafe}.jsonl
  trial_{lang}_{safe,unsafe}.jsonl
  canonical_membership_{safe,unsafe}.json
  manifest.json
  verify_report.json
  audit_contract.json
```

`manifest.json` records input and output SHA-256 values, schema version, seed,
algorithm, exact category/source allocations, row counts, and canonical-ID-set
hashes. `verify_report.json` and console output are text-free. The audit contract
contains paths, hashes, counts, and required PASS/FAIL gates for independent
review; it contains no prompt or response text.

## Verification gates

Materialization fails unless all gates pass:

- ten required full input files exist;
- exactly 1,400 rows and canonical IDs per input;
- exact canonical order/alignment across five languages;
- allowed direction/language metadata agree with filenames;
- no empty canonical ID, row ID, category ID, or prompt field;
- exactly 1,000 direction-fit and 400 trial rows per file;
- exact zero overlap and exact 1,400-row reconstruction;
- byte-identity of every emitted row against its source line;
- identical membership by canonical ID across all languages;
- category quota deviation from `2/7` is less than one row;
- every recorded SHA-256 verifies after files are closed.

The script writes into a temporary sibling directory and publishes the final
directory atomically only after all gates pass. Existing output is never
overwritten implicitly.

## Testing

Tests use synthetic metadata-only fixtures and never require corpus text. They
cover:

- exact 1,000/400 counts and complement reconstruction;
- largest-remainder category allocation;
- nested stock/authored allocation;
- deterministic output for a fixed seed;
- identical canonical membership in all languages;
- source-order and byte preservation;
- rejection of misordered translations, duplicate IDs, malformed rows,
  unexpected counts, overlap, and an existing output directory;
- text-free manifest, verify report, and console contract.
