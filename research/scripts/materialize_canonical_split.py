#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heretic.canonical_category_split import materialize_category_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create aligned category-stratified direction/trial JSONL subsets."
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--languages", default="en,ru,zh,es,fr")
    parser.add_argument("--trial-rows", default=400, type=int)
    parser.add_argument("--seed", default=20260811, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    manifest = materialize_category_split(
        args.corpus_root.resolve(),
        args.output_dir.resolve(),
        languages=languages,
        selected_rows=args.trial_rows,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "direction_rows": manifest["direction_rows"],
                "files": len(manifest["files"]),
                "output_dir": str(args.output_dir.resolve()),
                "status": manifest["status"],
                "trial_rows": manifest["selected_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
