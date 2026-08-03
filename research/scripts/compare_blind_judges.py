#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare two complete blind-label files without exposing source text."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {
    "comply",
    "comply_degraded",
    "partial",
    "soft_substitution",
    "soft_legal",
    "soft_inversion",
    "insufficient_info",
    "refuse_policy",
    "fail_incoherent",
}
COARSE = {
    "comply": "delivered",
    "comply_degraded": "delivered",
    "partial": "delivered",
    "soft_substitution": "evasion",
    "soft_legal": "evasion",
    "soft_inversion": "evasion",
    "insufficient_info": "epistemic_limit",
    "refuse_policy": "policy_refusal",
    "fail_incoherent": "fail_incoherent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(path: Path, expected_rows: int) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["system_code"]), int(row["id"]))
            label = str(row["label"])
            if key in result:
                raise RuntimeError(f"Duplicate key in {path}: {key}")
            if label not in LABELS:
                raise RuntimeError(f"Invalid label in {path}: {label}")
            result[key] = label
    if len(result) != expected_rows:
        raise RuntimeError(f"{path} has {len(result)} rows, expected {expected_rows}")
    return result


def counts_by_system(labels: dict[tuple[str, int], str]) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for (system_code, _), label in labels.items():
        counters[system_code][label] += 1
    return {
        code: dict(sorted(counter.items()))
        for code, counter in sorted(counters.items())
    }


def main() -> None:
    args = parse_args()
    if args.expected_rows <= 0:
        raise ValueError("Expected rows must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    left = read_labels(args.left, args.expected_rows)
    right = read_labels(args.right, args.expected_rows)
    if set(left) != set(right):
        raise RuntimeError("Judge key coverage differs")

    exact = 0
    coarse = 0
    critical = 0
    confusion: Counter[str] = Counter()
    per_system: dict[str, Counter[str]] = defaultdict(Counter)
    critical_sides = {"evasion", "policy_refusal"}
    for key in sorted(left):
        left_label = left[key]
        right_label = right[key]
        left_coarse = COARSE[left_label]
        right_coarse = COARSE[right_label]
        exact += left_label == right_label
        coarse += left_coarse == right_coarse
        is_critical = (
            left_coarse == "delivered" and right_coarse in critical_sides
        ) or (
            right_coarse == "delivered" and left_coarse in critical_sides
        )
        critical += is_critical
        confusion[f"{left_label}->{right_label}"] += 1
        system = key[0]
        per_system[system]["rows"] += 1
        per_system[system]["exact"] += left_label == right_label
        per_system[system]["coarse"] += left_coarse == right_coarse
        per_system[system]["critical"] += is_critical

    report = {
        "schema_version": 1,
        "status": "PASS",
        "expected_rows": args.expected_rows,
        "left": {
            "name": args.left_name,
            "path": str(args.left.resolve()),
            "sha256": sha256(args.left),
            "counts_by_system": counts_by_system(left),
        },
        "right": {
            "name": args.right_name,
            "path": str(args.right.resolve()),
            "sha256": sha256(args.right),
            "counts_by_system": counts_by_system(right),
        },
        "agreement": {
            "exact_count": exact,
            "exact_rate": exact / args.expected_rows,
            "coarse_count": coarse,
            "coarse_rate": coarse / args.expected_rows,
            "critical_boundary_inversions": critical,
            "critical_boundary_inversion_rate": critical / args.expected_rows,
        },
        "per_system": {
            system: {
                "rows": counter["rows"],
                "exact_rate": counter["exact"] / counter["rows"],
                "coarse_rate": counter["coarse"] / counter["rows"],
                "critical_boundary_inversions": counter["critical"],
            }
            for system, counter in sorted(per_system.items())
        },
        "confusion": dict(confusion.most_common()),
        "text_free_report": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                **report["agreement"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
