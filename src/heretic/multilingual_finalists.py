# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic multilingual TOP-6 and Balanced/Max selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_FORBIDDEN_PUBLIC_KEYS = {"prompt", "response", "answer", "text"}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_text_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public finalist manifest contains forbidden field {key}")
            _assert_text_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_text_free(nested)


def _finite(row: Mapping[str, Any], fields: Iterable[str]) -> bool:
    return all(
        isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
        for field in fields
    )


def _deduplicated(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in candidates
        if row.get("complete") is True
        and row.get("feasible") is True
        and isinstance(row.get("params_sha256"), str)
        and _finite(row, ("removal", "preservation_loss", "cost_up"))
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["cost_up"]),
            -float(row["removal"]),
            float(row["preservation_loss"]),
            int(row["trial_number"]),
        )
    )
    unique: list[dict[str, Any]] = []
    seen_params: set[str] = set()
    for row in eligible:
        params_hash = str(row["params_sha256"])
        if params_hash in seen_params:
            continue
        seen_params.add(params_hash)
        unique.append(row)
    return unique


def select_top_six(
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 6,
    preservation_removal_fraction: float = 0.65,
) -> list[dict[str, Any]]:
    """Build a diverse shortlist from Removal, Cost and preservation fronts."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not 0.0 <= preservation_removal_fraction <= 1.0:
        raise ValueError("preservation_removal_fraction must be in [0,1]")
    unique = _deduplicated(candidates)
    if len(unique) < top_n:
        raise ValueError(f"only {len(unique)} unique feasible candidates for TOP-{top_n}")
    removal_order = sorted(
        unique,
        key=lambda row: (
            -float(row["removal"]),
            float(row["preservation_loss"]),
            int(row["trial_number"]),
        ),
    )
    cost_order = sorted(
        unique,
        key=lambda row: (
            -float(row["cost_up"]),
            -float(row["removal"]),
            float(row["preservation_loss"]),
            int(row["trial_number"]),
        ),
    )
    removal_gate = float(removal_order[0]["removal"]) * preservation_removal_fraction
    preservation_order = sorted(
        (row for row in unique if float(row["removal"]) >= removal_gate),
        key=lambda row: (
            float(row["preservation_loss"]),
            -float(row["removal"]),
            int(row["trial_number"]),
        ),
    )
    role_rows = (
        ("max_removal", removal_order[:2]),
        ("max_cost", cost_order[:2]),
        ("preservation_front", preservation_order[:2]),
    )
    selected_by_hash: dict[str, dict[str, Any]] = {}
    for role, rows in role_rows:
        for source in rows:
            key = str(source["params_sha256"])
            selected = selected_by_hash.setdefault(
                key,
                {**source, "shortlist_roles": []},
            )
            selected["shortlist_roles"].append(role)
    for source in cost_order:
        if len(selected_by_hash) >= top_n:
            break
        key = str(source["params_sha256"])
        if key not in selected_by_hash:
            selected_by_hash[key] = {
                **source,
                "shortlist_roles": ["front_fill"],
            }
    selected = list(selected_by_hash.values())[:top_n]
    if len(selected) != top_n:
        raise ValueError(f"could not materialize TOP-{top_n}")
    for rank, row in enumerate(selected, start=1):
        row["shortlist_rank"] = rank
    return selected


def freeze_top_six_manifest(
    path: str | Path,
    selected: Sequence[Mapping[str, Any]],
    *,
    source_journal_sha256: str,
    final_holdout_sha256: str,
) -> dict[str, Any]:
    """Pin finalist membership before any candidate final-holdout generation."""

    if len(selected) != 6:
        raise ValueError("the multilingual finalist manifest requires exactly TOP-6")
    for name, value in (
        ("source_journal_sha256", source_journal_sha256),
        ("final_holdout_sha256", final_holdout_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    records = [dict(row) for row in selected]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN",
        "top_n": 6,
        "source_journal_sha256": source_journal_sha256,
        "final_holdout_sha256": final_holdout_sha256,
        "selection": records,
    }
    _assert_text_free(manifest)
    manifest["shortlist_contract_sha256"] = _canonical_sha256(manifest)
    destination = Path(path)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if destination.is_file():
        existing = destination.read_text(encoding="utf-8")
        if existing != payload:
            raise ValueError("existing TOP-6 manifest differs")
        return json.loads(existing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return manifest


def select_multilingual_winners(
    measured: Sequence[Mapping[str, Any]],
    *,
    balanced_removal_fraction: float = 0.80,
) -> dict[str, Any]:
    """Choose Balanced and Max using only complete full-recheck records."""

    if not 0.0 <= balanced_removal_fraction <= 1.0:
        raise ValueError("balanced_removal_fraction must be in [0,1]")
    fields = (
        "removal",
        "preservation_loss",
        "safe_ppl_drift",
        "safe_geometry_damage",
        "worst_language",
        "worst_category",
        "final_holdout_removal",
    )
    eligible = [
        dict(row)
        for row in measured
        if row.get("feasible") is True and _finite(row, fields)
    ]
    if not eligible:
        raise ValueError("no feasible complete multilingual recheck record")
    best_removal = max(float(row["removal"]) for row in eligible)
    if balanced_removal_fraction == 0.0:
        balanced_gate = -math.inf
    elif best_removal < 0.0:
        balanced_gate = best_removal / balanced_removal_fraction
    else:
        balanced_gate = best_removal * balanced_removal_fraction
    balanced_pool = [
        row for row in eligible if float(row["removal"]) >= balanced_gate
    ]
    if not balanced_pool:
        raise ValueError("no finalist passes the Balanced removal gate")
    balanced = min(
        balanced_pool,
        key=lambda row: (
            float(row["preservation_loss"]),
            float(row["safe_ppl_drift"]),
            float(row["safe_geometry_damage"]),
            -float(row["worst_language"]),
            -float(row["removal"]),
            int(row["source_trial_number"]),
        ),
    )
    maximum = max(
        eligible,
        key=lambda row: (
            float(row["removal"]),
            float(row["final_holdout_removal"]),
            float(row["worst_language"]),
            float(row["worst_category"]),
            -float(row["preservation_loss"]),
            -int(row["source_trial_number"]),
        ),
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "contract": "multilingual_v3_full_recheck",
        "balanced_removal_fraction": balanced_removal_fraction,
        "resolved_balanced_removal_gate": balanced_gate,
        "measured": sorted(
            (dict(row) for row in measured),
            key=lambda row: int(row["source_trial_number"]),
        ),
        "winners": {"Balanced": balanced, "Max": maximum},
        "winners_distinct": int(balanced["source_trial_number"])
        != int(maximum["source_trial_number"]),
    }
    _assert_text_free(report)
    return report
