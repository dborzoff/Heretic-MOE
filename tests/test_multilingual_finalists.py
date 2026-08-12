from __future__ import annotations

import json
from pathlib import Path

import pytest

from heretic.multilingual_finalists import (
    freeze_top_six_manifest,
    select_multilingual_winners,
    select_top_six,
)


def _candidate(number: int, removal: float, loss: float, cost: float) -> dict:
    return {
        "trial_number": number,
        "trial_index": number + 1,
        "params": {"x": float(number)},
        "params_sha256": f"{number:064x}",
        "removal": removal,
        "preservation_loss": loss,
        "cost_up": cost,
        "feasible": True,
        "complete": True,
    }


def test_top_six_combines_removal_cost_and_preservation_extremes() -> None:
    candidates = [
        _candidate(0, 0.95, 0.40, 0.70),
        _candidate(1, 0.90, 0.35, 0.72),
        _candidate(2, 0.80, 0.12, 0.88),
        _candidate(3, 0.78, 0.10, 0.89),
        _candidate(4, 0.70, 0.04, 0.86),
        _candidate(5, 0.68, 0.03, 0.84),
        _candidate(6, 0.20, 0.01, 0.60),
    ]

    selected = select_top_six(candidates, top_n=6)

    assert len(selected) == 6
    assert {row["trial_number"] for row in selected} == {0, 1, 2, 3, 4, 5}
    assert {role for row in selected for role in row["shortlist_roles"]} >= {
        "max_removal",
        "max_cost",
        "preservation_front",
    }


def test_top_six_deduplicates_identical_parameter_sets() -> None:
    candidates = [_candidate(i, 1.0 - i * 0.05, 0.1 + i * 0.01, 0.9) for i in range(7)]
    candidates[1]["params_sha256"] = candidates[0]["params_sha256"]

    selected = select_top_six(candidates, top_n=6)

    assert len(selected) == 6
    assert sum(row["params_sha256"] == candidates[0]["params_sha256"] for row in selected) == 1


def test_top_six_manifest_is_frozen_before_recheck(tmp_path: Path) -> None:
    selected = select_top_six(
        [_candidate(i, 1.0 - i * 0.05, 0.1 + i * 0.01, 0.9 - i * 0.01) for i in range(6)],
        top_n=6,
    )

    manifest = freeze_top_six_manifest(
        tmp_path / "top6_manifest.json",
        selected,
        source_journal_sha256="a" * 64,
        final_holdout_sha256="b" * 64,
    )

    assert manifest["status"] == "FROZEN"
    assert manifest["top_n"] == 6
    assert freeze_top_six_manifest(
        tmp_path / "top6_manifest.json",
        selected,
        source_journal_sha256="a" * 64,
        final_holdout_sha256="b" * 64,
    ) == manifest
    assert "prompt" not in json.dumps(manifest).lower()


def test_balanced_and_max_use_only_full_recheck_metrics() -> None:
    rows = [
        {
            "source_trial_number": 10,
            "source_trial_index": 11,
            "params_sha256": "1" * 64,
            "feasible": True,
            "removal": 0.82,
            "preservation_loss": 0.06,
            "safe_ppl_drift": 0.002,
            "safe_geometry_damage": 0.03,
            "worst_language": 0.60,
            "worst_category": 0.55,
            "final_holdout_removal": 0.70,
        },
        {
            "source_trial_number": 20,
            "source_trial_index": 21,
            "params_sha256": "2" * 64,
            "feasible": True,
            "removal": 0.94,
            "preservation_loss": 0.12,
            "safe_ppl_drift": 0.004,
            "safe_geometry_damage": 0.08,
            "worst_language": 0.72,
            "worst_category": 0.68,
            "final_holdout_removal": 0.91,
        },
        {
            "source_trial_number": 30,
            "source_trial_index": 31,
            "params_sha256": "3" * 64,
            "feasible": False,
            "removal": 0.99,
            "preservation_loss": 0.01,
            "safe_ppl_drift": 0.001,
            "safe_geometry_damage": 0.01,
            "worst_language": 0.90,
            "worst_category": 0.90,
            "final_holdout_removal": 0.99,
        },
    ]

    report = select_multilingual_winners(rows, balanced_removal_fraction=0.80)

    assert report["winners"]["Balanced"]["source_trial_number"] == 10
    assert report["winners"]["Max"]["source_trial_number"] == 20
    assert report["winners_distinct"] is True


def test_balanced_removal_gate_remains_reachable_when_all_scores_are_negative() -> None:
    rows = []
    for number, removal, loss in (
        (10, -0.30, 0.20),
        (20, -0.36, 0.10),
        (30, -0.55, 0.01),
    ):
        rows.append(
            {
                "source_trial_number": number,
                "source_trial_index": number + 1,
                "params_sha256": f"{number:064x}",
                "feasible": True,
                "removal": removal,
                "preservation_loss": loss,
                "safe_ppl_drift": loss,
                "safe_geometry_damage": loss,
                "worst_language": removal,
                "worst_category": removal,
                "final_holdout_removal": removal,
            }
        )

    report = select_multilingual_winners(rows, balanced_removal_fraction=0.80)

    assert report["resolved_balanced_removal_gate"] == pytest.approx(-0.375)
    assert report["winners"]["Balanced"]["source_trial_number"] == 20
    assert report["winners"]["Max"]["source_trial_number"] == 10
