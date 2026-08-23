from __future__ import annotations

import hashlib
import json

from heretic.english_consensus_filter import (
    materialize_cumulative_vote_pool,
    materialize_panel_support_pool,
)


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_cumulative_vote_pool_selects_soft_or_hard_at_threshold(
    tmp_path,
) -> None:
    """Catches treating exact 7/10 as the pool instead of cumulative >=7/10."""
    source = tmp_path / "source"
    safe = source / "direction_en_safe_strict.jsonl"
    unsafe = source / "direction_en_unsafe_strict.jsonl"
    write_jsonl(safe, [])
    write_jsonl(
        unsafe,
        [
            {
                "canonical_id": canonical_id,
                "row_id": f"EN-{canonical_id}",
                "language": "en",
                "direction_class": "unsafe",
                "category_ids": ["C01"],
                "source": "fixture",
                "prompt": f"PRIVATE-{canonical_id}",
            }
            for canonical_id in ("U1", "U2", "U3", "U4")
        ],
    )
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": ["en"],
                "directions": {"safe": 0, "unsafe": 4},
                "files": [
                    {
                        "language": "en",
                        "direction": "safe",
                        "path": safe.name,
                        "rows": 0,
                        "sha256": sha256(safe),
                    },
                    {
                        "language": "en",
                        "direction": "unsafe",
                        "path": unsafe.name,
                        "rows": 4,
                        "sha256": sha256(unsafe),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    consensus = tmp_path / "consensus.jsonl"
    write_jsonl(
        consensus,
        [
            {
                "canonical_id": "U1",
                "row_id": "EN-U1",
                "valid_model_votes": 10,
                "model_vote_counts": {"SOFT": 7, "HARD_REFUSE": 2, "DIRECT": 1},
                "category_ids": ["C01"],
                "source": "fixture",
            },
            {
                "canonical_id": "U2",
                "row_id": "EN-U2",
                "valid_model_votes": 10,
                "model_vote_counts": {"SOFT": 1, "HARD_REFUSE": 8, "DIRECT": 1},
                "category_ids": ["C01"],
                "source": "fixture",
            },
            {
                "canonical_id": "U3",
                "row_id": "EN-U3",
                "valid_model_votes": 10,
                "model_vote_counts": {"SOFT": 6, "HARD_REFUSE": 4},
                "category_ids": ["C01"],
                "source": "fixture",
            },
            {
                "canonical_id": "U4",
                "row_id": "EN-U4",
                "valid_model_votes": 9,
                "model_vote_counts": {"SOFT": 5, "HARD_REFUSE": 4, "INVALID": 1},
                "category_ids": ["C01"],
                "source": "fixture",
            },
        ],
    )
    output = tmp_path / "pool"

    result = materialize_cumulative_vote_pool(
        manifest,
        consensus,
        output,
        minimum_votes=7,
        panel_size=10,
    )

    assert result["status"] == "PASS"
    assert result["counts"] == {
        "source": 4,
        "selected": 2,
        "rejected": 2,
        "SOFT": 1,
        "HARD_REFUSE": 1,
    }
    selected = [
        json.loads(line)
        for line in (output / "direction_en_unsafe_strict.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["canonical_id"] for row in selected] == ["U1", "U2"]
    labels = (output / "selection_labels.jsonl").read_text(encoding="utf-8")
    assert "PRIVATE" not in labels
    assert (output / "manifest.json").is_file()


def test_materialize_panel_support_pool_uses_cumulative_13_model_support(
    tmp_path,
) -> None:
    """Catches selecting exact 11/13 instead of target support at least 11/13."""
    source = tmp_path / "source"
    safe = source / "direction_en_safe_strict.jsonl"
    unsafe = source / "direction_en_unsafe_strict.jsonl"
    write_jsonl(safe, [])
    write_jsonl(
        unsafe,
        [
            {
                "canonical_id": canonical_id,
                "row_id": f"EN-{canonical_id}",
                "language": "en",
                "direction_class": "unsafe",
                "category_ids": ["C01"],
                "source": "fixture",
                "prompt": f"PRIVATE-{canonical_id}",
            }
            for canonical_id in ("U1", "U2", "U3")
        ],
    )
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": ["en"],
                "directions": {"safe": 0, "unsafe": 3},
                "files": [
                    {
                        "language": "en",
                        "direction": "safe",
                        "path": safe.name,
                        "rows": 0,
                        "sha256": sha256(safe),
                    },
                    {
                        "language": "en",
                        "direction": "unsafe",
                        "path": unsafe.name,
                        "rows": 3,
                        "sha256": sha256(unsafe),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    combined = tmp_path / "combined.jsonl"
    write_jsonl(
        combined,
        [
            {
                "canonical_id": "U1",
                "row_id": "EN-U1",
                "target_behavior_class": "SOFT",
                "target_support": 11,
                "panel_size": 13,
                "combined_vote_counts": {"SOFT": 11, "HARD_REFUSE": 2},
                "additional_votes": {"a": "SOFT", "b": "SOFT", "c": "SOFT"},
                "category_ids": ["C01"],
                "source": "fixture",
            },
            {
                "canonical_id": "U2",
                "row_id": "EN-U2",
                "target_behavior_class": "SOFT",
                "target_support": 10,
                "panel_size": 13,
                "combined_vote_counts": {"SOFT": 10, "HARD_REFUSE": 3},
                "additional_votes": {"a": "SOFT", "b": "SOFT", "c": "HARD_REFUSE"},
                "category_ids": ["C01"],
                "source": "fixture",
            },
            {
                "canonical_id": "U3",
                "row_id": "EN-U3",
                "target_behavior_class": "HARD_REFUSE",
                "target_support": 13,
                "panel_size": 13,
                "combined_vote_counts": {"SOFT": 0, "HARD_REFUSE": 13},
                "additional_votes": {
                    "a": "HARD_REFUSE",
                    "b": "HARD_REFUSE",
                    "c": "HARD_REFUSE",
                },
                "category_ids": ["C01"],
                "source": "fixture",
            },
        ],
    )

    result = materialize_panel_support_pool(
        manifest,
        combined,
        tmp_path / "selected",
        minimum_support=11,
        panel_size=13,
    )

    assert result["counts"] == {
        "source": 3,
        "selected": 2,
        "rejected": 1,
        "SOFT": 1,
        "HARD_REFUSE": 1,
    }
    selected = [
        json.loads(line)
        for line in (tmp_path / "selected" / "selection_labels.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["canonical_id"] for row in selected] == ["U1", "U3"]
    assert "PRIVATE" not in json.dumps(selected)
