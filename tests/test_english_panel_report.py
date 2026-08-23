from __future__ import annotations

import hashlib
import json

from heretic.english_panel_report import build_english_panel_report


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_panel_report_combines_original_ten_and_three_new_votes(tmp_path) -> None:
    """Catches dropping invalid votes or replacing the frozen original ten votes."""
    original = tmp_path / "original.jsonl"
    write_jsonl(
        original,
        [
            {
                "canonical_id": "U1",
                "row_id": "EN-U1",
                "valid_model_votes": 10,
                "model_vote_counts": {"SOFT": 7, "HARD_REFUSE": 2, "DIRECT": 1},
                "category_ids": ["C01"],
                "source": "source-a",
            },
            {
                "canonical_id": "U2",
                "row_id": "EN-U2",
                "valid_model_votes": 10,
                "model_vote_counts": {"SOFT": 1, "HARD_REFUSE": 8, "DIRECT": 1},
                "category_ids": ["C02"],
                "source": "source-b",
            },
        ],
    )
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "canonical_id": "U1",
                "row_id": "EN-U1",
                "target_behavior_class": "SOFT",
                "soft_votes": 7,
                "hard_votes": 2,
                "valid_model_votes": 10,
                "panel_size": 10,
                "category_ids": ["C01"],
                "source": "source-a",
            },
            {
                "canonical_id": "U2",
                "row_id": "EN-U2",
                "target_behavior_class": "HARD_REFUSE",
                "soft_votes": 1,
                "hard_votes": 8,
                "valid_model_votes": 10,
                "panel_size": 10,
                "category_ids": ["C02"],
                "source": "source-b",
            },
        ],
    )
    model_rows = {
        "model-a": [("U1", "SOFT", True), ("U2", "HARD_REFUSE", True)],
        "model-b": [("U1", "HARD_REFUSE", True), ("U2", "HARD_REFUSE", True)],
        "model-c": [("U1", "SOFT", True), ("U2", None, False)],
    }
    result_dirs = []
    for model_id, values in model_rows.items():
        result_dir = tmp_path / model_id
        rows_path = result_dir / "rows.jsonl"
        write_jsonl(
            rows_path,
            [
                {
                    "canonical_id": canonical_id,
                    "row_id": f"EN-{canonical_id}",
                    "model_id": model_id,
                    "language": "en",
                    "variant": "code_permuted",
                    "classification": classification,
                    "valid": valid,
                    "category_ids": ["C01"],
                    "direction_class": "unsafe",
                    "output_tokens": 1,
                    "output_shape": "code",
                }
                for canonical_id, classification, valid in values
            ],
        )
        (result_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "model_id": model_id,
                    "rows": 2,
                    "valid": sum(valid for _, _, valid in values),
                    "result_sha256": sha256(rows_path),
                    "variants": ["code_permuted"],
                    "system_mode": "english",
                }
            ),
            encoding="utf-8",
        )
        result_dirs.append(result_dir)

    manifest = build_english_panel_report(
        original,
        labels,
        result_dirs,
        tmp_path / "report",
        original_panel_size=10,
    )

    assert manifest["status"] == "PASS"
    assert manifest["counts"] == {"rows": 2, "SOFT": 1, "HARD_REFUSE": 1}
    combined = [
        json.loads(line)
        for line in (tmp_path / "report" / "combined_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert combined[0]["combined_vote_counts"] == {
        "SOFT": 9,
        "HARD_REFUSE": 3,
        "DIRECT": 1,
        "PARTIAL": 0,
        "INVALID": 0,
    }
    assert combined[0]["target_support"] == 9
    assert combined[1]["combined_vote_counts"] == {
        "SOFT": 1,
        "HARD_REFUSE": 10,
        "DIRECT": 1,
        "PARTIAL": 0,
        "INVALID": 1,
    }
    assert combined[1]["target_support"] == 10
    summary = json.loads(
        (tmp_path / "report" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["target_support_distribution"] == {"9": 1, "10": 1}
    assert "prompt" not in (tmp_path / "report" / "combined_rows.jsonl").read_text(
        encoding="utf-8"
    )
