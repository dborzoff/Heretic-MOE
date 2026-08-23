from __future__ import annotations

import hashlib
import json

from heretic.language_panel_report import build_language_panel_report


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_language_panel_report_counts_target_and_english_agreement(tmp_path) -> None:
    """Catches mixing target support with per-model agreement to its English vote."""
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "canonical_id": "U1",
                "row_id": "EN-U1",
                "target_behavior_class": "SOFT",
                "category_ids": ["C01"],
                "source": "source-a",
            },
            {
                "canonical_id": "U2",
                "row_id": "EN-U2",
                "target_behavior_class": "HARD_REFUSE",
                "category_ids": ["C02"],
                "source": "source-b",
            },
        ],
    )
    languages = ("ru", "zh", "ja")
    votes = {
        "model-a": {
            "en": ("SOFT", "HARD_REFUSE"),
            "ru": ("SOFT", "HARD_REFUSE"),
            "zh": ("SOFT", "HARD_REFUSE"),
            "ja": ("SOFT", "SOFT"),
        },
        "model-b": {
            "en": ("SOFT", "HARD_REFUSE"),
            "ru": ("SOFT", "HARD_REFUSE"),
            "zh": ("HARD_REFUSE", "HARD_REFUSE"),
            "ja": ("SOFT", "SOFT"),
        },
        "model-c": {
            "en": ("SOFT", "HARD_REFUSE"),
            "ru": ("HARD_REFUSE", "HARD_REFUSE"),
            "zh": ("HARD_REFUSE", "SOFT"),
            "ja": ("SOFT", "HARD_REFUSE"),
        },
    }
    inputs = []
    for model_id, by_language in votes.items():
        english = tmp_path / model_id / "english.jsonl"
        write_jsonl(
            english,
            [
                {
                    "canonical_id": canonical_id,
                    "row_id": f"EN-{canonical_id}",
                    "model_id": model_id,
                    "language": "en",
                    "variant": "code_permuted",
                    "classification": classification,
                    "valid": True,
                }
                for canonical_id, classification in zip(
                    ("U1", "U2"), by_language["en"], strict=True
                )
            ],
        )
        multilingual = tmp_path / model_id / "languages.jsonl"
        write_jsonl(
            multilingual,
            [
                {
                    "canonical_id": canonical_id,
                    "row_id": f"{language.upper()}-{canonical_id}",
                    "model_id": model_id,
                    "language": language,
                    "variant": "code_permuted",
                    "classification": classification,
                    "valid": True,
                }
                for language in languages
                for canonical_id, classification in zip(
                    ("U1", "U2"), by_language[language], strict=True
                )
            ],
        )
        manifest = multilingual.with_suffix(".manifest.json")
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "model_id": model_id,
                    "rows": 6,
                    "result_sha256": sha256(multilingual),
                }
            ),
            encoding="utf-8",
        )
        inputs.append(
            {
                "model_id": model_id,
                "english_result_path": english,
                "language_result_path": multilingual,
                "language_manifest_path": manifest,
            }
        )

    manifest = build_language_panel_report(
        labels,
        inputs,
        tmp_path / "report",
        languages=languages,
        support_threshold=2,
    )

    assert manifest["status"] == "PASS"
    assert manifest["counts"] == {
        "rows": 2,
        "panel_size": 3,
        "support_threshold": 2,
        "confirmed_all_languages": 0,
        "confirmed_at_least_two_languages": 2,
    }
    combined = [
        json.loads(line)
        for line in (tmp_path / "report" / "combined_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert combined[0]["target_support"] == {"ru": 2, "zh": 1, "ja": 3}
    assert combined[0]["confirmed_languages"] == ["ru", "ja"]
    assert combined[1]["target_support"] == {"ru": 3, "zh": 2, "ja": 1}
    summaries = [
        json.loads(line)
        for line in (tmp_path / "report" / "model_language_summary.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    model_a_ru = next(
        row
        for row in summaries
        if row["model_id"] == "model-a" and row["language"] == "ru"
    )
    assert model_a_ru["target_agreement"] == 2
    assert model_a_ru["english_vote_agreement"] == 2
    assert "prompt" not in (tmp_path / "report" / "combined_rows.jsonl").read_text(
        encoding="utf-8"
    )
