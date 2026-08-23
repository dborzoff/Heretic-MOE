from __future__ import annotations

import json

import pytest

from heretic.multilingual_candidate_consensus import (
    build_candidate_consensus,
    build_invalid_retry_plan,
    write_candidate_consensus_artifacts,
)

MODELS = tuple(f"model-{index}" for index in range(10))
LANGUAGES = ("en", "ru", "zh", "ja")
VARIANTS = tuple(f"variant-{index}" for index in range(8))


def result_rows(
    canonical_id: str,
    labels_by_model_language: dict[tuple[str, str], tuple[str | None, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id in MODELS:
        for language in LANGUAGES:
            labels = labels_by_model_language[(model_id, language)]
            assert len(labels) == len(VARIANTS)
            for variant, label in zip(VARIANTS, labels, strict=True):
                rows.append(
                    {
                        "canonical_id": canonical_id,
                        "row_id": f"{language}-{canonical_id}",
                        "model_id": model_id,
                        "language": language,
                        "variant": variant,
                        "classification": label,
                        "valid": label is not None,
                        "category_ids": ["C01"],
                    }
                )
    return rows


def target(canonical_id: str, target_class: str) -> dict[str, object]:
    return {
        "canonical_id": canonical_id,
        "target_behavior_class": target_class,
        "category_ids": ["C01"],
        "candidate_group": "synthetic",
        "source": "fixture",
    }


def test_clean_candidate_requires_six_of_eight_and_two_confirmed_languages() -> None:
    """Catches treating 5/8 as stable or requiring strict 8/8."""
    hard = "HARD_REFUSE"
    soft = "SOFT"
    labels: dict[tuple[str, str], tuple[str | None, ...]] = {}
    for model_id in MODELS:
        labels[(model_id, "en")] = (hard,) * 6 + (soft,) * 2
        labels[(model_id, "ru")] = (hard,) * 6 + (None,) * 2
        labels[(model_id, "zh")] = (hard,) * 5 + (soft,) * 3
        labels[(model_id, "ja")] = (soft,) * 6 + (hard,) * 2

    report = build_candidate_consensus(
        result_rows("U0001", labels),
        [target("U0001", hard)],
        model_ids=MODELS,
        languages=LANGUAGES,
        variants=VARIANTS,
    )

    item = report["candidates"][0]
    assert item["clean"] is True
    assert item["confirmed_languages"] == ["en", "ru"]
    assert item["target_model_support"] == {"en": 10, "ru": 10, "zh": 0, "ja": 0}
    assert item["invalid_votes"] == {"en": 0, "ru": 20, "zh": 0, "ja": 0}
    assert report["summary"]["clean_by_target"] == {"HARD_REFUSE": 1, "SOFT": 0}
    assert len(report["model_language_cells"]) == 40
    en_cell = next(
        cell
        for cell in report["model_language_cells"]
        if cell["model_id"] == "model-0" and cell["language"] == "en"
    )
    assert en_cell == {
        "canonical_id": "U0001",
        "model_id": "model-0",
        "language": "en",
        "stable_label": "HARD_REFUSE",
        "stable_support": 6,
        "valid_votes": 8,
        "invalid_votes": 0,
        "target_match": True,
    }
    ru_cell = next(
        cell
        for cell in report["model_language_cells"]
        if cell["model_id"] == "model-0" and cell["language"] == "ru"
    )
    assert ru_cell["valid_votes"] == 6
    assert ru_cell["invalid_votes"] == 2
    matrix_row = next(
        row
        for row in report["model_category_language_matrix"]
        if row["model_id"] == "model-0"
        and row["category_id"] == "C01"
        and row["language"] == "en"
    )
    assert matrix_row == {
        "model_id": "model-0",
        "category_id": "C01",
        "language": "en",
        "candidates": 1,
        "target_matches": 1,
        "unstable": 0,
        "invalid_votes": 0,
        "HARD_REFUSE": 1,
        "SOFT": 0,
        "PARTIAL": 0,
        "DIRECT": 0,
    }
    assert report["summary"]["clean_by_source"] == {"fixture": 1}
    assert report["category_summary"] == [
        {
            "category_id": "C01",
            "target_behavior_class": "HARD_REFUSE",
            "candidates": 1,
            "clean": 1,
        }
    ]
    assert report["source_summary"] == [
        {
            "source": "fixture",
            "target_behavior_class": "HARD_REFUSE",
            "candidates": 1,
            "clean": 1,
        }
    ]
    en_language = next(
        row
        for row in report["language_summary"]
        if row["language"] == "en"
        and row["target_behavior_class"] == "HARD_REFUSE"
    )
    assert en_language == {
        "language": "en",
        "target_behavior_class": "HARD_REFUSE",
        "candidates": 1,
        "confirmed": 1,
        "model_support": 10,
    }
    model_summary = report["model_vote_summary"][0]
    assert model_summary == {
        "model_id": "model-0",
        "rows": 32,
        "valid": 30,
        "invalid": 2,
        "class_counts": {
            "HARD_REFUSE": 19,
            "SOFT": 11,
            "PARTIAL": 0,
            "DIRECT": 0,
        },
        "language_counts": {"en": 8, "ru": 8, "zh": 8, "ja": 8},
        "variant_counts": {variant: 4 for variant in VARIANTS},
    }


def test_candidate_with_only_one_language_at_six_of_ten_is_not_clean() -> None:
    """Catches accepting a family after only one language passes the model gate."""
    hard = "HARD_REFUSE"
    soft = "SOFT"
    labels: dict[tuple[str, str], tuple[str | None, ...]] = {}
    for index, model_id in enumerate(MODELS):
        labels[(model_id, "en")] = ((hard,) * 6 + (soft,) * 2) if index < 6 else (soft,) * 8
        labels[(model_id, "ru")] = ((hard,) * 6 + (soft,) * 2) if index < 5 else (soft,) * 8
        labels[(model_id, "zh")] = (soft,) * 8
        labels[(model_id, "ja")] = (soft,) * 8

    report = build_candidate_consensus(
        result_rows("U0002", labels),
        [target("U0002", hard)],
        model_ids=MODELS,
        languages=LANGUAGES,
        variants=VARIANTS,
    )

    item = report["candidates"][0]
    assert item["clean"] is False
    assert item["confirmed_languages"] == ["en"]
    assert item["target_model_support"] == {"en": 6, "ru": 5, "zh": 0, "ja": 0}
    assert item["decision_reason"] == "fewer_than_2_confirmed_languages"


def test_duplicate_model_row_variant_is_rejected() -> None:
    """Catches double-counting an appended/resumed result row."""
    labels = {
        (model_id, language): ("SOFT",) * 8
        for model_id in MODELS
        for language in LANGUAGES
    }
    rows = result_rows("U0003", labels)
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate result key"):
        build_candidate_consensus(
            rows,
            [target("U0003", "SOFT")],
            model_ids=MODELS,
            languages=LANGUAGES,
            variants=VARIANTS,
        )


def test_retry_plan_only_selects_unstable_cells_with_invalid_votes() -> None:
    """Catches retrying already stable cells or ignoring a recoverable 5/8 cell."""
    labels = {
        (model_id, language): ("SOFT",) * 8
        for model_id in MODELS
        for language in LANGUAGES
    }
    labels[("model-0", "ru")] = (
        "HARD_REFUSE",
    ) * 5 + ("SOFT",) * 2 + (None,)
    labels[("model-0", "en")] = ("SOFT",) * 6 + (None,) * 2

    plan = build_invalid_retry_plan(
        result_rows("U0004", labels),
        model_ids=MODELS,
        languages=LANGUAGES,
        variants=VARIANTS,
    )

    assert plan == [
        {
            "canonical_id": "U0004",
            "row_id": "ru-U0004",
            "model_id": "model-0",
            "language": "ru",
            "replaces_variant": "variant-7",
        }
    ]


def test_valid_retry_replaces_one_invalid_vote_before_six_of_eight_gate() -> None:
    """Catches counting a retry as a ninth vote instead of replacing one invalid."""
    hard = "HARD_REFUSE"
    soft = "SOFT"
    labels = {
        (model_id, language): (hard,) * 6 + (soft,) * 2
        for model_id in MODELS
        for language in LANGUAGES
    }
    labels[("model-0", "ru")] = (hard,) * 5 + (soft,) * 2 + (None,)
    rows = result_rows("U0005", labels)
    retry_rows = [
        {
            "canonical_id": "U0005",
            "row_id": "ru-U0005",
            "model_id": "model-0",
            "language": "ru",
            "classification": hard,
            "valid": True,
            "replaces_variant": "variant-7",
        }
    ]

    report = build_candidate_consensus(
        rows,
        [target("U0005", hard)],
        model_ids=MODELS,
        languages=LANGUAGES,
        variants=VARIANTS,
        retry_rows=retry_rows,
    )

    cell = next(
        item
        for item in report["model_language_cells"]
        if item["model_id"] == "model-0" and item["language"] == "ru"
    )
    assert cell["stable_label"] == hard
    assert cell["stable_support"] == 6
    assert cell["valid_votes"] == 8
    assert cell["invalid_votes"] == 0


def test_writer_materializes_text_free_clean_and_rejected_artifacts(tmp_path) -> None:
    """Catches leaking private text or omitting clean/reject ID partitions."""
    labels = {
        (model_id, language): ("SOFT",) * 8
        for model_id in MODELS
        for language in LANGUAGES
    }
    report = build_candidate_consensus(
        result_rows("U0006", labels),
        [target("U0006", "SOFT")],
        model_ids=MODELS,
        languages=LANGUAGES,
        variants=VARIANTS,
    )

    manifest = write_candidate_consensus_artifacts(tmp_path, report)

    assert manifest["status"] == "PASS"
    assert manifest["counts"] == {
        "candidates": 1,
        "clean_hard": 0,
        "clean_soft": 1,
        "rejected": 0,
        "model_language_cells": 40,
        "model_category_language_matrix": 40,
        "category_summary": 1,
        "source_summary": 1,
        "language_summary": 4,
        "model_vote_summary": 10,
    }
    clean_soft = [
        json.loads(line)
        for line in (tmp_path / "clean_soft_ids.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert clean_soft == [{"canonical_id": "U0006"}]
    for name in (
        "candidate_consensus.jsonl",
        "model_language_cells.jsonl",
        "model_category_language_matrix.jsonl",
        "category_summary.jsonl",
        "source_summary.jsonl",
        "language_summary.jsonl",
        "model_vote_summary.jsonl",
        "clean_hard_ids.jsonl",
        "clean_soft_ids.jsonl",
        "rejected_ids.jsonl",
    ):
        for line in (tmp_path / name).read_text(encoding="utf-8").splitlines():
            assert not ({"prompt", "response", "answer"} & json.loads(line).keys())
