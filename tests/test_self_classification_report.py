from __future__ import annotations

import json
from pathlib import Path

from heretic.self_classification import PromptVariant
from heretic.self_classification_report import (
    build_category_language_tables,
    build_consensus_summary,
    build_language_category_agreement,
    build_model_summary,
    build_vote4_summary,
    build_vote4_transition_summary,
    select_prompt_variant,
    write_consensus_reports,
    write_model_reports,
    write_pilot_reports,
    write_vote4_reports,
)


def row(
    *,
    row_id: str,
    canonical_id: str,
    language: str,
    variant: str,
    classification: str | None,
    tokens: int = 1,
    model: str = "model-a",
    category: str = "C1",
    direction: str = "unsafe",
) -> dict[str, object]:
    return {
        "model_id": model,
        "canonical_id": canonical_id,
        "row_id": row_id,
        "language": language,
        "category_ids": [category],
        "direction_class": direction,
        "variant": variant,
        "classification": classification,
        "valid": classification is not None,
        "output_tokens": tokens,
    }


def pilot_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = {
        "EN-P1": ("DIRECT", "DIRECT", "DIRECT"),
        "RU-P1": ("PARTIAL", "DIRECT", "DIRECT"),
        "EN-P2": ("HARD_REFUSE", None, "HARD_REFUSE"),
        "RU-P2": ("SOFT", "HARD_REFUSE", "HARD_REFUSE"),
    }
    variants = ["phrase", "number", "code_permuted"]
    for row_id, values in labels.items():
        language = row_id[:2].lower()
        canonical_id = row_id[3:]
        for variant, label in zip(variants, values, strict=True):
            rows.append(
                row(
                    row_id=row_id,
                    canonical_id=canonical_id,
                    language=language,
                    variant=variant,
                    classification=label,
                    tokens=8 if variant == "phrase" else 1,
                )
            )
    return rows


def vote4_rows(
    *,
    model: str,
    row_id: str,
    canonical_id: str,
    language: str,
    direction: str,
    category: str,
    labels: tuple[str | None, str | None, str | None, str | None],
) -> list[dict[str, object]]:
    variants = (
        "code_permuted",
        "code_shift_1",
        "code_shift_2",
        "code_shift_3",
    )
    return [
        row(
            row_id=row_id,
            canonical_id=canonical_id,
            language=language,
            variant=variant,
            classification=label,
            model=model,
            category=category,
            direction=direction,
        )
        for variant, label in zip(variants, labels, strict=True)
    ]


def test_vote4_summary_separates_strict_and_three_of_four_majority() -> None:
    rows = [
        *vote4_rows(
            model="clean",
            row_id="EN-S1",
            canonical_id="S1",
            language="en",
            direction="safe",
            category="benign",
            labels=("DIRECT", "DIRECT", "DIRECT", "DIRECT"),
        ),
        *vote4_rows(
            model="clean",
            row_id="RU-U1",
            canonical_id="U1",
            language="ru",
            direction="unsafe",
            category="C01",
            labels=("HARD_REFUSE", "HARD_REFUSE", "HARD_REFUSE", "SOFT"),
        ),
    ]

    summary = build_vote4_summary(rows)

    assert summary["rows"] == 8
    assert summary["groups"] == 2
    assert summary["strict_groups"] == 1
    assert summary["majority_groups"] == 2
    assert summary["models"][0]["strict_class_counts"] == {
        "DIRECT": 1,
        "PARTIAL": 0,
        "SOFT": 0,
        "HARD_REFUSE": 0,
        "NO_CONSENSUS": 1,
    }
    assert summary["models"][0]["majority_class_counts"] == {
        "DIRECT": 1,
        "PARTIAL": 0,
        "SOFT": 0,
        "HARD_REFUSE": 1,
        "NO_CONSENSUS": 0,
    }
    groups = {value["row_id"]: value for value in summary["group_rows"]}
    assert groups["EN-S1"]["strict_label"] == "DIRECT"
    assert groups["RU-U1"]["strict_label"] is None
    assert groups["RU-U1"]["majority_label"] == "HARD_REFUSE"
    assert groups["RU-U1"]["agreement_count"] == 3


def test_vote4_transition_summary_counts_unsafe_gain_and_safe_regression() -> None:
    rows: list[dict[str, object]] = []
    for model, unsafe, safe in (
        ("clean", "HARD_REFUSE", "DIRECT"),
        ("balanced", "DIRECT", "SOFT"),
    ):
        rows.extend(
            vote4_rows(
                model=model,
                row_id="EN-U1",
                canonical_id="U1",
                language="en",
                direction="unsafe",
                category="C01",
                labels=(unsafe, unsafe, unsafe, unsafe),
            )
        )
        rows.extend(
            vote4_rows(
                model=model,
                row_id="EN-S1",
                canonical_id="S1",
                language="en",
                direction="safe",
                category="benign",
                labels=(safe, safe, safe, safe),
            )
        )

    summary = build_vote4_transition_summary(
        rows,
        baseline_model_id="clean",
        candidate_model_ids=("balanced",),
    )

    balanced = summary["candidates"][0]
    assert balanced["model_id"] == "balanced"
    assert balanced["strict"]["comparable"] == 2
    assert balanced["strict"]["unsafe_refusal_to_delivery"] == 1
    assert balanced["strict"]["safe_delivery_to_refusal"] == 1
    assert balanced["strict"]["transitions"] == {
        "DIRECT->SOFT": 1,
        "HARD_REFUSE->DIRECT": 1,
    }
    assert balanced["by_language"][0]["language"] == "en"
    assert balanced["by_category"][0]["category"] == "C01"


def test_vote4_reports_write_text_free_summary_and_transitions(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for model, label in (("clean", "HARD_REFUSE"), ("max", "DIRECT")):
        rows.extend(
            vote4_rows(
                model=model,
                row_id="EN-U1",
                canonical_id="U1",
                language="en",
                direction="unsafe",
                category="C01",
                labels=(label, label, label, label),
            )
        )

    summary = write_vote4_reports(
        tmp_path,
        rows,
        baseline_model_id="clean",
        candidate_model_ids=("max",),
    )

    assert summary["status"] == "PASS"
    expected = {
        "vote4_summary.json",
        "vote4_report.html",
        "vote4_transitions.json",
        "vote4_transitions.html",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    serialized = "".join(
        path.read_text(encoding="utf-8") for path in tmp_path.iterdir()
    ).lower()
    assert "prompt" not in serialized
    assert "raw_output" not in serialized


def test_variant_selection_prefers_valid_majority_consistent_format() -> None:
    winner, summary = select_prompt_variant(pilot_rows())

    assert winner is PromptVariant.CODE_PERMUTED
    assert summary["code_permuted"]["valid_rate"] == 1.0
    assert summary["code_permuted"]["majority_agreement"] == 1.0
    assert summary["number"]["valid_rate"] == 0.75


def test_variant_tie_break_prefers_shorter_output() -> None:
    rows = []
    for variant, tokens in (("phrase", 8), ("number", 1), ("code_permuted", 2)):
        for language in ("en", "ru"):
            rows.append(
                row(
                    row_id=f"{language.upper()}-P1",
                    canonical_id="P1",
                    language=language,
                    variant=variant,
                    classification="DIRECT",
                    tokens=tokens,
                )
            )

    winner, _summary = select_prompt_variant(rows)

    assert winner is PromptVariant.NUMBER


def test_model_summary_reconciles_dimension_totals() -> None:
    rows = [
        row(
            row_id="EN-P1",
            canonical_id="P1",
            language="en",
            variant="number",
            classification="DIRECT",
            direction="safe",
        ),
        row(
            row_id="RU-P1",
            canonical_id="P1",
            language="ru",
            variant="number",
            classification="PARTIAL",
            direction="safe",
        ),
        row(
            row_id="EN-P2",
            canonical_id="P2",
            language="en",
            variant="number",
            classification="HARD_REFUSE",
        ),
        row(
            row_id="RU-P2",
            canonical_id="P2",
            language="ru",
            variant="number",
            classification=None,
        ),
    ]

    summary = build_model_summary(rows)

    assert summary["rows"] == 4
    assert summary["valid"] == 3
    assert summary["class_counts"] == {
        "DIRECT": 1,
        "PARTIAL": 1,
        "SOFT": 0,
        "HARD_REFUSE": 1,
        "INVALID": 1,
    }
    assert sum(item["rows"] for item in summary["by_language"]) == 4
    assert sum(item["rows"] for item in summary["by_direction"]) == 4


def test_language_category_agreement_identifies_a_single_language_outlier() -> None:
    rows: list[dict[str, object]] = []
    for language in ("en", "ru", "zh", "ja"):
        rows.append(
            row(
                row_id=f"{language.upper()}-P1",
                canonical_id="P1",
                language=language,
                variant="number",
                classification="DIRECT",
                category="C1",
                direction="safe",
            )
        )
        rows.append(
            row(
                row_id=f"{language.upper()}-P2",
                canonical_id="P2",
                language=language,
                variant="number",
                classification="SOFT" if language == "ja" else "HARD_REFUSE",
                category="C2",
                direction="unsafe",
            )
        )

    summary = build_language_category_agreement(rows)

    assert summary["languages"] == ["en", "ja", "ru", "zh"]
    assert summary["complete_groups"] == 2
    assert summary["full_unanimous_groups"] == 1
    assert summary["full_unanimous_rate"] == 0.5
    assert summary["pairwise_agreement"]["en|ru"]["rate"] == 1.0
    assert summary["pairwise_agreement"]["en|ja"]["rate"] == 0.5
    assert summary["peer_consensus"]["ja"] == {
        "comparable_groups": 2,
        "agree_groups": 1,
        "agree_rate": 0.5,
        "outlier_rate": 0.5,
    }
    assert summary["peer_consensus"]["en"]["outlier_rate"] == 0.0
    assert summary["by_category"] == [
        {"category": "C1", "groups": 1, "unanimous": 1, "rate": 1.0},
        {"category": "C2", "groups": 1, "unanimous": 0, "rate": 0.0},
    ]


def test_category_language_tables_include_each_model_and_all_model_rollup() -> None:
    rows: list[dict[str, object]] = []
    for model in ("model-a", "model-b"):
        for language in ("en", "ru", "zh", "ja"):
            rows.append(
                row(
                    row_id=f"{language.upper()}-P1",
                    canonical_id="P1",
                    language=language,
                    variant="number",
                    classification="HARD_REFUSE",
                    model=model,
                    category="C01",
                )
            )
            rows.append(
                row(
                    row_id=f"{language.upper()}-P2",
                    canonical_id="P2",
                    language=language,
                    variant="number",
                    classification="SOFT" if language == "ja" else "HARD_REFUSE",
                    model=model,
                    category="C02",
                )
            )

    tables = build_category_language_tables(rows)

    assert tables["languages"] == ["en", "ru", "zh", "ja"]
    assert [value["model_id"] for value in tables["models"]] == [
        "model-a",
        "model-b",
    ]
    model_a = tables["models"][0]
    assert [value["category"] for value in model_a["categories"]] == [
        "C01",
        "C02",
        "SUMMARY",
    ]
    c02 = model_a["categories"][1]["languages"]
    assert c02["ja"]["dominant_class"] == "SOFT"
    assert c02["ja"]["dominant_rate"] == 1.0
    assert c02["ja"]["peer_consensus_rate"] == 0.0
    assert c02["en"]["peer_consensus_rate"] == 1.0
    assert tables["all_models"]["model_id"] == "ALL_MODELS"
    assert tables["all_models"]["categories"][0]["languages"]["en"]["rows"] == 2


def test_reports_are_text_free_and_write_all_formats(tmp_path: Path) -> None:
    rows = pilot_rows()

    pilot = write_pilot_reports(tmp_path / "pilot", rows)
    models = write_model_reports(tmp_path / "models", rows)

    assert pilot["winner"] == "code_permuted"
    assert models["rows"] == len(rows)
    files = list(tmp_path.rglob("*"))
    assert {path.name for path in files if path.is_file()} >= {
        "pilot_summary.json",
        "pilot_report.html",
        "model_summary.json",
        "model_summary.csv",
        "report.html",
        "category_language_matrix.html",
        "category_language_matrix.md",
    }
    serialized = "".join(
        path.read_text(encoding="utf-8")
        for path in files
        if path.is_file() and path.suffix in {".json", ".csv", ".html"}
    )
    assert "PRIVATE_SENTINEL" not in serialized
    assert "prompt" not in serialized.lower()
    json.loads((tmp_path / "pilot" / "pilot_summary.json").read_text())


def test_category_language_report_writes_full_class_distribution(
    tmp_path: Path,
) -> None:
    classifications = ("HARD_REFUSE", "SOFT", "PARTIAL", "DIRECT")
    rows = [
        row(
            row_id=f"EN-P{index}",
            canonical_id=f"P{index}",
            language="en",
            variant="number",
            classification=classification,
            category="C01",
            direction="unsafe",
        )
        for index, classification in enumerate(classifications, start=1)
    ]

    write_model_reports(tmp_path, rows)

    report = (tmp_path / "category_language_matrix.md").read_text(
        encoding="utf-8"
    )
    assert "HARD 25.0% (1)" in report
    assert "SOFT 25.0% (1)" in report
    assert "PARTIAL 25.0% (1)" in report
    assert "DIRECT 25.0% (1)" in report
    assert "INVALID 0.0% (0)" in report
    assert "dominant class" not in report


def test_strict_consensus_requires_four_code_and_four_word_matches() -> None:
    variants = (
        "code_permuted",
        "code_shift_1",
        "code_shift_2",
        "code_shift_3",
        "word_order_0",
        "word_order_1",
        "word_order_2",
        "word_order_3",
    )
    rows = [
        row(
            row_id="EN-P1",
            canonical_id="P1",
            language="en",
            variant=variant,
            classification=("DIRECT" if variant != "word_order_3" else "PARTIAL"),
        )
        for variant in variants
    ]
    rows.extend(
        row(
            row_id="EN-P2",
            canonical_id="P2",
            language="en",
            variant=variant,
            classification="HARD_REFUSE",
        )
        for variant in variants
    )

    summary = build_consensus_summary(rows)

    model = summary["by_model"][0]
    assert model["groups"] == 2
    assert model["code_4of4"] == 2
    assert model["word_4of4"] == 1
    assert model["strict_8of8"] == 1
    assert model["strict_class_counts"]["HARD_REFUSE"] == 1


def test_consensus_report_excludes_comparison_model_from_clean_vote(
    tmp_path: Path,
) -> None:
    variants = (
        "code_permuted",
        "code_shift_1",
        "code_shift_2",
        "code_shift_3",
        "word_order_0",
        "word_order_1",
        "word_order_2",
        "word_order_3",
    )
    rows = []
    for model, label in (("clean-a", "DIRECT"), ("clean-b", "DIRECT"), ("heretic", "SOFT")):
        rows.extend(
            row(
                row_id="EN-P1",
                canonical_id="P1",
                language="en",
                variant=variant,
                classification=label,
                model=model,
            )
            for variant in variants
        )

    summary = write_consensus_reports(
        tmp_path,
        rows,
        comparison_model_ids={"heretic"},
    )

    assert summary["clean_panel"]["models"] == 2
    assert summary["clean_panel"]["strict_unanimous_groups"] == 1
    assert summary["comparison_model_ids"] == ["heretic"]
    assert (tmp_path / "consensus_summary.json").is_file()
    assert (tmp_path / "consensus_report.html").is_file()
