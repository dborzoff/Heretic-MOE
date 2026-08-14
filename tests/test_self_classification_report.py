from __future__ import annotations

import json
from pathlib import Path

from heretic.self_classification import PromptVariant
from heretic.self_classification_report import (
    build_model_summary,
    select_prompt_variant,
    write_model_reports,
    write_pilot_reports,
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
    }
    serialized = "".join(
        path.read_text(encoding="utf-8")
        for path in files
        if path.is_file() and path.suffix in {".json", ".csv", ".html"}
    )
    assert "PRIVATE_SENTINEL" not in serialized
    assert "prompt" not in serialized.lower()
    json.loads((tmp_path / "pilot" / "pilot_summary.json").read_text())
