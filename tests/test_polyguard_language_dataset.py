from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heretic.polyguard_language_dataset import (
    materialize_polyguard_language_dataset,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_materializes_only_unanimous_aligned_safe_and_unsafe_rows(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for language in ("English", "Russian"):
        rows.extend(
            [
                {
                    "id": 10,
                    "language": language,
                    "prompt": f"safe-{language}",
                    "response": "private-response",
                    "prompt_harm_label": "unharmful",
                    "response_refusal_label": "compliance",
                    "prompt_harm_agreement": 3.0,
                    "response_refusal_agreement": 3.0,
                    "prompt_categories": "",
                    "subcategory": "benign",
                },
                {
                    "id": 20,
                    "language": language,
                    "prompt": f"unsafe-{language}",
                    "response": "private-response",
                    "prompt_harm_label": "harmful",
                    "response_refusal_label": "refusal",
                    "prompt_harm_agreement": 3.0,
                    "response_refusal_agreement": 3.0,
                    "prompt_categories": "S9 S1 S9",
                    "subcategory": "violence_and_physical_harm",
                },
                {
                    "id": 30,
                    "language": language,
                    "prompt": f"ambiguous-{language}",
                    "response": "private-response",
                    "prompt_harm_label": "harmful",
                    "response_refusal_label": "refusal",
                    "prompt_harm_agreement": 2.0,
                    "response_refusal_agreement": 3.0,
                    "prompt_categories": "S2",
                    "subcategory": "others",
                },
            ]
        )
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    output = tmp_path / "dataset"
    manifest = materialize_polyguard_language_dataset(
        source,
        output,
        language_codes={"English": "en", "Russian": "ru"},
        expected_counts={"safe": 1, "unsafe": 1},
    )

    assert manifest["status"] == "PASS"
    assert manifest["languages"] == ["en", "ru"]
    assert manifest["directions"] == {"safe": 1, "unsafe": 1}
    assert manifest["rows"] == 4
    unsafe = _read_jsonl(output / "direction_en_unsafe_strict.jsonl")
    assert unsafe == [
        {
            "canonical_id": "PG-20",
            "category_id": "S1",
            "category_ids": ["S1", "S9"],
            "direction_class": "unsafe",
            "language": "en",
            "prompt": "unsafe-English",
            "row_id": "EN-PG-20",
            "source_category": "violence_and_physical_harm",
        }
    ]
    assert "response" not in unsafe[0]


def test_rejects_missing_aligned_language_row(tmp_path: Path) -> None:
    rows = [
        {
            "id": 10,
            "language": "English",
            "prompt": "safe-English",
            "prompt_harm_label": "unharmful",
            "response_refusal_label": "compliance",
            "prompt_harm_agreement": 3.0,
            "response_refusal_agreement": 3.0,
            "prompt_categories": "",
            "subcategory": "benign",
        },
        {
            "id": 20,
            "language": "English",
            "prompt": "unsafe-English",
            "prompt_harm_label": "harmful",
            "response_refusal_label": "refusal",
            "prompt_harm_agreement": 3.0,
            "response_refusal_agreement": 3.0,
            "prompt_categories": "S1",
            "subcategory": "violence_and_physical_harm",
        },
        {
            "id": 10,
            "language": "Russian",
            "prompt": "safe-Russian",
            "prompt_harm_label": "unharmful",
            "response_refusal_label": "compliance",
            "prompt_harm_agreement": 3.0,
            "response_refusal_agreement": 3.0,
            "prompt_categories": "",
            "subcategory": "benign",
        },
    ]
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    try:
        materialize_polyguard_language_dataset(
            source,
            tmp_path / "dataset",
            language_codes={"English": "en", "Russian": "ru"},
            expected_counts={"safe": 1, "unsafe": 1},
        )
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("missing aligned translation must fail")
