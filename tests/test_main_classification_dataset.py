from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heretic.main_classification_dataset import (
    apply_aligned_translations,
    materialize_main_classification_dataset,
)
from heretic.self_classification_data import load_classification_rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for language in ("en", "ru", "zh"):
        prefix = language.upper()
        _write_jsonl(
            root / f"direction_{language}_safe.jsonl",
            [
                {
                    "canonical_id": "S0001",
                    "row_id": f"{prefix}-S0001",
                    "language": language,
                    "direction_class": "safe",
                    "category_id": "GEN",
                    "prompt": f"safe-{language}",
                }
            ],
        )
        _write_jsonl(
            root / f"direction_{language}_unsafe.jsonl",
            [
                {
                    "canonical_id": "U0001",
                    "row_id": f"{prefix}-U0001",
                    "language": language,
                    "direction_class": "unsafe",
                    "category_id": "C01",
                    "prompt": f"unsafe-{language}",
                }
            ],
        )
        for stem, base_id in (("search_unsafe", "Q0001"), ("srg_calibration", "R0001")):
            _write_jsonl(
                root / f"{stem}_{language}.jsonl",
                [
                    {
                        "base_id": base_id,
                        "row_id": f"{prefix}-{base_id}",
                        "language": language,
                        "category_id": "C02",
                        "prompt": f"{stem}-{language}",
                    }
                ],
            )
    return root


def test_materializes_direction_search_and_srg_as_one_aligned_dataset(
    tmp_path: Path,
) -> None:
    source = _source_root(tmp_path)
    output = tmp_path / "dataset"

    manifest = materialize_main_classification_dataset(
        source,
        output,
        languages=("en", "ru", "zh"),
    )

    assert manifest["status"] == "PASS"
    assert manifest["directions"] == {"safe": 1, "unsafe": 3}
    assert manifest["canonical_rows"] == 4
    assert manifest["rows"] == 12
    loaded = load_classification_rows(output / "manifest.json", ("en", "ru", "zh"))
    assert len(loaded) == 12
    assert [row.canonical_id for row in loaded[:4]] == [
        "S0001",
        "U0001",
        "Q0001",
        "R0001",
    ]
    assert not ({"prompt", "response", "text"} & set(manifest))


def test_applies_complete_aligned_translation_without_changing_ids(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    translations = tmp_path / "translations.jsonl"
    _write_jsonl(
        translations,
        [
            {"canonical_id": "S0001", "direction_class": "safe", "prompt": "ja-safe"},
            {"canonical_id": "U0001", "direction_class": "unsafe", "prompt": "ja-unsafe"},
            {"canonical_id": "Q0001", "direction_class": "unsafe", "prompt": "ja-search"},
            {"canonical_id": "R0001", "direction_class": "unsafe", "prompt": "ja-srg"},
        ],
    )
    output = tmp_path / "translated"

    summary = apply_aligned_translations(
        source,
        translations,
        output,
        target_language="ja",
    )

    assert summary == {"safe": 1, "unsafe": 3, "language": "ja"}
    safe = json.loads((output / "direction_ja_safe.jsonl").read_text(encoding="utf-8"))
    assert safe["canonical_id"] == "S0001"
    assert safe["row_id"] == "JA-S0001"
    assert safe["language"] == "ja"
    assert safe["prompt"] == "ja-safe"
    assert safe["translation_status"] == "aligned"


def test_rejects_incomplete_or_duplicate_translation_coverage(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    translations = tmp_path / "translations.jsonl"
    duplicate = {"canonical_id": "S0001", "direction_class": "safe", "prompt": "ja"}
    _write_jsonl(translations, [duplicate, duplicate])

    with pytest.raises(ValueError, match="translation coverage"):
        apply_aligned_translations(
            source,
            translations,
            tmp_path / "translated",
            target_language="ja",
        )


def test_manifest_hashes_detect_later_dataset_drift(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    output = tmp_path / "dataset"
    materialize_main_classification_dataset(source, output, languages=("en", "ru", "zh"))
    path = output / "direction_en_safe_strict.jsonl"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    assert hashlib.sha256(path.read_bytes()).hexdigest() != original_hash
    with pytest.raises(ValueError, match="hash drift"):
        load_classification_rows(output / "manifest.json", ("en",))
