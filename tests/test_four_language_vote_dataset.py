from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heretic.four_language_vote_dataset import materialize_vote_dataset
from heretic.self_classification_data import load_classification_rows

LANGUAGES = ("en", "ru", "zh", "ja")
POOLS = (("map", 2), ("trial", 1), ("final", 1))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_source(root: Path) -> Path:
    files: dict[str, dict[str, object]] = {}
    counts: dict[str, dict[str, int]] = {}
    for pool, rows_per_direction in POOLS:
        counts[pool] = {"safe": rows_per_direction, "unsafe": rows_per_direction}
        for language in LANGUAGES:
            for direction in ("safe", "unsafe"):
                path = root / f"{pool}_{language}_{direction}_{rows_per_direction}.jsonl"
                rows = []
                prefix = "S" if direction == "safe" else "U"
                for index in range(rows_per_direction):
                    canonical_id = f"{prefix}-{pool}-{index}"
                    rows.append(
                        {
                            "canonical_id": canonical_id,
                            "row_id": f"{language.upper()}-{canonical_id}",
                            "language": language,
                            "direction_class": direction,
                            "category_id": f"{direction}-{pool}",
                            "category_ids": [f"{direction}-{pool}"],
                            "prompt": f"PRIVATE-{language}-{canonical_id}",
                        }
                    )
                path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                files[path.name] = {
                    "pool": pool,
                    "language": language,
                    "direction_class": direction,
                    "rows": rows_per_direction,
                    "sha256": _sha256(path),
                }
    manifest = {
        "schema_version": 4,
        "status": "PASS",
        "languages": list(LANGUAGES),
        "counts": counts,
        "files": files,
        "cross_pool_overlap": {"safe": 0, "unsafe": 0},
        "cross_pool_content_overlap": {
            "safe": {"map_trial": 0, "map_final": 0, "trial_final": 0},
            "unsafe": {"map_trial": 0, "map_final": 0, "trial_final": 0},
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def test_materializes_aligned_schema_v1_vote_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _build_source(source)
    output = tmp_path / "vote4"

    manifest = materialize_vote_dataset(source, output)

    assert manifest["status"] == "PASS"
    assert manifest["schema_version"] == 1
    assert manifest["directions"] == {"safe": 4, "unsafe": 4}
    assert manifest["languages"] == list(LANGUAGES)
    assert manifest["rows"] == 32
    assert len(manifest["files"]) == 8
    assert manifest["source_pools"] == ["map", "trial", "final"]
    rows = load_classification_rows(output / "manifest.json", LANGUAGES)
    assert len(rows) == 32
    en_safe = [
        row.canonical_id
        for row in rows
        if row.language == "en" and row.direction_class == "safe"
    ]
    assert en_safe == ["S-map-0", "S-map-1", "S-trial-0", "S-final-0"]
    public = (output / "manifest.json").read_text(encoding="utf-8")
    assert "PRIVATE-" not in public
    assert '"prompt"' not in public


def test_rejects_source_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _build_source(source)
    target = source / "map_en_safe_2.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash drift"):
        materialize_vote_dataset(source, tmp_path / "vote4")

