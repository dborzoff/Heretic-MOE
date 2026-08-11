from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "research" / "scripts" / "apply_translation_audit.py"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run(corpus: Path, audit: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus-root",
            str(corpus),
            "--audit",
            str(audit),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )


def test_audited_repairs_are_applied_by_id_and_provenance_is_appended(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [
        {
            "canonical_id": "U0001",
            "row_id": "RU-U0001",
            "language": "ru",
            "direction_class": "unsafe",
            "category_id": "C01",
            "prompt": "old-one",
        },
        {
            "canonical_id": "U0002",
            "row_id": "RU-U0002",
            "language": "ru",
            "direction_class": "unsafe",
            "category_id": "C01",
            "prompt": "old-two",
        },
    ]
    _write_jsonl(corpus / "direction_ru_unsafe.jsonl", rows)
    _write_jsonl(corpus / "direction_ru_unsafe_train.jsonl", rows)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "direction_ru_unsafe.jsonl": {"rows": 2, "sha256": "old"},
                    "direction_ru_unsafe_train.jsonl": {"rows": 2, "sha256": "old"},
                }
            }
        ),
        encoding="utf-8",
    )
    first_audit = tmp_path / "first.jsonl"
    _write_jsonl(
        first_audit,
        [
            {
                "case_id": "X001",
                "canonical_id": "U0001",
                "verdict": "translation_collapse",
                "confidence": 0.9,
                "corrected_ru_prompt": "new-one",
            },
            {
                "case_id": "X001",
                "canonical_id": "U0002",
                "verdict": "equivalent_but_acceptable",
                "confidence": 0.9,
            },
        ],
    )

    first = _run(corpus, first_audit)

    assert first.returncode == 0, first.stderr
    repaired = [
        json.loads(line)
        for line in (corpus / "direction_ru_unsafe.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["prompt"] for row in repaired] == ["new-one", "old-two"]
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["translation_repairs"]) == 1
    assert manifest["translation_repairs"][0]["corrected_unique_ids"] == 1

    second_audit = tmp_path / "second.jsonl"
    _write_jsonl(
        second_audit,
        [
            {
                "case_id": "X001",
                "canonical_id": "U0002",
                "verdict": "hash_uniqueness_rephrase",
                "confidence": 0.9,
                "corrected_ru_prompt": "new-two",
            }
        ],
    )
    second = _run(corpus, second_audit)

    assert second.returncode == 0, second.stderr
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["translation_repairs"]) == 2
