from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_hamilton_category_quotas_select_exactly_two_sevenths() -> None:
    from heretic.canonical_category_split import hamilton_category_quotas

    counts = {
        "C01": 180,
        "C02": 120,
        "C03": 100,
        "C04": 100,
        "C05": 100,
        "C06": 80,
        "C07": 120,
        "C08": 80,
        "C09": 80,
        "C10": 100,
        "C11": 80,
        "C12": 80,
        "C13": 100,
        "C14": 80,
    }

    assert hamilton_category_quotas(counts, selected_rows=400) == {
        "C01": 51,
        "C02": 34,
        "C03": 29,
        "C04": 29,
        "C05": 29,
        "C06": 23,
        "C07": 34,
        "C08": 23,
        "C09": 23,
        "C10": 28,
        "C11": 23,
        "C12": 23,
        "C13": 28,
        "C14": 23,
    }


def _write_direction_file(
    root: Path,
    *,
    language: str,
    direction: str,
    category_counts: dict[str, int],
) -> list[str]:
    prefix = "S" if direction == "safe" else "U"
    rows: list[str] = []
    index = 1
    for category, count in category_counts.items():
        for _ in range(count):
            canonical_id = f"{prefix}{index:04d}"
            row = {
                "canonical_id": canonical_id,
                "row_id": f"{language.upper()}-{canonical_id}",
                "language": language,
                "direction_class": direction,
                "category_id": category,
                "prompt": f"opaque-{language}-{canonical_id}",
                "source": "fixture",
                "source_ref": canonical_id,
                "canonical_language": "en",
                "translation_status": "canonical" if language == "en" else "aligned",
            }
            rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            index += 1
    path = root / f"direction_{language}_{direction}.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return rows


def _read_ids(path: Path) -> list[str]:
    return [
        json.loads(line)["canonical_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_materialize_split_keeps_identical_membership_in_every_language(
    tmp_path: Path,
) -> None:
    from heretic.canonical_category_split import materialize_category_split

    counts = {
        "C01": 180,
        "C02": 120,
        "C03": 100,
        "C04": 100,
        "C05": 100,
        "C06": 80,
        "C07": 120,
        "C08": 80,
        "C09": 80,
        "C10": 100,
        "C11": 80,
        "C12": 80,
        "C13": 100,
        "C14": 80,
    }
    languages = ("en", "ru", "zh", "es", "fr")
    source = tmp_path / "source"
    source.mkdir()
    source_lines: dict[tuple[str, str], list[str]] = {}
    for language in languages:
        for direction in ("safe", "unsafe"):
            source_lines[(language, direction)] = _write_direction_file(
                source,
                language=language,
                direction=direction,
                category_counts=counts,
            )

    output = tmp_path / "split"
    manifest = materialize_category_split(
        source,
        output,
        languages=languages,
        selected_rows=400,
        seed=20260811,
    )

    assert manifest["status"] == "PASS"
    assert manifest["selected_rows"] == 400
    assert manifest["direction_rows"] == 1000
    assert len(manifest["inputs"]) == 10
    verify_report = json.loads(
        (output / "verify_report.json").read_text(encoding="utf-8")
    )
    assert verify_report["status"] == "PASS"
    assert all(verify_report["checks"].values())
    for direction in ("safe", "unsafe"):
        expected_trial = _read_ids(output / f"trial_en_{direction}_400.jsonl")
        expected_fit = _read_ids(output / f"direction_en_{direction}_1000.jsonl")
        membership = json.loads(
            (output / f"canonical_membership_{direction}.json").read_text(
                encoding="utf-8"
            )
        )
        assert membership["trial"] == expected_trial
        assert membership["direction"] == expected_fit
        assert len(expected_trial) == 400
        assert len(expected_fit) == 1000
        assert set(expected_trial).isdisjoint(expected_fit)
        assert set(expected_trial) | set(expected_fit) == {
            json.loads(line)["canonical_id"]
            for line in source_lines[("en", direction)]
        }
        for language in languages:
            assert _read_ids(output / f"trial_{language}_{direction}_400.jsonl") == expected_trial
            assert _read_ids(output / f"direction_{language}_{direction}_1000.jsonl") == expected_fit


def test_alignment_failure_does_not_publish_partial_output(tmp_path: Path) -> None:
    from heretic.canonical_category_split import materialize_category_split

    counts = {"C01": 7, "C02": 7}
    languages = ("en", "ru")
    source = tmp_path / "source"
    source.mkdir()
    for language in languages:
        for direction in ("safe", "unsafe"):
            _write_direction_file(
                source,
                language=language,
                direction=direction,
                category_counts=counts,
            )
    bad_path = source / "direction_ru_unsafe.jsonl"
    lines = bad_path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    bad_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = tmp_path / "split"
    with pytest.raises(ValueError, match="canonical alignment mismatch"):
        materialize_category_split(
            source,
            output,
            languages=languages,
            selected_rows=4,
            seed=7,
        )

    assert not output.exists()


def test_cli_prints_only_text_free_summary(tmp_path: Path) -> None:
    counts = {"C01": 7, "C02": 7}
    languages = ("en", "ru")
    source = tmp_path / "source"
    source.mkdir()
    for language in languages:
        for direction in ("safe", "unsafe"):
            _write_direction_file(
                source,
                language=language,
                direction=direction,
                category_counts=counts,
            )
    output = tmp_path / "split"
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "research" / "scripts" / "materialize_canonical_split.py"),
            "--corpus-root",
            str(source),
            "--output-dir",
            str(output),
            "--languages",
            "en,ru",
            "--trial-rows",
            "4",
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary == {
        "direction_rows": 10,
        "files": 8,
        "output_dir": str(output.resolve()),
        "status": "PASS",
        "trial_rows": 4,
    }
    assert "opaque" not in result.stdout


def test_missing_redundant_direction_field_is_allowed_in_aligned_translation(
    tmp_path: Path,
) -> None:
    from heretic.canonical_category_split import materialize_category_split

    counts = {"C01": 7, "C02": 7}
    source = tmp_path / "source"
    source.mkdir()
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            _write_direction_file(
                source,
                language=language,
                direction=direction,
                category_counts=counts,
            )
    for direction in ("safe", "unsafe"):
        path = source / f"direction_ru_{direction}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            del row["direction_class"]
        path.write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                for row in rows
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = materialize_category_split(
        source,
        tmp_path / "split",
        languages=("en", "ru"),
        selected_rows=4,
        seed=7,
    )

    assert manifest["status"] == "PASS"
