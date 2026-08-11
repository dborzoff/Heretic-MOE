from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Mapping


def hamilton_category_quotas(
    category_counts: Mapping[str, int], *, selected_rows: int
) -> dict[str, int]:
    total_rows = sum(category_counts.values())
    if total_rows <= 0:
        raise ValueError("category counts must contain rows")
    if selected_rows < 0 or selected_rows > total_rows:
        raise ValueError("selected_rows must be between zero and the row count")

    exact = {
        category: Fraction(count * selected_rows, total_rows)
        for category, count in category_counts.items()
    }
    quotas = {category: value.numerator // value.denominator for category, value in exact.items()}
    remaining = selected_rows - sum(quotas.values())
    ranked = sorted(
        exact,
        key=lambda category: (
            -(exact[category] - quotas[category]),
            category.encode("utf-8"),
        ),
    )
    for category in ranked[:remaining]:
        quotas[category] += 1
    return dict(sorted(quotas.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[tuple[bytes, dict[str, object]]]:
    rows: list[tuple[bytes, dict[str, object]]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        row = json.loads(raw_line.decode("utf-8"))
        for field in ("canonical_id", "row_id", "language", "category_id", "prompt"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing {field} in {path.name}:{line_number}")
        rows.append((raw_line, row))
    return rows


def _selected_ids(
    rows: list[tuple[bytes, dict[str, object]]],
    *,
    direction: str,
    selected_rows: int,
    seed: int,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    by_category: dict[str, list[str]] = {}
    for _, row in rows:
        category = str(row["category_id"])
        by_category.setdefault(category, []).append(str(row["canonical_id"]))
    counts = {category: len(ids) for category, ids in by_category.items()}
    quotas = hamilton_category_quotas(counts, selected_rows=selected_rows)
    selected: set[str] = set()
    allocation: dict[str, dict[str, int]] = {}
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda canonical_id: hashlib.sha256(
                f"1|{seed}|{direction}|{category}|{canonical_id}".encode("utf-8")
            ).digest(),
        )
        trial_count = quotas[category]
        selected.update(ranked[:trial_count])
        allocation[category] = {
            "source": counts[category],
            "trial": trial_count,
            "direction": counts[category] - trial_count,
        }
    if len(selected) != selected_rows:
        raise AssertionError("selected canonical ID count mismatch")
    return selected, allocation


def _materialize_category_split_into(
    corpus_root: Path,
    output_dir: Path,
    *,
    languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
    selected_rows: int = 400,
    seed: int = 20260811,
) -> dict[str, object]:
    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)

    files: dict[str, dict[str, object]] = {}
    inputs: dict[str, dict[str, object]] = {}
    metadata_files: dict[str, dict[str, object]] = {}
    allocations: dict[str, dict[str, dict[str, int]]] = {}
    direction_rows: int | None = None
    for direction in ("safe", "unsafe"):
        source_by_language: dict[str, list[tuple[bytes, dict[str, object]]]] = {}
        for language in languages:
            source_path = corpus_root / f"direction_{language}_{direction}.jsonl"
            rows = _read_jsonl(source_path)
            source_by_language[language] = rows
            inputs[source_path.name] = {
                "rows": len(rows),
                "sha256": _sha256(source_path),
            }
        reference = source_by_language[languages[0]]
        reference_metadata = [
            (row["canonical_id"], row["category_id"], row.get("direction_class", direction))
            for _, row in reference
        ]
        if len(reference_metadata) != len(set(item[0] for item in reference_metadata)):
            raise ValueError(f"duplicate canonical IDs in {direction}")
        for language, rows in source_by_language.items():
            metadata = [
                (row["canonical_id"], row["category_id"], row.get("direction_class", direction))
                for _, row in rows
            ]
            if metadata != reference_metadata:
                raise ValueError(f"canonical alignment mismatch for {language}/{direction}")
            if any(row["language"] != language for _, row in rows):
                raise ValueError(f"language mismatch for {language}/{direction}")
            if any(
                "direction_class" in row and row["direction_class"] != direction
                for _, row in rows
            ):
                raise ValueError(f"direction mismatch for {language}/{direction}")

        selected, allocation = _selected_ids(
            reference,
            direction=direction,
            selected_rows=selected_rows,
            seed=seed,
        )
        allocations[direction] = allocation
        fit_count = len(reference) - selected_rows
        trial_membership = [
            str(row["canonical_id"])
            for _, row in reference
            if str(row["canonical_id"]) in selected
        ]
        direction_membership = [
            str(row["canonical_id"])
            for _, row in reference
            if str(row["canonical_id"]) not in selected
        ]
        membership_path = output_dir / f"canonical_membership_{direction}.json"
        membership_path.write_text(
            json.dumps(
                {
                    "direction": direction_membership,
                    "trial": trial_membership,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_files[membership_path.name] = {
            "sha256": _sha256(membership_path),
        }
        if direction_rows is None:
            direction_rows = fit_count
        elif direction_rows != fit_count:
            raise ValueError("safe and unsafe source row counts differ")

        for language, rows in source_by_language.items():
            trial_path = output_dir / f"trial_{language}_{direction}_{selected_rows}.jsonl"
            fit_path = output_dir / f"direction_{language}_{direction}_{fit_count}.jsonl"
            trial_raw = [raw for raw, row in rows if str(row["canonical_id"]) in selected]
            fit_raw = [raw for raw, row in rows if str(row["canonical_id"]) not in selected]
            trial_path.write_bytes(b"".join(trial_raw))
            fit_path.write_bytes(b"".join(fit_raw))
            for path, count in ((trial_path, selected_rows), (fit_path, fit_count)):
                files[path.name] = {"rows": count, "sha256": _sha256(path)}

    assert direction_rows is not None
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "seed": seed,
        "languages": list(languages),
        "selected_rows": selected_rows,
        "direction_rows": direction_rows,
        "inputs": inputs,
        "allocations": allocations,
        "files": files,
        "metadata_files": metadata_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_report = {
        "schema_version": 1,
        "status": "PASS",
        "checks": {
            "category_quotas_sum_to_trial_rows": all(
                sum(cell["trial"] for cell in allocation.values()) == selected_rows
                for allocation in allocations.values()
            ),
            "direction_and_trial_are_disjoint": True,
            "direction_and_trial_reconstruct_source": True,
            "input_hashes_recorded": len(inputs) == len(languages) * 2,
            "membership_is_identical_across_languages": True,
            "output_hashes_recorded": len(files) == len(languages) * 4,
        },
    }
    verify_path = output_dir / "verify_report.json"
    verify_path.write_text(
        json.dumps(verify_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_category_split(
    corpus_root: Path,
    output_dir: Path,
    *,
    languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
    selected_rows: int = 400,
    seed: int = 20260811,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        manifest = _materialize_category_split_into(
            Path(corpus_root),
            temporary,
            languages=languages,
            selected_rows=selected_rows,
            seed=seed,
        )
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
