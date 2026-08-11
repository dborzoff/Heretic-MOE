import hashlib
import json
from pathlib import Path

import pytest

from heretic.multilingual_contract import (
    build_frozen_run_contract,
    load_multilingual_dataset_bundle,
    write_or_verify_frozen_contract,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aligned_rows(
    *, language: str, direction: str, ids: list[str], prompt_prefix: str
) -> list[dict[str, object]]:
    return [
        {
            "canonical_id": canonical_id,
            "row_id": f"{language.upper()}-{canonical_id}",
            "language": language,
            "direction_class": direction,
            "category_id": "C01",
            "prompt": f"{prompt_prefix}-{language}-{canonical_id}",
        }
        for canonical_id in ids
    ]


def _calibration_rows(
    *, language: str, prefix: str, prompt_prefix: str, count: int
) -> list[dict[str, object]]:
    return [
        {
            "base_id": f"{prefix}{index:04d}",
            "row_id": f"{language.upper()}-{prefix}{index:04d}",
            "language": language,
            "category_id": "C01",
            "difficulty": "direct",
            "prompt": f"{prompt_prefix}-{language}-{index}",
            "source": "test",
        }
        for index in range(1, count + 1)
    ]


def _build_fixture(root: Path) -> tuple[Path, Path]:
    split = root / "operative_split_1000_400_v1"
    languages = ("en", "ru")
    for language in languages:
        for direction, prefix in (("safe", "S"), ("unsafe", "U")):
            _write_jsonl(
                split / f"direction_{language}_{direction}_2.jsonl",
                _aligned_rows(
                    language=language,
                    direction=direction,
                    ids=[f"{prefix}0001", f"{prefix}0002"],
                    prompt_prefix=f"direction-{direction}",
                ),
            )
            _write_jsonl(
                split / f"trial_{language}_{direction}_1.jsonl",
                _aligned_rows(
                    language=language,
                    direction=direction,
                    ids=[f"{prefix}0003"],
                    prompt_prefix=f"trial-{direction}",
                ),
            )
        _write_jsonl(
            root / f"search_unsafe_{language}.jsonl",
            _calibration_rows(
                language=language, prefix="Q", prompt_prefix="search", count=2
            ),
        )
        _write_jsonl(
            root / f"srg_calibration_{language}.jsonl",
            _calibration_rows(
                language=language, prefix="R", prompt_prefix="final", count=2
            ),
        )
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (split / "manifest.json").write_text("{}\n", encoding="utf-8")
    return root, split


def test_loads_text_free_frozen_contract_with_exact_pool_counts(tmp_path: Path):
    root, split = _build_fixture(tmp_path)

    bundle = load_multilingual_dataset_bundle(
        dataset_root=root,
        split_root=split,
        languages=("en", "ru"),
        direction_rows_per_cell=2,
        trial_rows_per_cell=1,
        calibration_rows_per_language=2,
    )

    assert len(bundle.direction_rows) == 8
    assert len(bundle.trial_rows) == 4
    assert len(bundle.search_rows) == 4
    assert len(bundle.final_rows) == 4
    assert bundle.manifest["counts"] == {
        "direction": 8,
        "trial": 4,
        "srg_calibration": 4,
        "final_holdout": 4,
    }
    serialized = json.dumps(bundle.manifest, sort_keys=True)
    assert "prompt" not in serialized
    assert "direction-safe" not in serialized
    assert len(bundle.manifest["contract_sha256"]) == 64


def test_rejects_cross_pool_prompt_overlap(tmp_path: Path):
    root, split = _build_fixture(tmp_path)
    trial = split / "trial_en_safe_1.jsonl"
    rows = [json.loads(line) for line in trial.read_text(encoding="utf-8").splitlines()]
    rows[0]["prompt"] = "search-en-1"
    _write_jsonl(trial, rows)

    with pytest.raises(ValueError, match="prompt overlap"):
        load_multilingual_dataset_bundle(
            dataset_root=root,
            split_root=split,
            languages=("en", "ru"),
            direction_rows_per_cell=2,
            trial_rows_per_cell=1,
            calibration_rows_per_language=2,
        )


def test_rejects_cross_language_calibration_id_order_drift(tmp_path: Path):
    root, split = _build_fixture(tmp_path)
    path = root / "search_unsafe_ru.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(path, list(reversed(rows)))

    with pytest.raises(ValueError, match="coverage or order drift"):
        load_multilingual_dataset_bundle(
            dataset_root=root,
            split_root=split,
            languages=("en", "ru"),
            direction_rows_per_cell=2,
            trial_rows_per_cell=1,
            calibration_rows_per_language=2,
        )


def test_contract_sha_changes_when_a_source_file_changes(tmp_path: Path):
    root, split = _build_fixture(tmp_path)
    first = load_multilingual_dataset_bundle(
        dataset_root=root,
        split_root=split,
        languages=("en", "ru"),
        direction_rows_per_cell=2,
        trial_rows_per_cell=1,
        calibration_rows_per_language=2,
    )
    manifest = root / "manifest.json"
    manifest.write_text('{"revision": 2}\n', encoding="utf-8")
    second = load_multilingual_dataset_bundle(
        dataset_root=root,
        split_root=split,
        languages=("en", "ru"),
        direction_rows_per_cell=2,
        trial_rows_per_cell=1,
        calibration_rows_per_language=2,
    )

    assert first.manifest["contract_sha256"] != second.manifest["contract_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == second.manifest[
        "source_manifests"
    ]["dataset_manifest_sha256"]


def test_frozen_run_contract_is_deterministic_and_text_free():
    contract = build_frozen_run_contract(
        dataset_contract_sha256="a" * 64,
        model_id="model-a",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        map_sha256="b" * 64,
        srg_profile_sha256="c" * 64,
        schedule_seed=42,
        schedule_version=2,
        generation_contract={"ordinary_max_new_tokens": 512},
        metric_contract={"version": 3},
        constraint_contract={"safe_ppl_drift_max": 0.01},
    )

    assert contract["contract_sha256"] == build_frozen_run_contract(
        dataset_contract_sha256="a" * 64,
        model_id="model-a",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        map_sha256="b" * 64,
        srg_profile_sha256="c" * 64,
        schedule_seed=42,
        schedule_version=2,
        generation_contract={"ordinary_max_new_tokens": 512},
        metric_contract={"version": 3},
        constraint_contract={"safe_ppl_drift_max": 0.01},
    )["contract_sha256"]
    assert not ({"prompt", "response", "answer", "text"} & set(contract))


def test_resume_rejects_a_changed_frozen_contract(tmp_path: Path):
    path = tmp_path / "contract.json"
    original = build_frozen_run_contract(
        dataset_contract_sha256="a" * 64,
        model_id="model-a",
        model_revision=None,
        tokenizer_revision=None,
        map_sha256="b" * 64,
        srg_profile_sha256="c" * 64,
        schedule_seed=42,
        schedule_version=2,
        generation_contract={"ordinary_max_new_tokens": 512},
        metric_contract={"version": 3},
        constraint_contract={},
    )
    write_or_verify_frozen_contract(path, original)
    changed = {**original, "schedule_seed": 43}

    with pytest.raises(RuntimeError, match="contract mismatch"):
        write_or_verify_frozen_contract(path, changed)
