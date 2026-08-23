from __future__ import annotations

import hashlib
import json

from heretic.multilingual_clean_dataset import materialize_clean_candidate_datasets


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_clean_sets_preserves_aligned_ids_and_private_prompts(tmp_path) -> None:
    """Catches language drift or writing rejected candidates into clean datasets."""
    languages = ["en", "ru", "zh", "ja"]
    aligned = tmp_path / "aligned"
    files = []
    for language in languages:
        for direction in ("safe", "unsafe"):
            path = aligned / f"direction_{language}_{direction}.jsonl"
            rows = []
            if direction == "unsafe":
                rows = [
                    {
                        "canonical_id": canonical_id,
                        "row_id": f"{language}-{canonical_id}",
                        "language": language,
                        "direction_class": "unsafe",
                        "category_ids": ["C01"],
                        "prompt": f"PRIVATE-{language}-{canonical_id}",
                    }
                    for canonical_id in ("U0001", "U0002", "U0003")
                ]
            write_jsonl(path, rows)
            files.append(
                {
                    "language": language,
                    "direction": direction,
                    "path": path.name,
                    "rows": len(rows),
                    "sha256": sha256(path),
                }
            )
    aligned_manifest = aligned / "manifest.json"
    aligned_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "languages": languages,
                "directions": {"safe": 0, "unsafe": 3},
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    consensus = tmp_path / "candidate_consensus.jsonl"
    write_jsonl(
        consensus,
        [
            {
                "canonical_id": "U0001",
                "target_behavior_class": "HARD_REFUSE",
                "clean": True,
            },
            {
                "canonical_id": "U0002",
                "target_behavior_class": "SOFT",
                "clean": True,
            },
            {
                "canonical_id": "U0003",
                "target_behavior_class": "SOFT",
                "clean": False,
            },
        ],
    )

    manifest = materialize_clean_candidate_datasets(
        aligned_manifest,
        consensus,
        tmp_path / "clean",
    )

    assert manifest["status"] == "PASS"
    assert manifest["clean_hard_per_language"] == 1
    assert manifest["clean_soft_per_language"] == 1
    for language in languages:
        hard = (tmp_path / "clean" / f"clean_hard_{language}.jsonl").read_text(
            encoding="utf-8"
        )
        soft = (tmp_path / "clean" / f"clean_soft_{language}.jsonl").read_text(
            encoding="utf-8"
        )
        assert "U0001" in hard and "U0002" not in hard and "U0003" not in hard
        assert "U0002" in soft and "U0001" not in soft and "U0003" not in soft
        assert f"PRIVATE-{language}-U0001" in hard
        assert f"PRIVATE-{language}-U0002" in soft
