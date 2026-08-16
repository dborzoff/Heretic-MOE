from __future__ import annotations

import json
from pathlib import Path

from heretic.four_language_corpus import build_four_language_corpus

LANGUAGES = ("en", "ru", "zh", "ja")


def _write_source(
    root: Path,
    *,
    direction: str,
    rows: list[tuple[str, str]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        path = root / f"direction_{language}_{direction}_strict.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for canonical_id, category in rows:
                handle.write(
                    json.dumps(
                        {
                            "canonical_id": canonical_id,
                            "row_id": f"{language.upper()}-{canonical_id}",
                            "language": language,
                            "direction_class": direction,
                            "category_id": category,
                            "category_ids": [category],
                            "prompt": f"{language}-{canonical_id}",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "direction": direction,
                "canonical_rows": len(rows),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_consensus(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for canonical_id, _ in rows:
            score = int(canonical_id.removeprefix("U"))
            for language in LANGUAGES:
                for model in ("m0", "m1"):
                    for variant in ("v0", "v1"):
                        label = (
                            "HARD_REFUSE"
                            if score <= 4 or (score <= 8 and variant == "v0")
                            else "SOFT"
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "canonical_id": canonical_id,
                                    "row_id": f"{language.upper()}-{canonical_id}",
                                    "language": language,
                                    "model_id": model,
                                    "variant": variant,
                                    "direction_class": "unsafe",
                                    "classification": label,
                                    "valid": True,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )


def _ids(path: Path) -> list[str]:
    return [
        str(json.loads(line)["canonical_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_builds_disjoint_aligned_category_stratified_pools(tmp_path: Path) -> None:
    safe_ranked = tmp_path / "safe-ranked"
    safe_map = tmp_path / "safe-map"
    unsafe = tmp_path / "unsafe"
    safe_ranked_rows = [(f"P{index:02d}", "polyguard_benign") for index in range(6)]
    safe_map_rows = [
        (f"S{index:02d}", "safe_a" if index % 2 == 0 else "safe_b")
        for index in range(14)
    ]
    unsafe_rows = [
        (f"U{index:02d}", "C01" if index % 2 == 0 else "C02")
        for index in range(16)
    ]
    _write_source(safe_ranked, direction="safe", rows=safe_ranked_rows)
    _write_source(safe_map, direction="safe", rows=safe_map_rows)
    _write_source(unsafe, direction="unsafe", rows=unsafe_rows)
    consensus = tmp_path / "unsafe-consensus.jsonl"
    _write_consensus(consensus, unsafe_rows)

    output = tmp_path / "output"
    manifest = build_four_language_corpus(
        output_root=output,
        safe_ranked_root=safe_ranked,
        safe_map_root=safe_map,
        unsafe_root=unsafe,
        unsafe_consensus_path=consensus,
        languages=LANGUAGES,
        map_rows=10,
        trial_rows=4,
        final_rows=2,
    )

    assert manifest["status"] == "PASS"
    assert manifest["counts"] == {
        "map": {"safe": 10, "unsafe": 10},
        "trial": {"safe": 4, "unsafe": 4},
        "final": {"safe": 2, "unsafe": 2},
        "reserve": {"safe": 4, "unsafe": 0},
    }
    for direction in ("safe", "unsafe"):
        pools = {}
        for pool, count in (("map", 10), ("trial", 4), ("final", 2)):
            reference = _ids(
                output / f"{pool}_en_{direction}_{count}.jsonl"
            )
            pools[pool] = set(reference)
            for language in LANGUAGES[1:]:
                assert _ids(
                    output / f"{pool}_{language}_{direction}_{count}.jsonl"
                ) == reference
        assert pools["map"].isdisjoint(pools["trial"])
        assert pools["map"].isdisjoint(pools["final"])
        assert pools["trial"].isdisjoint(pools["final"])

    assert _ids(output / "trial_en_unsafe_4.jsonl") == [
        "U00",
        "U02",
        "U01",
        "U03",
    ]
    public = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "prompt" not in json.dumps(public).lower()
    assert public["cross_pool_overlap"] == {
        "safe": 0,
        "unsafe": 0,
    }
    assert (
        output / "categories" / "trial" / "unsafe" / "C01.jsonl"
    ).is_file()
