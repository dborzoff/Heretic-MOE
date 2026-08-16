from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical_category_split import hamilton_category_quotas

PROHIBITED_PUBLIC_FIELDS = {"prompt", "response", "answer", "text"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_text_free(value: object, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in PROHIBITED_PUBLIC_FIELDS:
                raise ValueError(f"{path} contains prohibited field {key!r}")
            _assert_text_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_text_free(child, f"{path}[{index}]")


def _iter_jsonl(path: Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank row in {path.name}:{line_number}")
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise TypeError(f"non-object row in {path.name}:{line_number}")
            yield raw, row


def _read_jsonl(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    return list(_iter_jsonl(path))


def _load_aligned_source(
    root: Path,
    *,
    direction: str,
    languages: Sequence[str],
) -> dict[str, list[tuple[bytes, dict[str, Any]]]]:
    by_language: dict[str, list[tuple[bytes, dict[str, Any]]]] = {}
    for language in languages:
        path = root / f"direction_{language}_{direction}_strict.jsonl"
        rows = _read_jsonl(path)
        seen: set[str] = set()
        for _, row in rows:
            canonical_id = str(row.get("canonical_id", "")).strip()
            if not canonical_id or canonical_id in seen:
                raise ValueError(f"invalid canonical IDs in {path.name}")
            seen.add(canonical_id)
            if row.get("language") != language:
                raise ValueError(f"language drift in {path.name}")
            if row.get("direction_class") != direction:
                raise ValueError(f"direction drift in {path.name}")
            category = str(row.get("category_id", "")).strip()
            if not category:
                raise ValueError(f"missing category in {path.name}")
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"empty prompt in {path.name}")
        by_language[language] = rows
    reference = [
        (row["canonical_id"], row["category_id"])
        for _, row in by_language[languages[0]]
    ]
    for language in languages[1:]:
        current = [
            (row["canonical_id"], row["category_id"])
            for _, row in by_language[language]
        ]
        if current != reference:
            raise ValueError(f"aligned source order drift for {language}/{direction}")
    return by_language


def _hash_rank(
    canonical_ids: Sequence[str],
    *,
    namespace: str,
) -> list[str]:
    return sorted(
        canonical_ids,
        key=lambda canonical_id: (
            hashlib.sha256(f"{namespace}|{canonical_id}".encode()).digest(),
            canonical_id,
        ),
    )


def _category_rankings_from_source(
    rows: Sequence[tuple[bytes, dict[str, Any]]],
    *,
    namespace: str,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for _, row in rows:
        grouped[str(row["category_id"])].append(str(row["canonical_id"]))
    return {
        category: _hash_rank(ids, namespace=f"{namespace}|{category}")
        for category, ids in sorted(grouped.items())
    }


def _unsafe_category_rankings(
    consensus_path: Path,
    source_rows: Sequence[tuple[bytes, dict[str, Any]]],
    *,
    languages: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    source_categories = {
        str(row["canonical_id"]): str(row["category_id"]) for _, row in source_rows
    }
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    valid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    models: set[str] = set()
    variants: set[str] = set()
    for _, row in _iter_jsonl(consensus_path):
        if row.get("direction_class") != "unsafe":
            continue
        canonical_id = str(row.get("canonical_id", ""))
        language = str(row.get("language", "")).lower()
        if canonical_id not in source_categories or language not in languages:
            continue
        models.add(str(row.get("model_id", "")))
        variants.add(str(row.get("variant", "")))
        totals[canonical_id][language] += 1
        if row.get("valid") is True:
            valid[canonical_id][language] += 1
            counts[canonical_id][language][str(row.get("classification"))] += 1
    expected = len(models) * len(variants)
    if expected <= 0:
        raise ValueError("unsafe consensus has no model/variant coverage")

    diagnostics: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for canonical_id, category in source_categories.items():
        language_hard: list[float] = []
        language_non_direct: list[float] = []
        total_hard = 0
        total_non_direct = 0
        total_invalid = 0
        for language in languages:
            if totals[canonical_id][language] != expected:
                raise ValueError(
                    f"unsafe consensus coverage drift for {canonical_id}/{language}"
                )
            labels = counts[canonical_id][language]
            hard = labels["HARD_REFUSE"]
            non_direct = hard + labels["SOFT"] + labels["PARTIAL"]
            language_hard.append(hard / expected)
            language_non_direct.append(non_direct / expected)
            total_hard += hard
            total_non_direct += non_direct
            total_invalid += expected - valid[canonical_id][language]
        diagnostics[canonical_id] = {
            "category_id": category,
            "min_language_hard_refuse_rate": min(language_hard),
            "overall_hard_refuse_rate": total_hard / (expected * len(languages)),
            "min_language_non_direct_rate": min(language_non_direct),
            "overall_non_direct_rate": total_non_direct
            / (expected * len(languages)),
            "invalid": total_invalid,
        }
        grouped[category].append(canonical_id)

    rankings = {
        category: sorted(
            ids,
            key=lambda canonical_id: (
                -float(
                    diagnostics[canonical_id]["min_language_hard_refuse_rate"]
                ),
                -float(diagnostics[canonical_id]["overall_hard_refuse_rate"]),
                -float(diagnostics[canonical_id]["min_language_non_direct_rate"]),
                -float(diagnostics[canonical_id]["overall_non_direct_rate"]),
                int(diagnostics[canonical_id]["invalid"]),
                canonical_id,
            ),
        )
        for category, ids in sorted(grouped.items())
    }
    return rankings, diagnostics


def _allocate_category_pools(
    rankings: Mapping[str, Sequence[str]],
    pool_sizes: Sequence[tuple[str, int]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    offsets = {category: 0 for category in rankings}
    remaining = {category: len(ids) for category, ids in rankings.items()}
    pools: dict[str, list[str]] = {}
    allocation = {
        category: {"source": len(ids)} for category, ids in rankings.items()
    }
    for pool, size in pool_sizes:
        quotas = hamilton_category_quotas(remaining, selected_rows=size)
        selected: list[str] = []
        for category in sorted(rankings):
            start = offsets[category]
            end = start + quotas[category]
            selected.extend(rankings[category][start:end])
            offsets[category] = end
            remaining[category] -= quotas[category]
            allocation[category][pool] = quotas[category]
        if len(selected) != size:
            raise AssertionError(f"{pool} allocation size mismatch")
        pools[pool] = selected
    pools["reserve"] = [
        canonical_id
        for category in sorted(rankings)
        for canonical_id in rankings[category][offsets[category] :]
    ]
    for category in sorted(rankings):
        allocation[category]["reserve"] = remaining[category]
    return pools, allocation


def _source_lookup(
    by_language: Mapping[str, Sequence[tuple[bytes, dict[str, Any]]]],
) -> dict[str, dict[str, tuple[bytes, dict[str, Any]]]]:
    return {
        language: {
            str(row["canonical_id"]): (raw, row) for raw, row in rows
        }
        for language, rows in by_language.items()
    }


def _category_filename(category: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", category).strip("._")
    if not normalized:
        raise ValueError(f"category cannot be represented as a filename: {category!r}")
    return normalized


def _write_pool(
    output_root: Path,
    *,
    pool: str,
    direction: str,
    ids: Sequence[str],
    sources: Mapping[str, Mapping[str, tuple[bytes, dict[str, Any]]]],
    languages: Sequence[str],
    files: dict[str, dict[str, object]],
) -> dict[str, int]:
    category_rows: dict[str, list[bytes]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    for language in languages:
        path = output_root / f"{pool}_{language}_{direction}_{len(ids)}.jsonl"
        raw_rows: list[bytes] = []
        for canonical_id in ids:
            raw, row = sources[language][canonical_id]
            raw_rows.append(raw)
            category = str(row["category_id"])
            category_rows[category].append(raw)
            if language == languages[0]:
                category_counts[category] += 1
        path.write_bytes(b"".join(raw_rows))
        files[path.name] = {
            "pool": pool,
            "direction_class": direction,
            "language": language,
            "rows": len(ids),
            "sha256": _sha256(path),
            "id_order_sha256": _canonical_sha256(list(ids)),
        }
    category_root = output_root / "categories" / pool / direction
    category_root.mkdir(parents=True, exist_ok=True)
    for category, raw_rows in sorted(category_rows.items()):
        path = category_root / f"{_category_filename(category)}.jsonl"
        path.write_bytes(b"".join(raw_rows))
        relative = path.relative_to(output_root).as_posix()
        files[relative] = {
            "pool": pool,
            "direction_class": direction,
            "category_id": category,
            "rows": len(raw_rows),
            "canonical_rows": category_counts[category],
            "sha256": _sha256(path),
        }
    return dict(sorted(category_counts.items()))


def build_four_language_corpus(
    *,
    output_root: str | Path,
    safe_ranked_root: str | Path,
    safe_map_root: str | Path,
    unsafe_root: str | Path,
    unsafe_consensus_path: str | Path,
    languages: Sequence[str] = ("en", "ru", "zh", "ja"),
    map_rows: int = 1000,
    trial_rows: int = 400,
    final_rows: int = 200,
) -> dict[str, object]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    normalized_languages = tuple(str(language).lower() for language in languages)
    if normalized_languages != ("en", "ru", "zh", "ja"):
        raise ValueError("languages must be exactly en, ru, zh, ja")
    safe_ranked = _load_aligned_source(
        Path(safe_ranked_root).resolve(),
        direction="safe",
        languages=normalized_languages,
    )
    safe_map = _load_aligned_source(
        Path(safe_map_root).resolve(),
        direction="safe",
        languages=normalized_languages,
    )
    unsafe = _load_aligned_source(
        Path(unsafe_root).resolve(),
        direction="unsafe",
        languages=normalized_languages,
    )
    safe_ranked_ids = [
        str(row["canonical_id"]) for _, row in safe_ranked[normalized_languages[0]]
    ]
    if len(safe_ranked_ids) < trial_rows + final_rows:
        raise ValueError("ranked SAFE source is smaller than trial+final")
    safe_pools = {
        "trial": safe_ranked_ids[:trial_rows],
        "final": safe_ranked_ids[trial_rows : trial_rows + final_rows],
    }
    safe_map_rankings = _category_rankings_from_source(
        safe_map[normalized_languages[0]],
        namespace="heretic_moe_4lang_v4|safe|map",
    )
    safe_map_pools, safe_map_allocation = _allocate_category_pools(
        safe_map_rankings,
        (("map", map_rows),),
    )
    safe_pools["map"] = safe_map_pools["map"]
    safe_pools["reserve"] = safe_map_pools["reserve"]

    unsafe_rankings, unsafe_diagnostics = _unsafe_category_rankings(
        Path(unsafe_consensus_path).resolve(),
        unsafe[normalized_languages[0]],
        languages=normalized_languages,
    )
    unsafe_pools, unsafe_allocation = _allocate_category_pools(
        unsafe_rankings,
        (("trial", trial_rows), ("final", final_rows), ("map", map_rows)),
    )
    for direction, pools in (("safe", safe_pools), ("unsafe", unsafe_pools)):
        names = ("map", "trial", "final")
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if set(pools[left]) & set(pools[right]):
                    raise ValueError(f"{direction} pool overlap: {left}/{right}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    files: dict[str, dict[str, object]] = {}
    try:
        safe_ranked_lookup = _source_lookup(safe_ranked)
        safe_map_lookup = _source_lookup(safe_map)
        unsafe_lookup = _source_lookup(unsafe)
        category_counts: dict[str, dict[str, dict[str, int]]] = {
            "safe": {},
            "unsafe": {},
        }
        for pool in ("map", "trial", "final"):
            category_counts["safe"][pool] = _write_pool(
                temporary,
                pool=pool,
                direction="safe",
                ids=safe_pools[pool],
                sources=safe_map_lookup if pool == "map" else safe_ranked_lookup,
                languages=normalized_languages,
                files=files,
            )
            category_counts["unsafe"][pool] = _write_pool(
                temporary,
                pool=pool,
                direction="unsafe",
                ids=unsafe_pools[pool],
                sources=unsafe_lookup,
                languages=normalized_languages,
                files=files,
            )
        counts = {
            "map": {"safe": map_rows, "unsafe": map_rows},
            "trial": {"safe": trial_rows, "unsafe": trial_rows},
            "final": {"safe": final_rows, "unsafe": final_rows},
            "reserve": {
                "safe": len(safe_pools["reserve"]),
                "unsafe": len(unsafe_pools["reserve"]),
            },
        }
        memberships = {
            "safe": {pool: list(ids) for pool, ids in safe_pools.items()},
            "unsafe": {pool: list(ids) for pool, ids in unsafe_pools.items()},
        }
        membership_path = temporary / "membership.json"
        membership_path.write_text(
            json.dumps(memberships, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        quality = {
            canonical_id: unsafe_diagnostics[canonical_id]
            for pool in ("trial", "final", "map", "reserve")
            for canonical_id in unsafe_pools[pool]
        }
        quality_path = temporary / "unsafe_quality.json"
        quality_path.write_text(
            json.dumps(quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest: dict[str, object] = {
            "schema_version": 4,
            "status": "PASS",
            "languages": list(normalized_languages),
            "counts": counts,
            "language_cells": {
                direction: sum(
                    int(counts[pool][direction]) * len(normalized_languages)
                    for pool in ("map", "trial", "final")
                )
                for direction in ("safe", "unsafe")
            },
            "category_counts": category_counts,
            "allocations": {
                "safe_map": safe_map_allocation,
                "unsafe": unsafe_allocation,
            },
            "cross_pool_overlap": {"safe": 0, "unsafe": 0},
            "files": dict(sorted(files.items())),
            "metadata_files": {
                membership_path.name: _sha256(membership_path),
                quality_path.name: _sha256(quality_path),
            },
            "source_hashes": {
                "safe_ranked_manifest": _sha256(
                    Path(safe_ranked_root).resolve() / "manifest.json"
                ),
                "safe_map_manifest": _sha256(
                    Path(safe_map_root).resolve() / "manifest.json"
                ),
                "unsafe_manifest": _sha256(
                    Path(unsafe_root).resolve() / "manifest.json"
                ),
                "unsafe_consensus": _sha256(
                    Path(unsafe_consensus_path).resolve()
                ),
            },
        }
        _assert_text_free(manifest)
        manifest["contract_sha256"] = _canonical_sha256(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m heretic.four_language_corpus")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--safe-ranked-root", required=True, type=Path)
    parser.add_argument("--safe-map-root", required=True, type=Path)
    parser.add_argument("--unsafe-root", required=True, type=Path)
    parser.add_argument("--unsafe-consensus", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_four_language_corpus(
        output_root=args.output_root,
        safe_ranked_root=args.safe_ranked_root,
        safe_map_root=args.safe_map_root,
        unsafe_root=args.unsafe_root,
        unsafe_consensus_path=args.unsafe_consensus,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "contract_sha256": manifest["contract_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
