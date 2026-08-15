# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free aggregation and HTML reports for self-classification."""

from __future__ import annotations

import csv
import html
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .self_classification import BehaviorClass, PromptVariant

_CLASS_NAMES = tuple(value.value for value in BehaviorClass) + ("INVALID",)
_CODE_VARIANTS = {
    PromptVariant.CODE_PERMUTED.value,
    PromptVariant.CODE_SHIFT_1.value,
    PromptVariant.CODE_SHIFT_2.value,
    PromptVariant.CODE_SHIFT_3.value,
}
_WORD_VARIANTS = {
    PromptVariant.WORD_ORDER_0.value,
    PromptVariant.WORD_ORDER_1.value,
    PromptVariant.WORD_ORDER_2.value,
    PromptVariant.WORD_ORDER_3.value,
}


def _classification(raw: Mapping[str, object]) -> str:
    value = raw.get("classification")
    return str(value) if value is not None else "INVALID"


def _count_classes(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(_classification(row) for row in rows)
    return {name: int(counts.get(name, 0)) for name in _CLASS_NAMES}


def _modal(values: Sequence[str]) -> str | None:
    if not values:
        return None
    counts = Counter(values)
    best = counts.most_common()
    return best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None


def select_prompt_variant(
    rows: Sequence[Mapping[str, object]],
) -> tuple[PromptVariant, dict[str, dict[str, object]]]:
    if not rows:
        raise ValueError("pilot rows must be non-empty")
    by_item: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        variant = str(row["variant"])
        by_variant[variant].append(row)
        by_item[(str(row["model_id"]), str(row["row_id"]))].append(row)

    majority: dict[tuple[str, str], str | None] = {}
    for key, values in by_item.items():
        majority[key] = _modal(
            [
                str(row["classification"])
                for row in values
                if bool(row.get("valid")) and row.get("classification") is not None
            ]
        )

    summary: dict[str, dict[str, object]] = {}
    present_variants = tuple(
        variant for variant in PromptVariant if variant.value in by_variant
    )
    for variant in present_variants:
        values = by_variant.get(variant.value, [])
        if not values:
            raise ValueError(f"pilot is missing variant: {variant.value}")
        valid_rows = [row for row in values if bool(row.get("valid"))]
        comparable = [
            row
            for row in valid_rows
            if majority[(str(row["model_id"]), str(row["row_id"]))] is not None
        ]
        agreement = (
            sum(
                str(row["classification"])
                == majority[(str(row["model_id"]), str(row["row_id"]))]
                for row in comparable
            )
            / len(comparable)
            if comparable
            else 0.0
        )
        translation_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in valid_rows:
            translation_groups[
                (str(row["model_id"]), str(row["canonical_id"]))
            ].append(str(row["classification"]))
        translation_scores = [
            max(Counter(group).values()) / len(group)
            for group in translation_groups.values()
            if group
        ]
        token_values = [int(row.get("output_tokens", 0)) for row in valid_rows]
        class_counts = _count_classes(values)
        dominant = max(class_counts[name] for name in _CLASS_NAMES[:-1]) / len(values)
        summary[variant.value] = {
            "rows": len(values),
            "valid": len(valid_rows),
            "valid_rate": len(valid_rows) / len(values),
            "majority_agreement": agreement,
            "translation_agreement": (
                statistics.fmean(translation_scores) if translation_scores else 0.0
            ),
            "dominant_class_rate": dominant,
            "mean_output_tokens": (
                statistics.fmean(token_values) if token_values else 0.0
            ),
            "class_counts": class_counts,
        }

    def rank(variant: PromptVariant) -> tuple[float, float, float, float]:
        item = summary[variant.value]
        return (
            float(item["valid_rate"]),
            float(item["majority_agreement"]),
            float(item["translation_agreement"]),
            -float(item["mean_output_tokens"]),
        )

    winner = max(present_variants, key=rank)
    return winner, summary


def _strict_label(
    values: Sequence[Mapping[str, object]],
    variants: set[str],
) -> str | None:
    selected = [row for row in values if str(row["variant"]) in variants]
    if len(selected) != len(variants):
        return None
    labels = [
        str(row["classification"])
        for row in selected
        if bool(row.get("valid")) and row.get("classification") is not None
    ]
    return labels[0] if len(labels) == len(variants) and len(set(labels)) == 1 else None


def _consensus_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = str(row["model_id"]), str(row["row_id"]), str(row["variant"])
        if key in seen:
            raise ValueError(f"duplicate consensus result key: {key}")
        seen.add(key)
        grouped[(key[0], key[1])].append(row)
    result: list[dict[str, object]] = []
    for (model_id, row_id), values in sorted(grouped.items()):
        first = values[0]
        code = _strict_label(values, _CODE_VARIANTS)
        word = _strict_label(values, _WORD_VARIANTS)
        strict = code if code is not None and code == word else None
        result.append(
            {
                "model_id": model_id,
                "canonical_id": str(first["canonical_id"]),
                "row_id": row_id,
                "language": str(first["language"]),
                "category_ids": list(first.get("category_ids", [])),
                "direction_class": str(first["direction_class"]),
                "code_4of4": code,
                "word_4of4": word,
                "strict_8of8": strict,
            }
        )
    return result


def _consensus_model_rows(
    consensus: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in consensus:
        grouped[str(row["model_id"])].append(row)
    result = []
    for model_id, values in sorted(grouped.items()):
        counts = Counter(
            str(row["strict_8of8"])
            for row in values
            if row.get("strict_8of8") is not None
        )
        result.append(
            {
                "model_id": model_id,
                "groups": len(values),
                "code_4of4": sum(row.get("code_4of4") is not None for row in values),
                "word_4of4": sum(row.get("word_4of4") is not None for row in values),
                "strict_8of8": sum(row.get("strict_8of8") is not None for row in values),
                "strict_class_counts": {
                    name: int(counts.get(name, 0)) for name in _CLASS_NAMES[:-1]
                },
            }
        )
    return result


def build_consensus_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    comparison_model_ids: set[str] | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("consensus rows must be non-empty")
    comparison = set(comparison_model_ids or ())
    consensus = _consensus_rows(rows)
    by_model = _consensus_model_rows(consensus)
    clean_models = sorted(
        {str(row["model_id"]) for row in consensus} - comparison
    )
    by_item: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in consensus:
        if str(row["model_id"]) in clean_models:
            by_item[str(row["row_id"])].append(row)
    strict_unanimous = 0
    fully_covered = 0
    for values in by_item.values():
        if len(values) != len(clean_models):
            continue
        fully_covered += 1
        labels = [row.get("strict_8of8") for row in values]
        strict_unanimous += int(
            all(label is not None for label in labels) and len(set(labels)) == 1
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "source_rows": len(rows),
        "consensus_groups": len(consensus),
        "comparison_model_ids": sorted(comparison),
        "by_model": by_model,
        "clean_panel": {
            "models": len(clean_models),
            "model_ids": clean_models,
            "fully_covered_groups": fully_covered,
            "strict_unanimous_groups": strict_unanimous,
            "strict_unanimous_rate": (
                strict_unanimous / fully_covered if fully_covered else 0.0
            ),
        },
    }


def _group_summary(
    rows: Sequence[Mapping[str, object]],
    key_name: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if key_name == "category":
            values = row.get("category_ids")
            value = str(values[0]) if isinstance(values, list) and values else "unknown"
        else:
            value = str(row[key_name])
        grouped[value].append(row)
    result = []
    for value, values in sorted(grouped.items()):
        valid = sum(bool(row.get("valid")) for row in values)
        result.append(
            {
                key_name: value,
                "rows": len(values),
                "valid": valid,
                "invalid": len(values) - valid,
                "class_counts": _count_classes(values),
            }
        )
    return result


def _exact_group_agreement(
    rows: Sequence[Mapping[str, object]],
    group_fields: tuple[str, ...],
) -> dict[str, object]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("valid")) or row.get("classification") is None:
            continue
        groups[tuple(str(row[field]) for field in group_fields)].append(
            str(row["classification"])
        )
    complete = [values for values in groups.values() if len(values) > 1]
    exact = sum(len(set(values)) == 1 for values in complete)
    return {
        "groups": len(complete),
        "exact": exact,
        "exact_rate": exact / len(complete) if complete else 0.0,
    }


def build_model_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("model rows must be non-empty")
    valid = sum(bool(row.get("valid")) for row in rows)
    return {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(rows),
        "valid": valid,
        "invalid": len(rows) - valid,
        "class_counts": _count_classes(rows),
        "by_model": _group_summary(rows, "model_id"),
        "by_language": _group_summary(rows, "language"),
        "by_category": _group_summary(rows, "category"),
        "by_direction": _group_summary(rows, "direction_class"),
        "translation_agreement": _exact_group_agreement(
            rows, ("model_id", "canonical_id", "variant")
        ),
        "model_agreement": _exact_group_agreement(
            rows, ("row_id", "variant")
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _table_html(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _page(title: str, blocks: Sequence[str]) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px system-ui;background:#0c111b;color:#e7edf7;margin:24px}}h1,h2{{color:#fff}}table{{border-collapse:collapse;margin:12px 0 28px;width:100%}}th,td{{border:1px solid #2f3a4d;padding:6px 8px;text-align:left}}th{{background:#172033;position:sticky;top:0}}tr:nth-child(even){{background:#111827}}.ok{{color:#60d394}}</style></head><body><h1>{html.escape(title)}</h1>{''.join(blocks)}</body></html>"""


def write_pilot_reports(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    winner, variants = select_prompt_variant(rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "winner": winner.value,
        "rows": len(rows),
        "variants": variants,
    }
    _write_json(output_dir / "pilot_summary.json", summary)
    table_rows = [
        [
            name,
            item["rows"],
            f"{float(item['valid_rate']):.2%}",
            f"{float(item['majority_agreement']):.2%}",
            f"{float(item['translation_agreement']):.2%}",
            f"{float(item['mean_output_tokens']):.2f}",
        ]
        for name, item in variants.items()
    ]
    page = _page(
        "Self-classification pilot",
        [
            f"<p class=ok>Winner: {html.escape(winner.value)}</p>",
            _table_html(
                [
                    "Variant",
                    "Rows",
                    "Valid",
                    "Majority agreement",
                    "Translation agreement",
                    "Mean tokens",
                ],
                table_rows,
            ),
        ],
    )
    (output_dir / "pilot_report.html").write_text(page, encoding="utf-8")
    return summary


def _flatten_group_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dimension, field in (
        ("model", "by_model"),
        ("language", "by_language"),
        ("category", "by_category"),
        ("direction", "by_direction"),
    ):
        for item in summary[field]:
            name_key = {
                "model": "model_id",
                "language": "language",
                "category": "category",
                "direction": "direction_class",
            }[dimension]
            flat = {
                "dimension": dimension,
                "value": item[name_key],
                "rows": item["rows"],
                "valid": item["valid"],
                "invalid": item["invalid"],
            }
            flat.update(item["class_counts"])
            result.append(flat)
    return result


def write_model_reports(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_model_summary(rows)
    _write_json(output_dir / "model_summary.json", summary)
    flat = _flatten_group_rows(summary)
    headers = [
        "dimension",
        "value",
        "rows",
        "valid",
        "invalid",
        *_CLASS_NAMES,
    ]
    with (output_dir / "model_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flat)
    page = _page(
        "Self-classification model comparison",
        [
            f"<p>Rows: {summary['rows']} | Valid: {summary['valid']} | Invalid: {summary['invalid']}</p>",
            _table_html(headers, [[item[key] for key in headers] for item in flat]),
        ],
    )
    (output_dir / "report.html").write_text(page, encoding="utf-8")
    return summary


def write_consensus_reports(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    comparison_model_ids: set[str] | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_consensus_summary(
        rows,
        comparison_model_ids=comparison_model_ids,
    )
    _write_json(output_dir / "consensus_summary.json", summary)
    table_rows = []
    for item in summary["by_model"]:
        table_rows.append(
            [
                item["model_id"],
                item["groups"],
                item["code_4of4"],
                item["word_4of4"],
                item["strict_8of8"],
                *[
                    item["strict_class_counts"][name]
                    for name in _CLASS_NAMES[:-1]
                ],
            ]
        )
    headers = [
        "Model",
        "Groups",
        "Code 4/4",
        "Word 4/4",
        "Strict 8/8",
        *_CLASS_NAMES[:-1],
    ]
    page = _page(
        "Eight-pass strict consensus",
        [
            (
                "<p>Clean models: "
                f"{summary['clean_panel']['models']} | Strict clean unanimity: "
                f"{float(summary['clean_panel']['strict_unanimous_rate']):.2%}</p>"
            ),
            _table_html(headers, table_rows),
        ],
    )
    (output_dir / "consensus_report.html").write_text(page, encoding="utf-8")
    return summary
