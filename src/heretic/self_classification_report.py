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
    for variant in PromptVariant:
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

    winner = max(PromptVariant, key=rank)
    return winner, summary


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
