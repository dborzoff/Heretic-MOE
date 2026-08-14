# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-free full-space language distances and deterministic k-medoids."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _positions(
    index: list[dict[str, object]],
) -> tuple[
    list[str],
    list[str],
    dict[tuple[str, str], list[int]],
    dict[tuple[str, str, str], list[int]],
]:
    languages = sorted({str(row["language"]).lower() for row in index})
    if len(languages) < 2:
        raise ValueError("at least two languages are required")
    language_direction: dict[tuple[str, str], list[int]] = defaultdict(list)
    category_cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    categories: set[str] = set()
    for position, row in enumerate(index):
        language = str(row["language"]).lower()
        direction = str(row["direction_class"]).lower()
        if direction not in {"safe", "unsafe"}:
            raise ValueError("direction_class must be safe or unsafe")
        raw_categories = row.get("category_ids", [row["category_id"]])
        if not isinstance(raw_categories, list) or not raw_categories:
            raise ValueError("category_ids must be a non-empty list")
        category_values = tuple(dict.fromkeys(str(value) for value in raw_categories))
        language_direction[(language, direction)].append(position)
        for category in category_values:
            categories.add(category)
            category_cells[(language, direction, category)].append(position)
    for language in languages:
        for direction in ("safe", "unsafe"):
            if not language_direction[(language, direction)]:
                raise ValueError("every language must contain both directions")
    return languages, sorted(categories), language_direction, category_cells


def _mean(x: Tensor, positions: list[int]) -> Tensor:
    return x[positions].mean(dim=0)


def _cosine_distance(left: Tensor, right: Tensor) -> float:
    if float(torch.linalg.vector_norm(left)) == 0.0:
        return float(torch.linalg.vector_norm(right) != 0.0)
    if float(torch.linalg.vector_norm(right)) == 0.0:
        return 1.0
    cosine = float(F.cosine_similarity(left, right, dim=0).clamp(-1.0, 1.0))
    return (1.0 - cosine) / 2.0


def _normalize(matrix: Tensor) -> Tensor:
    maximum = float(torch.max(matrix))
    return matrix / maximum if maximum > 0.0 else torch.zeros_like(matrix)


def build_language_distance_report(
    index: list[dict[str, object]], residuals: Tensor
) -> dict[str, Any]:
    """Measure language divergence in full hidden space, never projected 3D."""

    if residuals.ndim != 3 or len(index) != residuals.shape[0] or not index:
        raise ValueError("index/residual shape mismatch")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals contain non-finite values")
    languages, categories, language_direction, category_cells = _positions(index)
    count = len(languages)
    component_totals = {
        name: torch.zeros((count, count), dtype=torch.float64)
        for name in ("direction", "language_offset", "category_interaction")
    }
    layer_components: list[dict[str, object]] = []
    raw_layer_weights: list[float] = []
    hidden_scale = math.sqrt(int(residuals.shape[2]))
    direction_positions = {
        direction: [
            position
            for position, row in enumerate(index)
            if str(row["direction_class"]).lower() == direction
        ]
        for direction in ("safe", "unsafe")
    }

    for layer in range(int(residuals.shape[1])):
        x = residuals[:, layer, :].detach().to(device="cpu", dtype=torch.float32)
        global_safe = _mean(x, direction_positions["safe"])
        global_unsafe = _mean(x, direction_positions["unsafe"])
        raw_layer_weights.append(
            float(torch.linalg.vector_norm(global_unsafe - global_safe)) / hidden_scale
        )
        means = {
            key: _mean(x, positions) for key, positions in language_direction.items()
        }
        language_means = {
            language: (means[(language, "safe")] + means[(language, "unsafe")]) / 2
            for language in languages
        }
        directions = {
            language: means[(language, "unsafe")] - means[(language, "safe")]
            for language in languages
        }
        category_effects = {
            key: _mean(x, positions) - means[(key[0], key[1])]
            for key, positions in category_cells.items()
        }
        matrices = {
            name: torch.zeros((count, count), dtype=torch.float64)
            for name in component_totals
        }
        for left_position, left in enumerate(languages):
            for right_position in range(left_position + 1, count):
                right = languages[right_position]
                matrices["direction"][left_position, right_position] = (
                    matrices["direction"][right_position, left_position]
                ) = _cosine_distance(directions[left], directions[right])
                matrices["language_offset"][left_position, right_position] = (
                    matrices["language_offset"][right_position, left_position]
                ) = float(
                    torch.linalg.vector_norm(language_means[left] - language_means[right])
                ) / hidden_scale
                shared_keys = [
                    (direction, category)
                    for direction in ("safe", "unsafe")
                    for category in categories
                    if (left, direction, category) in category_effects
                    and (right, direction, category) in category_effects
                ]
                category_distance = (
                    sum(
                        float(
                            torch.linalg.vector_norm(
                                category_effects[(left, direction, category)]
                                - category_effects[(right, direction, category)]
                            )
                        )
                        / hidden_scale
                        for direction, category in shared_keys
                    )
                    / len(shared_keys)
                    if shared_keys
                    else 0.0
                )
                matrices["category_interaction"][left_position, right_position] = (
                    matrices["category_interaction"][right_position, left_position]
                ) = category_distance
        normalized = {name: _normalize(value) for name, value in matrices.items()}
        layer_components.append(
            {
                "layer": layer,
                "weight_signal": raw_layer_weights[-1],
                "components": {
                    name: value.tolist() for name, value in normalized.items()
                },
            }
        )

    weights = torch.tensor(raw_layer_weights, dtype=torch.float64)
    if float(torch.sum(weights)) == 0.0:
        weights.fill_(1.0 / len(weights))
    else:
        weights /= torch.sum(weights)
    for layer, layer_report in enumerate(layer_components):
        layer_report["normalized_weight"] = float(weights[layer])
        for name in component_totals:
            component_totals[name] += weights[layer] * torch.tensor(
                layer_report["components"][name], dtype=torch.float64
            )
    distances = (
        0.60 * component_totals["direction"]
        + 0.25 * component_totals["language_offset"]
        + 0.15 * component_totals["category_interaction"]
    )
    distances.fill_diagonal_(0.0)
    return {
        "schema_version": 1,
        "status": "PASS",
        "method": "full_space_language_geometry_v1",
        "rows": len(index),
        "layers": int(residuals.shape[1]),
        "hidden_size": int(residuals.shape[2]),
        "languages": languages,
        "categories": categories,
        "weights": {
            "direction": 0.60,
            "language_offset": 0.25,
            "category_interaction": 0.15,
        },
        "distances": distances.tolist(),
        "component_distances": {
            name: value.tolist() for name, value in component_totals.items()
        },
        "layer_diagnostics": layer_components,
    }


def _assignment(distances: Tensor, medoids: tuple[int, ...]) -> list[int]:
    return [
        min(medoids, key=lambda medoid: (float(distances[row, medoid]), medoid))
        for row in range(int(distances.shape[0]))
    ]


def _cost(distances: Tensor, medoids: tuple[int, ...]) -> float:
    assignment = _assignment(distances, medoids)
    return sum(float(distances[row, medoid]) for row, medoid in enumerate(assignment))


def _k_medoids(distances: Tensor, k: int, fixed: int) -> tuple[tuple[int, ...], list[int]]:
    count = int(distances.shape[0])
    if not 1 <= k <= count:
        raise ValueError("k is outside the language count")
    medoids = [fixed]
    while len(medoids) < k:
        candidates = [value for value in range(count) if value not in medoids]
        medoids.append(
            max(
                candidates,
                key=lambda value: (
                    min(float(distances[value, medoid]) for medoid in medoids),
                    -value,
                ),
            )
        )
    current = tuple(sorted(medoids))
    while True:
        current_cost = _cost(distances, current)
        best = current
        best_cost = current_cost
        for outgoing in current:
            if outgoing == fixed:
                continue
            for incoming in range(count):
                if incoming in current:
                    continue
                proposal = tuple(sorted((set(current) - {outgoing}) | {incoming}))
                proposal_cost = _cost(distances, proposal)
                if (proposal_cost, proposal) < (best_cost, best):
                    best, best_cost = proposal, proposal_cost
        if best == current:
            return current, _assignment(distances, current)
        current = best


def _silhouette(distances: Tensor, assignment: list[int]) -> float:
    clusters: dict[int, list[int]] = defaultdict(list)
    for row, medoid in enumerate(assignment):
        clusters[medoid].append(row)
    values = []
    for row, medoid in enumerate(assignment):
        own = [value for value in clusters[medoid] if value != row]
        if not own:
            values.append(0.0)
            continue
        a = sum(float(distances[row, value]) for value in own) / len(own)
        b = min(
            sum(float(distances[row, value]) for value in members) / len(members)
            for other, members in clusters.items()
            if other != medoid
        )
        values.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return sum(values) / len(values)


def combine_language_distance_reports(
    reports: Iterable[dict[str, Any]],
    *,
    min_k: int = 4,
    max_k: int = 6,
    fixed_language: str = "en",
) -> dict[str, Any]:
    """Average model distances and select deterministic representative languages."""

    reports = list(reports)
    if not reports:
        raise ValueError("at least one language distance report is required")
    languages = [str(value) for value in reports[0]["languages"]]
    if fixed_language not in languages:
        raise ValueError("fixed language is missing")
    matrices = []
    for report in reports:
        if [str(value) for value in report["languages"]] != languages:
            raise ValueError("language axes differ between reports")
        matrix = torch.tensor(report["distances"], dtype=torch.float64)
        if matrix.shape != (len(languages), len(languages)):
            raise ValueError("language distance matrix shape mismatch")
        matrices.append(matrix)
    distances = torch.stack(matrices).mean(dim=0)
    fixed = languages.index(fixed_language)
    candidates: dict[str, dict[str, object]] = {}
    upper = min(max_k, len(languages))
    if min_k > upper:
        raise ValueError("k range is outside the language count")
    for k in range(min_k, upper + 1):
        medoids, assignment = _k_medoids(distances, k, fixed)
        clusters = {
            languages[medoid]: [
                languages[row]
                for row, assigned_medoid in enumerate(assignment)
                if assigned_medoid == medoid
            ]
            for medoid in medoids
        }
        candidates[str(k)] = {
            "k": k,
            "medoids": [languages[value] for value in medoids],
            "clusters": clusters,
            "cost": _cost(distances, medoids),
            "mean_silhouette": _silhouette(distances, assignment),
        }
    chosen = max(
        candidates.values(),
        key=lambda value: (float(value["mean_silhouette"]), -int(value["k"])),
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "method": "mean_full_space_distance_plus_fixed_en_pam_v1",
        "models": len(reports),
        "languages": languages,
        "distances": distances.tolist(),
        "fixed_language": fixed_language,
        "candidates": candidates,
        "recommended_k": int(chosen["k"]),
        "recommended_languages": list(chosen["medoids"]),
    }


def write_language_selection_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write machine and human-readable text-free language selection artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "language_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    (output_dir / "report.html").write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Heretic-MOE language selection</title><style>
:root{color-scheme:dark;font:14px system-ui;background:#080b12;color:#e8edf7}
body{margin:24px}table{border-collapse:collapse}th,td{padding:6px;border:1px solid #283149;text-align:center}
.card{background:#111725;border:1px solid #283149;border-radius:8px;padding:14px;margin:12px 0}.medoid{color:#65e6a7}
</style></head><body><h1>Language distance heatmap</h1><div id='summary' class='card'></div><div id='heatmap'></div><div id='clusters'></div>
<script>const DATA="""
        + payload
        + """;
const langs=DATA.languages,dist=DATA.distances,max=Math.max(...dist.flat(),1e-12);
document.getElementById('summary').textContent=`${DATA.models} model(s) · recommended_languages: ${DATA.recommended_languages.join(', ')} · k=${DATA.recommended_k}`;
let table='<table><tr><th></th>'+langs.map(x=>`<th>${x}</th>`).join('')+'</tr>';
for(let i=0;i<langs.length;i++){table+=`<tr><th>${langs[i]}</th>`+dist[i].map(v=>`<td style="background:rgba(255,83,100,${v/max})">${v.toFixed(3)}</td>`).join('')+'</tr>'}table+='</table>';document.getElementById('heatmap').innerHTML=table;
let cards='';for(const [k,value] of Object.entries(DATA.candidates)){cards+=`<div class="card"><h2>k=${k} · silhouette ${value.mean_silhouette.toFixed(3)}</h2>`;for(const [medoid,members] of Object.entries(value.clusters)){cards+=`<div><span class="medoid">${medoid}</span>: ${members.join(', ')}</div>`}cards+='</div>'}document.getElementById('clusters').innerHTML=cards;
</script></body></html>\n""",
        encoding="utf-8",
    )
