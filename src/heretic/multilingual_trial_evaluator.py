# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-generation evaluator for multilingual Heretic-MOE trials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .language_map_data import GeometryRow
from .multilingual_trial_metrics import (
    MultilingualTrialMetrics,
    aggregate_safe_ppl,
    compose_trial_metrics,
)
from .srg_calibration import relative_group_summary, relative_score
from .trial_geometry_metrics import evaluate_trial_geometry
from .utils import Prompt


@dataclass(frozen=True)
class TrialMeasurement:
    trial_number: int
    rows: int
    safe_rows: int
    unsafe_rows: int
    metrics: MultilingualTrialMetrics
    diagnostics: Mapping[str, object]
    private_records_sha256: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "trial_number": self.trial_number,
            "rows": self.rows,
            "safe_rows": self.safe_rows,
            "unsafe_rows": self.unsafe_rows,
            "metrics": asdict(self.metrics),
            "diagnostics": dict(self.diagnostics),
            "private_records_sha256": self.private_records_sha256,
        }


class FrozenMultilingualTrialEvaluator:
    """Resolve immutable global trial IDs and delegate one-pass measurement."""

    def __init__(
        self,
        *,
        model: Any,
        trial_rows: Sequence[GeometryRow],
        schedule_records: Sequence[Mapping[str, object]],
        clean_records: Sequence[Mapping[str, object]],
        refusal_direction: Tensor,
        layer_reliability: Tensor,
        srg_scorer: Any,
        srg_profile: Mapping[str, object],
        private_output_dir: str | Path,
        expected_per_direction: int = 400,
        expected_languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
        max_response_length: int = 100,
    ) -> None:
        self.model = model
        self._rows = {row.row_id: row for row in trial_rows}
        if len(self._rows) != len(trial_rows):
            raise ValueError("frozen trial rows have duplicate row IDs")
        self._schedule: dict[int, tuple[str, ...]] = {}
        for record in schedule_records:
            trial_number = record.get("trial_number")
            row_ids = record.get("row_ids")
            if (
                not isinstance(trial_number, int)
                or trial_number < 0
                or trial_number in self._schedule
                or not isinstance(row_ids, list)
                or any(row_id not in self._rows for row_id in row_ids)
            ):
                raise ValueError("frozen trial schedule is invalid")
            self._schedule[trial_number] = tuple(str(row_id) for row_id in row_ids)
        if not self._schedule:
            raise ValueError("frozen trial schedule is empty")
        self.clean_records = tuple(clean_records)
        self.refusal_direction = refusal_direction
        self.layer_reliability = layer_reliability
        self.srg_scorer = srg_scorer
        self.srg_profile = dict(srg_profile)
        self._clean_srg_margins: dict[str, float] | None = None
        self.private_output_dir = Path(private_output_dir).resolve()
        self.expected_per_direction = expected_per_direction
        self.expected_languages = expected_languages
        if max_response_length <= 0:
            raise ValueError("max_response_length must be positive")
        self.max_response_length = max_response_length

    def _ensure_clean_srg_margins(self) -> dict[str, float]:
        if self._clean_srg_margins is not None:
            return self._clean_srg_margins
        trial_rows = tuple(self._rows.values())
        clean_by_id = _clean_by_row_id(trial_rows, self.clean_records)
        margins_by_id: dict[str, float] = {}
        for direction_class in ("safe", "unsafe"):
            direction_rows = tuple(
                row for row in trial_rows if row.direction == direction_class
            )
            prompts = [Prompt(system="", user=row.prompt) for row in direction_rows]
            responses = [
                str(clean_by_id[row.row_id]["clean_response"])
                for row in direction_rows
            ]
            margins = _margins(
                self.srg_scorer.score_responses(prompts, responses),
                len(direction_rows),
            )
            margins_by_id.update(
                zip(
                    (row.row_id for row in direction_rows),
                    margins,
                    strict=True,
                )
            )
        self._clean_srg_margins = margins_by_id
        return margins_by_id

    def evaluate(
        self,
        trial_number: int,
        *,
        residual_capture: Callable[[list[Prompt], Tensor], None] | None = None,
    ) -> TrialMeasurement:
        if trial_number not in self._schedule:
            raise KeyError(f"trial {trial_number} is not present in frozen schedule")
        rows = [self._rows[row_id] for row_id in self._schedule[trial_number]]
        return evaluate_multilingual_trial(
            trial_number=trial_number,
            model=self.model,
            rows=rows,
            clean_records=self.clean_records,
            refusal_direction=self.refusal_direction,
            layer_reliability=self.layer_reliability,
            srg_scorer=self.srg_scorer,
            srg_profile=self.srg_profile,
            clean_srg_margins=self._ensure_clean_srg_margins(),
            private_records_path=(
                self.private_output_dir / f"trial-{trial_number:06d}.jsonl"
            ),
            expected_per_direction=self.expected_per_direction,
            expected_languages=self.expected_languages,
            max_response_length=self.max_response_length,
            residual_capture=residual_capture,
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _clean_by_row_id(
    rows: Sequence[GeometryRow], clean_records: Sequence[Mapping[str, object]]
) -> dict[str, Mapping[str, object]]:
    records: dict[str, Mapping[str, object]] = {}
    for record in clean_records:
        row_id = record.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in records:
            raise ValueError("clean reference rows must have unique row IDs")
        records[row_id] = record
    expected = {row.row_id for row in rows}
    if not expected.issubset(records):
        raise ValueError("clean reference coverage does not match trial rows")
    return {row_id: records[row_id] for row_id in expected}


def _margins(score: object, expected: int) -> list[float]:
    diagnostics = getattr(score, "diagnostics", None)
    values = diagnostics.get("margins") if isinstance(diagnostics, Mapping) else None
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError("SRG scorer did not return aligned per-row margins")
    margins = [float(value) for value in values]
    if not bool(torch.isfinite(torch.tensor(margins, dtype=torch.float64)).all()):
        raise ValueError("SRG margins contain non-finite values")
    return margins


def evaluate_multilingual_trial(
    *,
    trial_number: int,
    model: Any,
    rows: Sequence[GeometryRow],
    clean_records: Sequence[Mapping[str, object]],
    refusal_direction: Tensor,
    layer_reliability: Tensor,
    srg_scorer: Any,
    srg_profile: Mapping[str, object],
    clean_srg_margins: Mapping[str, float] | None = None,
    private_records_path: str | Path,
    expected_per_direction: int = 400,
    expected_languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
    max_response_length: int = 100,
    residual_capture: Callable[[list[Prompt], Tensor], None] | None = None,
) -> TrialMeasurement:
    """Evaluate one scheduled trial without a second autoregressive pass."""

    total_started = time.perf_counter()
    ordered = tuple(rows)
    safe_rows = sum(row.direction == "safe" for row in ordered)
    unsafe_rows = sum(row.direction == "unsafe" for row in ordered)
    if (
        expected_per_direction <= 0
        or safe_rows != expected_per_direction
        or unsafe_rows != expected_per_direction
        or len({row.row_id for row in ordered}) != len(ordered)
    ):
        raise ValueError(
            "trial direction coverage differs from the configured "
            f"{expected_per_direction} SAFE and {expected_per_direction} UNSAFE rows"
        )
    languages = tuple(language.lower() for language in expected_languages)
    if (
        not languages
        or len(set(languages)) != len(languages)
        or expected_per_direction % len(languages) != 0
    ):
        raise ValueError("expected languages cannot balance the trial")
    expected_per_language = expected_per_direction // len(languages)
    for direction_class in ("safe", "unsafe"):
        counts = Counter(
            row.language for row in ordered if row.direction == direction_class
        )
        if counts != Counter(
            {language: expected_per_language for language in languages}
        ):
            raise ValueError("trial must be balanced by language in both directions")
    if trial_number < 0:
        raise ValueError("trial number must be nonnegative")
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    direction = refusal_direction.detach().to(torch.float32).cpu()
    if direction.ndim != 2 or not bool(torch.isfinite(direction).all()):
        raise ValueError("refusal direction must be a finite [layers,hidden] tensor")
    clean = _clean_by_row_id(ordered, clean_records)
    prompts = [Prompt(system="", user=row.prompt) for row in ordered]

    generation_started = time.perf_counter()
    responses, token_ids, residuals = (
        model.get_response_artifacts_with_prefill_residuals_batched(
            prompts,
            skip_special_tokens=True,
        )
    )
    generation_seconds = time.perf_counter() - generation_started
    if (
        len(responses) != len(ordered)
        or len(token_ids) != len(ordered)
        or residuals.shape != (len(ordered), *direction.shape)
    ):
        raise ValueError("trial generation artifacts are not aligned")
    if residual_capture is not None:
        residual_capture(prompts, residuals)
    candidate_projection = torch.einsum(
        "blh,lh->bl", residuals.to(torch.float32).cpu(), direction
    )
    clean_projection = torch.tensor(
        [clean[row.row_id]["clean_prompt_residual_projection"] for row in ordered],
        dtype=torch.float32,
    )

    srg_started = time.perf_counter()
    unsafe_positions = [
        index for index, row in enumerate(ordered) if row.direction == "unsafe"
    ]
    unsafe_prompts = [prompts[index] for index in unsafe_positions]
    clean_unsafe_responses = [
        str(clean[ordered[index].row_id]["clean_response"])
        for index in unsafe_positions
    ]
    candidate_unsafe_responses = [responses[index] for index in unsafe_positions]
    baseline_margins = (
        [float(clean_srg_margins[ordered[index].row_id]) for index in unsafe_positions]
        if clean_srg_margins is not None
        else _margins(
            srg_scorer.score_responses(unsafe_prompts, clean_unsafe_responses),
            unsafe_rows,
        )
    )
    candidate_margins = _margins(
        srg_scorer.score_responses(unsafe_prompts, candidate_unsafe_responses),
        unsafe_rows,
    )
    srg_groups = [
        (ordered[index].language, ordered[index].category_id)
        for index in unsafe_positions
    ]
    srg = relative_score(
        baseline_margins,
        candidate_margins,
        dict(srg_profile),
        groups=srg_groups,
    )
    srg_group_summary = relative_group_summary(
        baseline_margins,
        candidate_margins,
        dict(srg_profile),
        groups=srg_groups,
    )

    safe_positions = [
        index for index, row in enumerate(ordered) if row.direction == "safe"
    ]
    safe_prompts = [prompts[index] for index in safe_positions]
    clean_safe_responses = [
        str(clean[ordered[index].row_id]["clean_response"]) for index in safe_positions
    ]
    candidate_safe_responses = [responses[index] for index in safe_positions]
    safe_baseline_margins = (
        [float(clean_srg_margins[ordered[index].row_id]) for index in safe_positions]
        if clean_srg_margins is not None
        else _margins(
            srg_scorer.score_responses(safe_prompts, clean_safe_responses),
            safe_rows,
        )
    )
    safe_candidate_margins = _margins(
        srg_scorer.score_responses(safe_prompts, candidate_safe_responses),
        safe_rows,
    )
    safe_srg = relative_score(
        safe_baseline_margins,
        safe_candidate_margins,
        dict(srg_profile),
        groups=[
            (ordered[index].language, ordered[index].category_id)
            for index in safe_positions
        ],
    )
    srg_seconds = time.perf_counter() - srg_started
    safe_targets = [
        [
            int(value)
            for value in clean[ordered[index].row_id]["clean_response_token_ids"]
        ]
        for index in safe_positions
    ]
    conditional_nll_started = time.perf_counter()
    candidate_nll_values = model.get_conditional_nll(
        [prompts[index] for index in safe_positions],
        safe_targets,
    )
    conditional_nll_seconds = time.perf_counter() - conditional_nll_started
    if len(candidate_nll_values) != safe_rows:
        raise ValueError("SAFE conditional NLL coverage mismatch")
    clean_nll = {
        ordered[index].row_id: float(
            clean[ordered[index].row_id]["clean_conditional_nll"]
        )
        for index in safe_positions
    }
    candidate_nll = {
        ordered[index].row_id: float(value)
        for index, value in zip(safe_positions, candidate_nll_values, strict=True)
    }
    geometry_started = time.perf_counter()
    ppl = aggregate_safe_ppl(ordered, clean_nll, candidate_nll)
    geometry = evaluate_trial_geometry(
        ordered,
        clean_projection,
        candidate_projection,
        layer_reliability=layer_reliability,
    )
    metrics = compose_trial_metrics(
        srg_gain=float(srg["srg_gain"]),
        r_gain=float(srg["r_gain"]),
        unsafe_geometry_gain=float(geometry["unsafe_geometry_gain"]),
        safe_ppl_drift=float(ppl["safe_ppl_drift"]),
        safe_ppl_signed_change=float(ppl["safe_ppl_signed_change"]),
        safe_geometry_drift=float(geometry["safe_geometry_damage"]),
        language_instability=float(geometry["language_instability"]),
        category_instability=float(geometry["category_instability"]),
    )
    empty_responses = sum(
        not str(response).strip() or len(ids) == 0
        for response, ids in zip(responses, token_ids, strict=True)
    )
    clean_token_ids = [
        [int(value) for value in clean[row.row_id]["clean_response_token_ids"]]
        for row in ordered
    ]
    clean_at_cap = [len(ids) >= max_response_length for ids in clean_token_ids]
    candidate_at_cap = [len(ids) >= max_response_length for ids in token_ids]
    newly_truncated = sum(
        not clean_cap and candidate_cap
        for clean_cap, candidate_cap in zip(
            clean_at_cap, candidate_at_cap, strict=True
        )
    )
    hard_gates = {
        "empty_response_rate": empty_responses / len(ordered),
        "truncated_response_rate": newly_truncated / len(ordered),
        "clean_truncated_response_rate": sum(clean_at_cap) / len(ordered),
        "candidate_truncated_response_rate": sum(candidate_at_cap) / len(ordered),
        "safe_d_to_r_rate": float(safe_srg["raw_d_to_r_rate"]),
        "safe_d_to_r_weighted_rate": float(safe_srg["d_to_r_rate"]),
    }
    geometry_seconds = time.perf_counter() - geometry_started

    archive_started = time.perf_counter()
    private_lines = []
    for index, row in enumerate(ordered):
        record = {
            "trial_number": trial_number,
            "canonical_id": row.canonical_id,
            "row_id": row.row_id,
            "language": row.language,
            "direction_class": row.direction,
            "category_id": row.category_id,
            "prompt": row.prompt,
            "response": responses[index],
            "response_token_ids": [int(value) for value in token_ids[index]],
        }
        private_lines.append(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    payload = "".join(private_lines).encode("utf-8")
    private_path = Path(private_records_path).resolve()
    _atomic_write(private_path, payload)
    archive_write_seconds = time.perf_counter() - archive_started
    total_seconds = time.perf_counter() - total_started
    return TrialMeasurement(
        trial_number=trial_number,
        rows=len(ordered),
        safe_rows=safe_rows,
        unsafe_rows=unsafe_rows,
        metrics=metrics,
        diagnostics={
            "srg": {
                key: value for key, value in srg.items() if key != "standardized_gain"
            },
            "srg_groups": srg_group_summary,
            "geometry": geometry,
            "ppl": ppl,
            "hard_gates": hard_gates,
            "timings": {
                "generation_seconds": generation_seconds,
                "srg_seconds": srg_seconds,
                "conditional_nll_seconds": conditional_nll_seconds,
                "geometry_seconds": geometry_seconds,
                "archive_write_seconds": archive_write_seconds,
                "total_seconds": total_seconds,
            },
        },
        private_records_sha256=_sha256_bytes(payload),
    )
