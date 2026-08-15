# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resident inference worker for text-private self-classification."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from .self_classification import (
    ClassificationInput,
    ClassificationResult,
    PromptVariant,
    RenderedClassifierPrompt,
    classify_output_shape,
    parse_classification_output,
    render_classifier_prompt,
)
from .self_classification_data import (
    append_result_atomic,
    load_classification_rows,
    load_completed_keys,
)


class ClassificationModel(Protocol):
    def prepare(self, prompts: Sequence[RenderedClassifierPrompt]) -> None: ...

    def classify_batch(
        self,
        prompts: Sequence[RenderedClassifierPrompt],
        *,
        max_new_tokens: int,
    ) -> tuple[list[str], list[int]]: ...

    def release_failed_batch(self) -> None: ...


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def classify_rows_with_model(
    model: ClassificationModel,
    *,
    model_id: str,
    rows: Sequence[ClassificationInput],
    variants: Sequence[PromptVariant],
    output_path: str | Path,
    batch_size: int,
    system_mode: str = "localized",
    max_new_tokens: int | None = None,
    event_sink: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("classification batch size must be positive")
    if not model_id or not variants or len(set(variants)) != len(variants):
        raise ValueError("model ID and unique variants are required")
    output_path = Path(output_path)
    completed_keys = load_completed_keys(output_path)
    skipped = sum(
        (model_id, row.row_id, variant.value) in completed_keys
        for variant in variants
        for row in rows
    )
    completed = 0
    invalid = 0
    active_batch_size = int(batch_size)

    def emit(event: dict[str, object]) -> None:
        if event_sink is not None:
            event_sink(event)

    total = len(rows) * len(variants)
    for variant in variants:
        pending = [
            row
            for row in rows
            if (model_id, row.row_id, variant.value) not in completed_keys
        ]
        rendered = [
            render_classifier_prompt(
                row,
                variant,
                system_mode=system_mode,
                max_new_tokens=max_new_tokens,
            )
            for row in pending
        ]
        if rendered:
            model.prepare(rendered)
        position = 0
        while position < len(pending):
            size = min(active_batch_size, len(pending) - position)
            batch_rows = pending[position : position + size]
            batch_prompts = rendered[position : position + size]
            try:
                outputs, token_counts = model.classify_batch(
                    batch_prompts,
                    max_new_tokens=batch_prompts[0].max_new_tokens,
                )
            except BaseException as error:
                if active_batch_size == 1 or not _is_oom(error):
                    raise
                previous = active_batch_size
                active_batch_size = max(1, active_batch_size // 2)
                model.release_failed_batch()
                emit(
                    {
                        "event": "batch_backoff",
                        "previous_batch_size": previous,
                        "batch_size": active_batch_size,
                    }
                )
                continue
            if len(outputs) != size or len(token_counts) != size:
                raise RuntimeError("classification model returned the wrong batch size")
            for row, prompt, output, output_tokens in zip(
                batch_rows, batch_prompts, outputs, token_counts, strict=True
            ):
                classification = parse_classification_output(
                    output,
                    row.row_id,
                    variant,
                    row.language,
                )
                valid = classification is not None
                invalid += int(not valid)
                result = ClassificationResult(
                    model_id=model_id,
                    canonical_id=row.canonical_id,
                    row_id=row.row_id,
                    language=row.language,
                    category_ids=row.category_ids,
                    direction_class=row.direction_class,
                    variant=variant,
                    classification=classification,
                    valid=valid,
                    output_tokens=int(output_tokens),
                    output_shape=classify_output_shape(
                        output,
                        prompt.expected_outputs.values(),
                    ),
                )
                append_result_atomic(output_path, result)
                completed_keys.add(result.key)
                completed += 1
            position += size
            emit(
                {
                    "event": "classification_progress",
                    "model_id": model_id,
                    "variant": variant.value,
                    "completed": completed + skipped,
                    "total": total,
                    "batch_size": active_batch_size,
                }
            )
    return {
        "completed": completed,
        "skipped": skipped,
        "invalid": invalid,
        "batch_size": active_batch_size,
    }


class HereticClassificationModel:
    def __init__(self, model: Any) -> None:
        self.model = model

    def _as_prompts(self, rows: Sequence[RenderedClassifierPrompt]):
        from .utils import Prompt

        return [Prompt(system=row.system, user=row.user) for row in rows]

    def prepare(self, prompts: Sequence[RenderedClassifierPrompt]) -> None:
        values = self._as_prompts(prompts)
        if hasattr(self.model, "prepare_prompt_cache"):
            self.model.prepare_prompt_cache(values)
            if hasattr(self.model, "pin_prompt_cache"):
                self.model.pin_prompt_cache()

    def classify_batch(
        self,
        prompts: Sequence[RenderedClassifierPrompt],
        *,
        max_new_tokens: int,
    ) -> tuple[list[str], list[int]]:
        import torch

        values = self._as_prompts(prompts)
        inputs, outputs = self.model.generate(values, max_new_tokens=max_new_tokens)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        responses = self.model.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        eos = self.model.tokenizer.eos_token_id
        eos_ids = (
            {int(value) for value in eos}
            if isinstance(eos, (tuple, list, set))
            else ({int(eos)} if eos is not None else set())
        )
        pad = self.model.tokenizer.pad_token_id
        counts: list[int] = []
        for raw in generated.detach().cpu().tolist():
            values = [int(value) for value in raw]
            stop = next(
                (index + 1 for index, value in enumerate(values) if value in eos_ids),
                None,
            )
            if stop is not None:
                values = values[:stop]
            elif pad is not None:
                while values and values[-1] == int(pad):
                    values.pop()
            counts.append(len(values))
        del inputs, outputs, sequences, generated
        with suppress(RuntimeError):
            torch.cuda.synchronize()
        return responses, counts

    def release_failed_batch(self) -> None:
        if hasattr(self.model, "_release_failed_cuda_batch"):
            self.model._release_failed_cuda_batch()


def _load_model(job: dict[str, object]) -> HereticClassificationModel:
    import torch

    from .config import QuantizationMethod, Settings
    from .model import Model

    cpu_threads = int(job.get("cpu_threads", 6))
    torch.set_num_threads(cpu_threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    settings = Settings(
        model=str(job["model"]),
        dtypes=[str(job.get("dtype", "bfloat16"))],
        quantization=QuantizationMethod.NONE,
        device_map="auto",
        batch_size=int(job["batch_size"]),
        max_batch_size=int(job.get("max_batch_size", 256)),
        max_response_length=32,
        chat_template_enable_thinking=False,
        offload_outputs_to_cpu=True,
        seed=int(job.get("seed", 20260815)),
        system_prompt="",
    )
    return HereticClassificationModel(Model(settings))


def run_worker_job(
    job_path: str | Path,
    *,
    device: str,
    worker_id: str,
    model_factory: Callable[[dict[str, object]], ClassificationModel] | None = None,
) -> dict[str, object]:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    if job.get("schema_version") != 1:
        raise ValueError("unsupported self-classification worker job schema")
    rows = load_classification_rows(job["dataset_manifest"], job["languages"])
    shard_index = int(job.get("shard_index", 0))
    shard_count = int(job.get("shard_count", 1))
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid worker shard")
    rows = rows[shard_index::shard_count]
    variants = tuple(PromptVariant(str(value)) for value in job["variants"])
    print(
        json.dumps(
            {"event": "worker_phase", "phase": "model_load", "worker_id": worker_id},
            sort_keys=True,
        ),
        flush=True,
    )
    model = (model_factory or _load_model)(job)
    print(
        json.dumps(
            {"event": "worker_phase", "phase": "model_ready", "worker_id": worker_id},
            sort_keys=True,
        ),
        flush=True,
    )
    summary = classify_rows_with_model(
        model,
        model_id=str(job["model_id"]),
        rows=rows,
        variants=variants,
        output_path=job["output_path"],
        batch_size=int(job["batch_size"]),
        system_mode=str(job.get("system_mode", "localized")),
        max_new_tokens=(
            int(job["max_new_tokens"])
            if job.get("max_new_tokens") is not None
            else None
        ),
        event_sink=lambda event: print(
            json.dumps({**event, "worker_id": worker_id}, sort_keys=True),
            flush=True,
        ),
    )
    return {"worker_id": worker_id, "device": device, **summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE self-classify worker")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--worker-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    result = run_worker_job(
        args.job,
        device=str(args.device),
        worker_id=str(args.worker_id),
    )
    print(json.dumps({"event": "worker_complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
