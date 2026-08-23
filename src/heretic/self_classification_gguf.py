from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from .self_classification import (
    ClassificationInput,
    ClassificationResult,
    PromptVariant,
    RenderedClassifierPrompt,
    parse_classification_output,
    render_classifier_prompt,
)
from .self_classification_data import (
    append_results_atomic,
    load_classification_rows,
    load_completed_keys,
    verify_result_coverage,
)
from .self_classification_report import write_consensus_reports, write_model_reports
from .utils import get_file_sha256

PostJson = Callable[[str, dict[str, object]], dict[str, object]]
ProgressCallback = Callable[[int, int], None]
CONSENSUS_VARIANTS = (
    PromptVariant.CODE_PERMUTED,
    PromptVariant.CODE_SHIFT_1,
    PromptVariant.CODE_SHIFT_2,
    PromptVariant.CODE_SHIFT_3,
    PromptVariant.WORD_ORDER_0,
    PromptVariant.WORD_ORDER_1,
    PromptVariant.WORD_ORDER_2,
    PromptVariant.WORD_ORDER_3,
)


class CompletionClient(Protocol):
    def generate(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
    ) -> tuple[str, int]: ...


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, object],
    *,
    timeout: float,
    retries: int = 2,
) -> dict[str, object]:
    if retries < 0:
        raise ValueError("HTTP retries must be nonnegative")
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except TimeoutError:
            if attempt == retries:
                raise
            time.sleep(0.25 * (2**attempt))
    if not isinstance(result, dict):
        raise TypeError("llama-server returned a non-object JSON response")
    return result


class LlamaCompletionClient:
    def __init__(self, *, post_json: PostJson) -> None:
        self._post_json = post_json

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
    ) -> tuple[str, int]:
        completion = self._post_json(
            "/v1/chat/completions",
            {
                "model": "local",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
        choices = completion.get("choices")
        usage = completion.get("usage")
        if not isinstance(choices, list) or not choices or not isinstance(usage, dict):
            raise TypeError("llama-server returned an invalid chat completion")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise TypeError("llama-server returned an invalid chat choice")
        return str(choice["message"].get("content", "")), int(
            usage.get("completion_tokens", 0)
        )


def build_server_process(
    *,
    server: str | Path,
    model: str | Path,
    device: str,
    port: int,
    parallel: int,
    context_per_slot: int,
    gpu_layers: int,
) -> tuple[list[str], dict[str, str]]:
    command = [
        str(server),
        "-m",
        str(model),
        "--port",
        str(port),
        "-np",
        str(parallel),
        "-c",
        str(context_per_slot * parallel),
        "-ngl",
        str(gpu_layers),
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--no-warmup",
        "--log-disable",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    return command, environment


def make_result_record(
    *,
    model_id: str,
    row: ClassificationInput,
    variant: PromptVariant,
    generated_text: str,
    output_tokens: int,
) -> dict[str, object]:
    return _make_result(
        model_id=model_id,
        row=row,
        variant=variant,
        generated_text=generated_text,
        output_tokens=output_tokens,
    ).to_public_dict()


def _make_result(
    *,
    model_id: str,
    row: ClassificationInput,
    variant: PromptVariant,
    generated_text: str,
    output_tokens: int,
) -> ClassificationResult:
    classification = parse_classification_output(
        generated_text,
        row.row_id,
        variant,
        row.language,
    )
    return ClassificationResult(
        model_id=model_id,
        canonical_id=row.canonical_id,
        row_id=row.row_id,
        language=row.language,
        category_ids=row.category_ids,
        direction_class=row.direction_class,
        variant=variant,
        classification=classification,
        valid=classification is not None,
        output_tokens=output_tokens,
    )


def classify_rows_with_gguf(
    *,
    client: CompletionClient,
    model_id: str,
    rows: Sequence[ClassificationInput],
    variant: PromptVariant | None = None,
    variants: Sequence[PromptVariant] | None = None,
    system_mode: str = "localized",
    max_new_tokens: int | None = None,
    output_path: str | Path,
    parallel: int,
    checkpoint_rows: int = 256,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    if parallel < 1:
        raise ValueError("parallel must be positive")
    if checkpoint_rows < 1:
        raise ValueError("checkpoint rows must be positive")
    if variant is not None and variants is not None:
        raise ValueError("pass variant or variants, not both")
    selected = tuple(variants or ((variant,) if variant is not None else ()))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("at least one unique GGUF prompt variant is required")
    completed_keys = load_completed_keys(output_path)
    total = len(rows) * len(selected)
    skipped = sum(
        (model_id, row.row_id, value.value) in completed_keys
        for value in selected
        for row in rows
    )
    generated = 0
    completed = skipped
    for current_variant in selected:
        pending = [
            row
            for row in rows
            if (model_id, row.row_id, current_variant.value) not in completed_keys
        ]
        rendered = [
            (
                row,
                render_classifier_prompt(
                    row,
                    current_variant,
                    system_mode=system_mode,
                    max_new_tokens=max_new_tokens,
                ),
            )
            for row in pending
        ]

        def classify_one(
            item: tuple[ClassificationInput, RenderedClassifierPrompt],
            variant_value: PromptVariant = current_variant,
        ) -> ClassificationResult:
            row, rendered_prompt = item
            generated_text, output_tokens = client.generate(
                system=rendered_prompt.system,
                user=rendered_prompt.user,
                max_new_tokens=rendered_prompt.max_new_tokens,
            )
            return _make_result(
                model_id=model_id,
                row=row,
                variant=variant_value,
                generated_text=generated_text,
                output_tokens=output_tokens,
            )

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            checkpoint: list[ClassificationResult] = []
            for result in executor.map(classify_one, rendered):
                checkpoint.append(result)
                if len(checkpoint) >= checkpoint_rows:
                    append_results_atomic(output_path, checkpoint)
                    completed += len(checkpoint)
                    generated += len(checkpoint)
                    checkpoint.clear()
                    if progress is not None:
                        progress(completed, total)
            if checkpoint:
                append_results_atomic(output_path, checkpoint)
                completed += len(checkpoint)
                generated += len(checkpoint)
                checkpoint.clear()
                if progress is not None:
                    progress(completed, total)
    return {"completed": completed, "generated": generated, "total": total}


def finalize_gguf_results(
    *,
    output_dir: str | Path,
    model: str | Path,
    dataset_manifest: str | Path,
    model_id: str,
    rows: Sequence[ClassificationInput],
    variant: PromptVariant | None = None,
    variants: Sequence[PromptVariant] | None = None,
    system_mode: str = "localized",
) -> dict[str, object]:
    if variant is not None and variants is not None:
        raise ValueError("pass variant or variants, not both")
    selected = tuple(variants or ((variant,) if variant is not None else ()))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("at least one unique GGUF prompt variant is required")
    output_dir = Path(output_dir)
    result_path = output_dir / "rows.jsonl"
    expected = {
        (model_id, row.row_id, value.value) for row in rows for value in selected
    }
    verify_result_coverage(result_path, expected)
    public_rows: list[dict[str, object]] = []
    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("GGUF result row is not an object")
            prohibited = {"prompt", "response", "raw_output", "text"} & value.keys()
            if prohibited:
                raise ValueError(f"GGUF result contains prohibited fields: {prohibited}")
            public_rows.append(value)
    write_model_reports(output_dir, public_rows)
    if len(selected) == 8:
        write_consensus_reports(output_dir, public_rows)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "model_id": model_id,
        "model_sha256": get_file_sha256(Path(model)),
        "dataset_sha256": get_file_sha256(Path(dataset_manifest)),
        "result_sha256": get_file_sha256(result_path),
        "variants": [value.value for value in selected],
        "system_mode": system_mode,
        "rows": len(public_rows),
        "valid": sum(bool(row["valid"]) for row in public_rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Text-private GGUF self-classification through llama-server"
    )
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--languages", default="en,ru,zh,ja,fr")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[value.value for value in PromptVariant],
    )
    parser.add_argument("--consensus", action="store_true")
    parser.add_argument(
        "--system-mode", choices=("localized", "english"), default="localized"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    languages = tuple(
        value.strip().lower() for value in str(args.languages).split(",") if value.strip()
    )
    if not languages:
        raise ValueError("at least one language is required")
    rows = load_classification_rows(args.dataset_manifest, languages)
    model = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.consensus and args.variant:
        raise ValueError("--consensus cannot be combined with --variant")
    variants = (
        CONSENSUS_VARIANTS
        if args.consensus
        else tuple(
            PromptVariant(str(value))
            for value in (args.variant or [PromptVariant.CODE_PERMUTED.value])
        )
    )
    system_mode = "english" if args.consensus else str(args.system_mode)
    client = LlamaCompletionClient(
        post_json=partial(
            post_json,
            str(args.base_url),
            timeout=float(args.timeout),
        )
    )

    def progress(completed: int, total: int) -> None:
        if completed % 32 == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "event": "gguf_classification_progress",
                        "model_id": model.name,
                        "total": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    run_summary = classify_rows_with_gguf(
        client=client,
        model_id=model.name,
        rows=rows,
        variants=variants,
        system_mode=system_mode,
        max_new_tokens=8 if args.consensus else None,
        output_path=output_dir / "rows.jsonl",
        parallel=int(args.parallel),
        progress=progress,
    )
    manifest = finalize_gguf_results(
        output_dir=output_dir,
        model=model,
        dataset_manifest=args.dataset_manifest,
        model_id=model.name,
        rows=rows,
        variants=variants,
        system_mode=system_mode,
    )
    manifest["generated"] = run_summary["generated"]
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


if __name__ == "__main__":
    main()
