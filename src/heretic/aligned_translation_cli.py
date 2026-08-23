# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two-GPU controller for a resumable aligned Japanese corpus translation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from .aligned_translation import TranslationInput, translate_rows_with_model
from .main_classification_dataset import (
    apply_aligned_translations,
    materialize_main_classification_dataset,
)
from .self_classification_data import load_classification_rows
from .utils import get_file_sha256

_SOURCE_TEMPLATES = (
    "direction_{language}_safe.jsonl",
    "direction_{language}_unsafe.jsonl",
    "search_unsafe_{language}.jsonl",
    "srg_calibration_{language}.jsonl",
)


def build_worker_jobs(
    *,
    dataset_manifest: str | Path,
    model: str | Path,
    output_root: str | Path,
    devices: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    output_root = Path(output_root)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be non-empty and unique")
    return [
        {
            "schema_version": 1,
            "dataset_manifest": str(Path(dataset_manifest).resolve()),
            "model": str(Path(model).resolve()),
            "output_path": str(
                (output_root / f"translations.worker-{index}.jsonl").resolve()
            ),
            "target_language": "Japanese",
            "shard_index": index,
            "shard_count": len(devices),
            "batch_size": int(batch_size),
            "max_new_tokens": int(max_new_tokens),
            "dtype": "bfloat16",
            "seed": 20260815,
        }
        for index, _ in enumerate(devices)
    ]


def merge_translation_sidecars(
    paths: Sequence[str | Path],
    rows: Sequence[TranslationInput],
    output_path: str | Path,
) -> dict[str, object]:
    values: dict[tuple[str, str], dict[str, object]] = {}
    for path_value in paths:
        with Path(path_value).open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                key = (str(raw["direction_class"]), str(raw["canonical_id"]))
                if key in values:
                    raise ValueError("duplicate translation across worker sidecars")
                if not str(raw.get("prompt", "")).strip():
                    raise ValueError("empty translation in worker sidecar")
                values[key] = raw
    expected = [row.key for row in rows]
    if set(values) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("worker translation coverage mismatch")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(values[key], ensure_ascii=False, sort_keys=True) + "\n"
            for key in expected
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {"rows": len(expected), "status": "PASS"}


class _HereticTranslationModel:
    def __init__(self, job: dict[str, object]) -> None:
        from .config import QuantizationMethod, Settings
        from .model import Model

        settings = Settings(
            model=str(job["model"]),
            dtypes=[str(job.get("dtype", "bfloat16"))],
            quantization=QuantizationMethod.NONE,
            device_map="auto",
            batch_size=int(job["batch_size"]),
            max_batch_size=int(job["batch_size"]),
            max_response_length=int(job["max_new_tokens"]),
            chat_template_enable_thinking=False,
            offload_outputs_to_cpu=True,
            seed=int(job.get("seed", 20260815)),
            system_prompt="",
        )
        self.model = Model(settings)

    def translate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        import torch

        from .utils import Prompt

        values = [Prompt(system="", user=value) for value in prompts]
        inputs, outputs = self.model.generate(values, max_new_tokens=max_new_tokens)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated = sequences[:, inputs["input_ids"].shape[1] :]
        responses = self.model.tokenizer.batch_decode(generated, skip_special_tokens=True)
        del inputs, outputs, sequences, generated
        with suppress(RuntimeError):
            torch.cuda.synchronize()
        return responses

    def release_failed_batch(self) -> None:
        if hasattr(self.model, "_release_failed_cuda_batch"):
            self.model._release_failed_cuda_batch()


def _worker(job_path: Path, device: str) -> dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1:
        raise ValueError("unsupported translation job schema")
    loaded = load_classification_rows(job["dataset_manifest"], ("en",))
    rows = [
        TranslationInput(row.canonical_id, row.direction_class, row.prompt)
        for row in loaded
    ]
    shard_index = int(job["shard_index"])
    shard_count = int(job["shard_count"])
    rows = rows[shard_index::shard_count]
    print(
        json.dumps(
            {
                "event": "translation_worker_phase",
                "phase": "model_load",
                "shard": shard_index,
                "rows": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model = _HereticTranslationModel(job)
    print(
        json.dumps(
            {
                "event": "translation_worker_phase",
                "phase": "model_ready",
                "shard": shard_index,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = translate_rows_with_model(
        model,
        rows=rows,
        output_path=job["output_path"],
        target_language=str(job["target_language"]),
        batch_size=int(job["batch_size"]),
        max_new_tokens=int(job["max_new_tokens"]),
        event_sink=lambda event: print(
            json.dumps({**event, "shard": shard_index}, sort_keys=True),
            flush=True,
        ),
    )
    return {"event": "translation_worker_complete", "shard": shard_index, **summary}


def _controller(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    work_root = output_root.parent / f".{output_root.name}.work"
    source_stage = work_root / "source"
    initial_dataset = work_root / "initial_dataset"
    jobs_root = work_root / "jobs"
    work_root.mkdir(parents=True, exist_ok=True)
    source_stage.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)

    for language in ("en", "ru", "zh"):
        for template in _SOURCE_TEMPLATES:
            source = source_root / template.format(language=language)
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, source_stage / source.name)
    materialize_main_classification_dataset(
        source_stage,
        initial_dataset,
        languages=("en", "ru", "zh"),
    )
    loaded = load_classification_rows(initial_dataset / "manifest.json", ("en",))
    rows = [
        TranslationInput(row.canonical_id, row.direction_class, row.prompt)
        for row in loaded
    ]
    jobs = build_worker_jobs(
        dataset_manifest=initial_dataset / "manifest.json",
        model=args.model,
        output_root=work_root,
        devices=args.devices,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    job_paths: list[Path] = []
    for index, job in enumerate(jobs):
        path = jobs_root / f"worker-{index}.json"
        path.write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        job_paths.append(path)
    if args.dry_run:
        return {"status": "DRY_RUN", "rows": len(rows), "workers": len(jobs)}

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "heretic.aligned_translation_cli",
                "worker",
                "--job",
                str(job_path),
                "--device",
                device,
            ],
            cwd=Path(__file__).resolve().parents[2],
        )
        for job_path, device in zip(job_paths, args.devices, strict=True)
    ]
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"translation worker failure: {failures}")

    merged = work_root / "translations.ja.jsonl"
    merge_translation_sidecars(
        tuple(job["output_path"] for job in jobs),
        rows,
        merged,
    )
    apply_aligned_translations(
        source_stage,
        merged,
        source_stage,
        target_language="ja",
    )
    manifest = materialize_main_classification_dataset(
        source_stage,
        output_root,
        languages=("en", "ru", "zh", "ja"),
    )
    provenance = {
        "schema_version": 1,
        "status": "PASS",
        "model": str(args.model.resolve()),
        "translation_sidecar_sha256": get_file_sha256(merged),
        "rows": len(rows),
        "workers": len(jobs),
        "dataset_manifest_sha256": get_file_sha256(output_root / "manifest.json"),
    }
    (output_root / "translation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {**provenance, "dataset_rows": manifest["rows"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE translate-aligned")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source-root", required=True, type=Path)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--devices", default="0,1")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--dry-run", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--job", required=True, type=Path)
    worker.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "worker":
        result = _worker(args.job, str(args.device))
    else:
        args.devices = tuple(
            value.strip() for value in str(args.devices).split(",") if value.strip()
        )
        result = _controller(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
