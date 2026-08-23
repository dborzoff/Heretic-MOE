# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resumable, text-private aligned corpus translation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TranslationInput:
    canonical_id: str
    direction_class: str
    prompt: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.direction_class, self.canonical_id)


class TranslationModel(Protocol):
    def translate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
    ) -> list[str]: ...

    def release_failed_batch(self) -> None: ...


def render_translation_prompt(row: TranslationInput, target_language: str) -> str:
    return (
        f"Translate the user text faithfully into {target_language}. Preserve meaning, "
        "tone, names, formatting, and level of detail. Do not answer or discuss the text. "
        "Return only the translation, without labels or quotation marks.\n\n"
        f"USER TEXT:\n{row.prompt}"
    )


def _load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    result: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            key = (str(raw["direction_class"]), str(raw["canonical_id"]))
            if key in result:
                raise ValueError(f"duplicate translation at line {line_number}")
            if not str(raw.get("prompt", "")).strip():
                raise ValueError(f"empty translation at line {line_number}")
            result.add(key)
    return result


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def translate_rows_with_model(
    model: TranslationModel,
    *,
    rows: Sequence[TranslationInput],
    output_path: str | Path,
    target_language: str,
    batch_size: int,
    max_new_tokens: int,
    event_sink: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, int]:
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch size and max new tokens must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = _load_completed(output_path)
    pending = [row for row in rows if row.key not in completed_keys]
    skipped = len(rows) - len(pending)
    completed = 0
    active_batch_size = batch_size
    position = 0

    while position < len(pending):
        size = min(active_batch_size, len(pending) - position)
        batch_rows = pending[position : position + size]
        prompts = [
            render_translation_prompt(row, target_language) for row in batch_rows
        ]
        try:
            outputs = model.translate_batch(prompts, max_new_tokens=max_new_tokens)
        except BaseException as error:
            if active_batch_size == 1 or not _is_oom(error):
                raise
            active_batch_size = max(1, active_batch_size // 2)
            model.release_failed_batch()
            if event_sink is not None:
                event_sink(
                    {
                        "event": "translation_batch_backoff",
                        "batch_size": active_batch_size,
                    }
                )
            continue
        if len(outputs) != size:
            raise RuntimeError("translation model returned the wrong batch size")
        encoded: list[str] = []
        for row, output in zip(batch_rows, outputs, strict=True):
            translated = output.strip()
            if not translated:
                raise ValueError(f"empty translation for {row.canonical_id}")
            encoded.append(
                json.dumps(
                    {
                        "canonical_id": row.canonical_id,
                        "direction_class": row.direction_class,
                        "prompt": translated,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.writelines(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        position += size
        completed += size
        if event_sink is not None:
            event_sink(
                {
                    "event": "translation_progress",
                    "completed": completed + skipped,
                    "total": len(rows),
                    "batch_size": active_batch_size,
                }
            )
    return {
        "completed": completed,
        "skipped": skipped,
        "batch_size": active_batch_size,
    }
