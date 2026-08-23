from __future__ import annotations

import json
from pathlib import Path

from heretic.aligned_translation import (
    TranslationInput,
    translate_rows_with_model,
)


class FakeTranslationModel:
    def translate_batch(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
        assert max_new_tokens == 64
        return [f"translated-{index}" for index, _ in enumerate(prompts)]

    def release_failed_batch(self) -> None:
        raise AssertionError("release is not expected")


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_translation_runner_writes_private_sidecar_and_resumes(tmp_path: Path) -> None:
    rows = [
        TranslationInput("S0001", "safe", "source-safe"),
        TranslationInput("U0001", "unsafe", "source-unsafe"),
    ]
    path = tmp_path / "translations.jsonl"
    events: list[dict[str, object]] = []

    first = translate_rows_with_model(
        FakeTranslationModel(),
        rows=rows,
        output_path=path,
        target_language="ja",
        batch_size=2,
        max_new_tokens=64,
        event_sink=events.append,
    )
    second = translate_rows_with_model(
        FakeTranslationModel(),
        rows=rows,
        output_path=path,
        target_language="ja",
        batch_size=2,
        max_new_tokens=64,
    )

    assert first == {"completed": 2, "skipped": 0, "batch_size": 2}
    assert second == {"completed": 0, "skipped": 2, "batch_size": 2}
    assert _read(path) == [
        {"canonical_id": "S0001", "direction_class": "safe", "prompt": "translated-0"},
        {"canonical_id": "U0001", "direction_class": "unsafe", "prompt": "translated-1"},
    ]
    assert events[-1] == {
        "event": "translation_progress",
        "completed": 2,
        "total": 2,
        "batch_size": 2,
    }


def test_translation_runner_rejects_empty_model_output(tmp_path: Path) -> None:
    class EmptyModel:
        def translate_batch(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
            return [""] * len(prompts)

        def release_failed_batch(self) -> None:
            return None

    try:
        translate_rows_with_model(
            EmptyModel(),
            rows=[TranslationInput("S0001", "safe", "source")],
            output_path=tmp_path / "translations.jsonl",
            target_language="ja",
            batch_size=1,
            max_new_tokens=64,
        )
    except ValueError as error:
        assert "empty translation" in str(error)
    else:
        raise AssertionError("empty translations must fail")
