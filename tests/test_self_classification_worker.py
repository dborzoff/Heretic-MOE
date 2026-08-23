from __future__ import annotations

import json
from pathlib import Path

import pytest

from heretic.self_classification import (
    BehaviorClass,
    ClassificationInput,
    ClassificationResult,
    PromptVariant,
    render_classifier_prompt,
)
from heretic.self_classification_data import append_result_atomic
from heretic.self_classification_worker import classify_rows_with_model


def rows(count: int) -> list[ClassificationInput]:
    return [
        ClassificationInput(
            canonical_id=f"P{index:04d}",
            row_id=f"EN-P{index:04d}",
            language="en",
            category_ids=("C1",),
            direction_class="safe",
            prompt=f"PRIVATE_SENTINEL_{index}",
        )
        for index in range(count)
    ]


class FakeClassificationModel:
    def __init__(self, *, maximum_batch: int = 999) -> None:
        self.maximum_batch = maximum_batch
        self.batch_sizes: list[int] = []
        self.cached_rows = 0

    def prepare(self, prompts) -> None:
        self.cached_rows += len(prompts)

    def classify_batch(self, prompts, *, max_new_tokens: int):
        self.batch_sizes.append(len(prompts))
        if len(prompts) > self.maximum_batch:
            raise RuntimeError("CUDA out of memory")
        assert max_new_tokens == 4
        return ["1"] * len(prompts), [1] * len(prompts)

    def release_failed_batch(self) -> None:
        return None


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_worker_classifies_rows_without_persisting_text(tmp_path: Path) -> None:
    model = FakeClassificationModel()
    output = tmp_path / "results.jsonl"
    events: list[dict[str, object]] = []

    summary = classify_rows_with_model(
        model,
        model_id="model-a",
        rows=rows(3),
        variants=(PromptVariant.NUMBER,),
        output_path=output,
        batch_size=2,
        event_sink=events.append,
    )

    saved = read_rows(output)
    assert summary == {"completed": 3, "skipped": 0, "invalid": 0, "batch_size": 2}
    assert [row["classification"] for row in saved] == ["DIRECT"] * 3
    assert model.batch_sizes == [2, 1]
    serialized = output.read_text(encoding="utf-8") + json.dumps(events)
    assert "PRIVATE_SENTINEL" not in serialized
    assert "raw_output" not in serialized


def test_worker_skips_completed_keys_on_resume(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    append_result_atomic(
        output,
        ClassificationResult(
            model_id="model-a",
            canonical_id="P0000",
            row_id="EN-P0000",
            language="en",
            category_ids=("C1",),
            direction_class="safe",
            variant=PromptVariant.NUMBER,
            classification=BehaviorClass.DIRECT,
            valid=True,
            output_tokens=1,
        ),
    )
    model = FakeClassificationModel()

    summary = classify_rows_with_model(
        model,
        model_id="model-a",
        rows=rows(3),
        variants=(PromptVariant.NUMBER,),
        output_path=output,
        batch_size=4,
    )

    assert summary["completed"] == 2
    assert summary["skipped"] == 1
    assert model.batch_sizes == [2]
    assert len(read_rows(output)) == 3


def test_worker_halves_batch_after_oom_without_losing_rows(tmp_path: Path) -> None:
    model = FakeClassificationModel(maximum_batch=2)
    output = tmp_path / "results.jsonl"

    summary = classify_rows_with_model(
        model,
        model_id="model-a",
        rows=rows(5),
        variants=(PromptVariant.NUMBER,),
        output_path=output,
        batch_size=4,
    )

    assert model.batch_sizes == [4, 2, 2, 1]
    assert summary["batch_size"] == 2
    assert len(read_rows(output)) == 5


def test_worker_marks_unparseable_output_invalid(tmp_path: Path) -> None:
    class InvalidModel(FakeClassificationModel):
        def classify_batch(self, prompts, *, max_new_tokens: int):
            return ["long explanation"] * len(prompts), [2] * len(prompts)

    output = tmp_path / "results.jsonl"

    summary = classify_rows_with_model(
        InvalidModel(),
        model_id="model-a",
        rows=rows(1),
        variants=(PromptVariant.NUMBER,),
        output_path=output,
        batch_size=1,
    )

    assert summary["invalid"] == 1
    saved = read_rows(output)[0]
    assert saved["classification"] is None
    assert saved["output_shape"] == "unknown"


def test_worker_honors_shared_system_and_generation_limit(tmp_path: Path) -> None:
    class CapturingModel(FakeClassificationModel):
        def __init__(self) -> None:
            super().__init__()
            self.systems: list[str] = []
            self.token_limits: list[int] = []

        def prepare(self, prompts) -> None:
            self.systems.extend(prompt.system for prompt in prompts)

        def classify_batch(self, prompts, *, max_new_tokens: int):
            self.token_limits.append(max_new_tokens)
            return ["1"] * len(prompts), [2] * len(prompts)

    model = CapturingModel()
    output = tmp_path / "results.jsonl"
    ru_row = ClassificationInput(
        canonical_id="P0001",
        row_id="RU-P0001",
        language="ru",
        category_ids=("C1",),
        direction_class="safe",
        prompt="PRIVATE_SENTINEL_RU",
    )

    classify_rows_with_model(
        model,
        model_id="model-a",
        rows=[ru_row],
        variants=(PromptVariant.NUMBER,),
        output_path=output,
        batch_size=1,
        system_mode="english",
        max_new_tokens=16,
    )

    english_system = render_classifier_prompt(
        rows(1)[0], PromptVariant.NUMBER, system_mode="english"
    ).system
    assert model.systems == [english_system]
    assert model.token_limits == [16]


def test_worker_batches_similar_prompt_lengths_together(tmp_path: Path) -> None:
    class LengthCapturingModel(FakeClassificationModel):
        def __init__(self) -> None:
            super().__init__()
            self.length_batches: list[list[int]] = []

        def classify_batch(self, prompts, *, max_new_tokens: int):
            self.length_batches.append([len(prompt.user) for prompt in prompts])
            return ["1"] * len(prompts), [1] * len(prompts)

    unsorted_rows = [
        ClassificationInput(
            canonical_id=f"P{index}",
            row_id=f"EN-P{index}",
            language="en",
            category_ids=("C1",),
            direction_class="safe",
            prompt="x" * length,
        )
        for index, length in enumerate((1, 100, 2, 99))
    ]
    model = LengthCapturingModel()

    classify_rows_with_model(
        model,
        model_id="model-a",
        rows=unsorted_rows,
        variants=(PromptVariant.NUMBER,),
        output_path=tmp_path / "rows.jsonl",
        batch_size=2,
    )

    assert len(model.length_batches) == 2
    assert all(max(batch) - min(batch) <= 2 for batch in model.length_batches)


def test_worker_checkpoints_once_per_completed_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from heretic.self_classification_data import append_results_atomic as persist

    batch_sizes: list[int] = []

    def checkpoint(path, results) -> None:
        batch_sizes.append(len(results))
        persist(path, results)

    monkeypatch.setattr(
        "heretic.self_classification_worker.append_results_atomic",
        checkpoint,
    )

    classify_rows_with_model(
        FakeClassificationModel(),
        model_id="model-a",
        rows=rows(5),
        variants=(PromptVariant.NUMBER,),
        output_path=tmp_path / "rows.jsonl",
        batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]


def test_worker_caps_batch_by_total_utf8_input_size(tmp_path: Path) -> None:
    """Catches packing long multilingual inputs into one memory-heavy batch."""
    model = FakeClassificationModel()

    classify_rows_with_model(
        model,
        model_id="model-a",
        rows=rows(3),
        variants=(PromptVariant.NUMBER,),
        output_path=tmp_path / "rows.jsonl",
        batch_size=3,
        max_batch_input_bytes=1,
    )

    assert model.batch_sizes == [1, 1, 1]


def test_worker_resets_oom_backoff_for_each_prompt_variant(tmp_path: Path) -> None:
    """Catches carrying a long-tail batch reduction into the next short prefix."""
    model = FakeClassificationModel(maximum_batch=2)

    classify_rows_with_model(
        model,
        model_id="model-a",
        rows=rows(3),
        variants=(PromptVariant.NUMBER, PromptVariant.CODE_PERMUTED),
        output_path=tmp_path / "rows.jsonl",
        batch_size=4,
    )

    assert model.batch_sizes == [3, 2, 1, 3, 2, 1]
