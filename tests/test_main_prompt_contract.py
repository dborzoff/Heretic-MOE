from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch


def test_main_imports_prompt_used_by_multilingual_runtime() -> None:
    from heretic import main
    from heretic.utils import Prompt

    assert main.Prompt is Prompt


def test_frozen_row_summary_uses_actual_runtime_bundle_sizes() -> None:
    from heretic.main import _format_multilingual_frozen_rows

    bundle = SimpleNamespace(
        direction_rows=tuple(range(40)),
        trial_rows=tuple(range(50)),
        final_rows=tuple(range(10)),
    )

    assert _format_multilingual_frozen_rows(bundle) == (
        "map 40, trial 50, built-in SRG profile, final holdout 10"
    )


def test_generation_runtime_fields_survive_checkpoint_continuation() -> None:
    from heretic.main import _ALWAYS_RUNTIME_FIELDS

    assert {
        "generation_backend",
        "generation_prompt_bucket_multiple",
        "generation_compile_mode",
    }.issubset(_ALWAYS_RUNTIME_FIELDS)


def test_config_backed_worker_value_is_not_rewritten_as_positional_model() -> None:
    from heretic.main import _normalize_legacy_positional_model_argv

    argv = [
        "hereticMOE",
        "--worker-queue-path",
        "queue.sqlite3",
        "--worker-heartbeat-interval-seconds",
        "15",
    ]

    assert _normalize_legacy_positional_model_argv(argv) == argv


def test_single_positional_model_remains_supported() -> None:
    from heretic.main import _normalize_legacy_positional_model_argv

    assert _normalize_legacy_positional_model_argv(
        ["hereticMOE", "org/model"]
    ) == ["hereticMOE", "--model", "org/model"]


def test_controller_created_empty_study_is_unfinished() -> None:
    from heretic.main import _study_is_finished

    study = SimpleNamespace(user_attrs={})

    assert _study_is_finished(study) is False


def test_queue_task_controls_progress_without_renumbering_optuna_trial() -> None:
    from heretic.main import _trial_progress_index

    retried_trial = SimpleNamespace(
        number=13,
        user_attrs={"queue_task_id": 11},
    )

    assert _trial_progress_index(retried_trial) == 12


def test_non_queue_progress_uses_optuna_trial_number() -> None:
    from heretic.main import _trial_progress_index

    assert _trial_progress_index(SimpleNamespace(number=13, user_attrs={})) == 14


def test_finished_study_flag_remains_respected() -> None:
    from heretic.main import _study_is_finished

    study = SimpleNamespace(user_attrs={"finished": True})

    assert _study_is_finished(study) is True


def test_controller_created_empty_study_is_not_a_resume_candidate() -> None:
    from heretic.main import _study_has_saved_settings

    assert _study_has_saved_settings(SimpleNamespace(user_attrs={})) is False
    assert (
        _study_has_saved_settings(SimpleNamespace(user_attrs={"settings": "{}"}))
        is True
    )


def test_supervised_worker_wires_model_batch_events() -> None:
    from heretic.main import _configure_supervised_model_events

    events: list[object] = []

    class FakeModel:
        def set_batch_event_sink(self, sink) -> None:
            sink({"event": "batch_probe", "batch_size": 32})

    _configure_supervised_model_events(
        FakeModel(),
        supervised=True,
        sink=events.append,
    )

    assert events == [{"event": "batch_probe", "batch_size": 32}]


def test_default_supervised_worker_event_sink_flushes_plain_json() -> None:
    from heretic.main import _configure_supervised_model_events

    class FakeModel:
        def set_batch_event_sink(self, sink) -> None:
            sink({"event": "batch_probe", "batch_size": 32})

    stream = io.StringIO()
    with patch("sys.stdout", stream):
        _configure_supervised_model_events(FakeModel(), supervised=True)

    assert stream.getvalue() == '{"batch_size": 32, "event": "batch_probe"}\n'


def test_cached_generation_batch_revalidation_emits_visible_events() -> None:
    from heretic.main import _emit_cached_batch_revalidation

    events: list[dict[str, object]] = []

    class FakeModel:
        def _emit_batch_event(self, event, mode, **values) -> None:
            events.append({"event": event, "mode": mode, **values})

    model = FakeModel()
    _emit_cached_batch_revalidation(
        model,
        phase="start",
        batch_size=168,
    )
    _emit_cached_batch_revalidation(
        model,
        phase="result",
        batch_size=168,
        status="PASS",
        free_bytes=8 * 1024**3,
        recovered_bytes=12 * 1024**3,
    )
    _emit_cached_batch_revalidation(
        model,
        phase="selected",
        batch_size=168,
    )

    assert events == [
        {
            "event": "batch_validation",
            "mode": "generation cache",
            "batch_size": 168,
            "max_new_tokens": 100,
        },
        {
            "event": "batch_validation_result",
            "mode": "generation cache",
            "batch_size": 168,
            "status": "PASS",
            "free_gib": 8.0,
            "recovered_gib": 12.0,
        },
        {
            "event": "batch_selected",
            "mode": "generation cache",
            "batch_size": 168,
        },
    ]
