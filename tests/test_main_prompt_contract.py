from __future__ import annotations

from types import SimpleNamespace


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
