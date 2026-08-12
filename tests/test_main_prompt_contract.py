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
        search_rows=tuple(range(10)),
        final_rows=tuple(range(10)),
    )

    assert _format_multilingual_frozen_rows(bundle) == (
        "map 40, trial 50, SRG calibration 10, final holdout 10"
    )


def test_generation_runtime_fields_survive_checkpoint_continuation() -> None:
    from heretic.main import _ALWAYS_RUNTIME_FIELDS

    assert {
        "generation_backend",
        "generation_prompt_bucket_multiple",
        "generation_compile_mode",
    }.issubset(_ALWAYS_RUNTIME_FIELDS)
