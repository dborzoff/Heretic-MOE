from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

import heretic.main as heretic_main
from heretic.main import _trial_display_label
from research.scripts.finalist_recheck import finalist_ranking_settings
from research.scripts.run_adaptive_search import (
    build_stage,
    preserve_existing_search_provenance,
    validate_adaptive_cost_contract,
)


EXPECTED_TARGETS = {
    "Sparse refusal geometry": -0.0088,
    "Keywords": 2 / 136,
    "Perplexity drift": 0.0,
}
EXPECTED_WEIGHTS = {
    "Sparse refusal geometry": 344.0,
    "Keywords": 697.68,
    "Perplexity drift": 200.0,
}


def adaptive_config() -> dict:
    return {
        "selection_policy": "feasible_cost",
        "selection_score_targets": dict(EXPECTED_TARGETS),
        "selection_score_weights": dict(EXPECTED_WEIGHTS),
        "scorers": [
            {
                "plugin": (
                    "heretic.scorers.sparse_refusal_geometry."
                    "SparseRefusalGeometry"
                )
            },
            {"plugin": "heretic.scorers.keyword_rate.KeywordRate"},
            {"plugin": "heretic.scorers.perplexity.Perplexity"},
        ],
    }


def test_cost_display_marks_the_higher_is_better_contract() -> None:
    formatter = getattr(heretic_main, "_format_selection_cost", None)

    assert formatter is not None
    assert formatter(0.25) == "Cost↑ 0.800"


def test_ppl_console_display_keeps_magnitude_and_signed_direction() -> None:
    record = {
        "name": "Perplexity drift",
        "score": {
            "value": 0.0752688172,
            "rich_display": "unused",
            "diagnostics": {"signed_relative_change": -0.07},
        },
    }

    assert heretic_main._display_score_record(record) == "7.53% (signed -7.00%)"
    assert heretic_main._leaderboard_score_parts(record) == [
        "PPL 7.53% (-7.00%)"
    ]


def test_all_adaptive_profiles_use_calibrated_cost() -> None:
    config_root = (
        Path(__file__).parents[1] / "research" / "configs" / "adaptive_search"
    )
    profiles = sorted(config_root.glob("*.toml"))
    assert profiles

    for profile in profiles:
        with profile.open("rb") as stream:
            config = tomllib.load(stream)
        validate_adaptive_cost_contract(config, source=profile)
        multilingual = config.get("multilingual_search")
        if isinstance(multilingual, dict) and multilingual.get("enabled"):
            assert "selection_score_targets" not in config
            assert "selection_score_weights" not in config
            continue
        assert config["selection_score_targets"] == EXPECTED_TARGETS
        assert config["selection_score_weights"] == EXPECTED_WEIGHTS


def test_adaptive_cost_contract_rejects_missing_weight() -> None:
    config = adaptive_config()
    del config["selection_score_weights"]["Keywords"]

    with pytest.raises(ValueError, match="Keywords"):
        validate_adaptive_cost_contract(config, source=Path("broken.toml"))


def test_adaptive_cost_contract_rejects_missing_scorer() -> None:
    config = adaptive_config()
    config["scorers"] = config["scorers"][:-1]

    with pytest.raises(ValueError, match="Perplexity"):
        validate_adaptive_cost_contract(config, source=Path("broken.toml"))


def test_recheck_label_preserves_source_and_local_trial_numbers() -> None:
    trial = SimpleNamespace(
        number=1,
        user_attrs={"index": 2, "recheck_source_trial_index": 962},
    )

    assert _trial_display_label(trial) == "T962 (recheck T2)"


def test_normal_label_keeps_search_trial_number() -> None:
    trial = SimpleNamespace(number=132, user_attrs={"index": 133})

    assert _trial_display_label(trial) == "T133"


def test_recheck_preserves_existing_search_stage_config(tmp_path: Path) -> None:
    stage_dir = tmp_path / "shared_tpe"
    stage_dir.mkdir()
    config_path = stage_dir / "config.toml"
    original = b'model = "immutable-search-model"\n'
    config_path.write_bytes(original)

    stage = build_stage(
        tmp_path,
        "shared_tpe",
        {"model": "new-finalist-model"},
        n_trials=1000,
        n_startup_trials=0,
        startup_design="random",
        device="0",
        response_archive=tmp_path / "responses.sqlite3",
        response_number_offset=0,
        response_number_stride=1,
        parallel_workers=2,
        dry_run=True,
        preserve_existing_config=True,
    )

    assert stage.config == config_path
    assert config_path.read_bytes() == original


def test_recheck_ranking_uses_current_cost_contract_not_stale_journal() -> None:
    source_settings = {
        "selection_diagnostics": ["legacy diagnostic"],
        "selection_score_targets": {},
        "selection_score_weights": {},
    }
    base_config = adaptive_config()
    base_config["selection_diagnostics"] = [
        "Sparse refusal geometry",
        "Keywords",
        "Perplexity drift",
    ]

    diagnostics, targets, weights = finalist_ranking_settings(
        source_settings,
        base_config,
    )

    assert diagnostics == base_config["selection_diagnostics"]
    assert targets == EXPECTED_TARGETS
    assert weights == EXPECTED_WEIGHTS


def test_new_search_with_recheck_post_step_creates_new_search_config(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(recheck_only=True, continue_shared_only=False)

    assert not preserve_existing_search_provenance(args, tmp_path)


def test_existing_recheck_only_continuation_preserves_search_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "shared_tpe" / "config.toml"
    config.parent.mkdir()
    config.write_text('model = "immutable"\n', encoding="utf-8")
    args = SimpleNamespace(recheck_only=True, continue_shared_only=True)

    assert preserve_existing_search_provenance(args, tmp_path)
