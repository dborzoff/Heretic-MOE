# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from heretic.config import (
    MultilingualSearchSettings,
    ScorerConfig,
    SeedSelection,
    SelectionPolicy,
    Settings,
    StartupDesign,
)


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )

    def test_accepts_score_constraints(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.perplexity.Perplexity",
            optimization="minimize",
            constraint_lower=-0.001,
            constraint_upper=0.005,
        )

        self.assertEqual(config.constraint_lower, -0.001)
        self.assertEqual(config.constraint_upper, 0.005)

    def test_rejects_inverted_score_constraints(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "constraint_lower cannot exceed constraint_upper"
        ):
            ScorerConfig(
                plugin="heretic.scorers.perplexity.Perplexity",
                optimization="minimize",
                constraint_lower=0.01,
                constraint_upper=0.005,
            )


class SearchSettingsTests(unittest.TestCase):
    def test_generation_backend_defaults_are_stable(self) -> None:
        settings = Settings(model="example/model")

        self.assertEqual(settings.generation_backend.value, "dynamic_eager")
        self.assertEqual(settings.generation_prompt_bucket_multiple, 32)
        self.assertEqual(settings.generation_compile_mode, "default")

    def test_multilingual_search_contract_defaults_are_frozen(self) -> None:
        contract = MultilingualSearchSettings(
            enabled=True,
            dataset_root="F:/datasets/heretic_moe_5lang_v1",
        )

        self.assertEqual(contract.languages, ["en", "ru", "zh", "es", "fr"])
        self.assertEqual(contract.direction_rows_per_cell, 1000)
        self.assertEqual(contract.trial_rows_per_cell, 400)
        self.assertEqual(contract.calibration_rows_per_language, 132)
        self.assertEqual(contract.ordinary_max_new_tokens, 100)
        self.assertEqual(contract.final_max_new_tokens, 100)
        self.assertEqual(contract.evaluation_phase, "search")
        self.assertEqual(contract.schedule_capacity, 1000)
        self.assertIsNone(contract.runtime_root)
        self.assertEqual(contract.max_safe_ppl_drift, 0.005)
        self.assertEqual(contract.max_safe_geometry_damage, 1.0)

    def test_enabled_multilingual_search_requires_dataset_root(self) -> None:
        with self.assertRaisesRegex(ValidationError, "dataset_root"):
            MultilingualSearchSettings(enabled=True)

    def test_multilingual_search_contract_survives_journal_serialization(self) -> None:
        settings = Settings(
            model="example/model",
            multilingual_search={
                "enabled": True,
                "dataset_root": "F:/datasets/heretic_moe_5lang_v1",
                "runtime_root": "F:/runs/model-v3/runtime",
            },
        )

        restored = Settings.model_validate_json(settings.model_dump_json())

        self.assertTrue(restored.multilingual_search.enabled)
        self.assertEqual(
            restored.multilingual_search.runtime_root,
            "F:/runs/model-v3/runtime",
        )

    def test_multilingual_search_rejects_duplicate_languages(self) -> None:
        with self.assertRaisesRegex(ValidationError, "languages"):
            MultilingualSearchSettings(
                enabled=True,
                dataset_root="F:/datasets/heretic_moe_5lang_v1",
                languages=["en", "ru", "en"],
            )

    def test_multilingual_search_rejects_unbalanced_trial_size(self) -> None:
        with self.assertRaisesRegex(ValidationError, "trial_rows_per_cell"):
            MultilingualSearchSettings(
                enabled=True,
                dataset_root="F:/datasets/heretic_moe_5lang_v1",
                trial_rows_per_cell=2,
            )

    def test_search_extensions_are_disabled_by_default(self) -> None:
        with patch("sys.argv", ["test"]):
            settings = Settings(model="example/model")

        self.assertEqual(settings.startup_design, StartupDesign.RANDOM)
        self.assertEqual(settings.parameter_importance_interval, 0)
        self.assertFalse(settings.optimization_only)
        self.assertEqual(settings.seed_selection, SeedSelection.FIRST_OBJECTIVE)
        self.assertEqual(
            settings.selection_policy, SelectionPolicy.FEASIBLE_COST
        )
        self.assertFalse(settings.conditional_components)
        self.assertFalse(settings.tpe_group)
        self.assertEqual(settings.fused_expert_chunk_size, 8)
        self.assertTrue(settings.record_edit_telemetry)
        self.assertFalse(settings.multilingual_search.enabled)

    def test_sobol_and_optimization_only_are_explicit(self) -> None:
        with patch("sys.argv", ["test"]):
            settings = Settings(
                model="example/model",
                startup_design="sobol",
                parameter_importance_interval=20,
                optimization_only=True,
                seed_selection="spread",
            )

        self.assertEqual(settings.startup_design, StartupDesign.SOBOL)
        self.assertEqual(settings.parameter_importance_interval, 20)
        self.assertTrue(settings.optimization_only)
        self.assertEqual(settings.seed_selection, SeedSelection.SPREAD)

    def test_conditional_components_require_grouped_tpe(self) -> None:
        with patch("sys.argv", ["test"]):
            with self.assertRaisesRegex(
                ValidationError, "conditional_components requires tpe_group=true"
            ):
                Settings(
                    model="example/model",
                    conditional_components=True,
                )

    def test_grouped_tpe_allows_conditional_components(self) -> None:
        with patch("sys.argv", ["test"]):
            settings = Settings(
                model="example/model",
                conditional_components=True,
                tpe_group=True,
            )

        self.assertTrue(settings.conditional_components)
        self.assertTrue(settings.tpe_group)


if __name__ == "__main__":
    unittest.main()
