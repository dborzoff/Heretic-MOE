# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from heretic.config import ScorerConfig, Settings, StartupDesign


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


class SearchSettingsTests(unittest.TestCase):
    def test_search_extensions_are_disabled_by_default(self) -> None:
        with patch("sys.argv", ["test"]):
            settings = Settings(model="example/model")

        self.assertEqual(settings.startup_design, StartupDesign.RANDOM)
        self.assertEqual(settings.parameter_importance_interval, 0)
        self.assertFalse(settings.optimization_only)

    def test_sobol_and_optimization_only_are_explicit(self) -> None:
        with patch("sys.argv", ["test"]):
            settings = Settings(
                model="example/model",
                startup_design="sobol",
                parameter_importance_interval=20,
                optimization_only=True,
            )

        self.assertEqual(settings.startup_design, StartupDesign.SOBOL)
        self.assertEqual(settings.parameter_importance_interval, 20)
        self.assertTrue(settings.optimization_only)


if __name__ == "__main__":
    unittest.main()
