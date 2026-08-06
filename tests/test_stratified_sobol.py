import unittest
import warnings
from collections import Counter

import optuna

from heretic.config import StartupDesign
from heretic.search import OptimizationRunner


class StratifiedSobolTests(unittest.TestCase):
    def test_direction_scope_is_balanced_without_qmc_fallback_warning(self):
        runner = OptimizationRunner(
            startup_design=StartupDesign.SOBOL,
            n_startup_trials=10,
            seed=17,
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=runner.initial_sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_categorical("direction_scope", ["global", "per layer"])
            return trial.suggest_float("x", 0.0, 1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.optimize_to(study, objective, target_trial_count=10)

        counts = Counter(trial.params["direction_scope"] for trial in study.trials)
        self.assertEqual(counts, {"global": 5, "per layer": 5})
        self.assertFalse(
            any("direction_scope" in str(warning.message) for warning in caught)
        )


if __name__ == "__main__":
    unittest.main()
