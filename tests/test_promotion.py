# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState, create_trial

from heretic.config import SeedSelection
from heretic.promotion import load_seed_parameters


class PromotionTests(unittest.TestCase):
    def test_promotes_feasible_external_params_and_renames_routed_expert(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source.jsonl"
            storage = JournalStorage(
                JournalFileBackend(
                    str(path),
                    lock_obj=JournalFileOpenLock(str(path)),
                )
            )
            study = optuna.create_study(
                study_name="source",
                storage=storage,
                directions=["minimize", "minimize"],
            )
            study.set_user_attr("constraint_names", ["ppl <= 0.005"])

            distributions = {
                "direction_scope": CategoricalDistribution(["global", "per layer"]),
                "direction_index": FloatDistribution(4.0, 9.0),
                "mlp.down_proj.max_weight": FloatDistribution(-0.25, 2.5),
            }
            common = {
                "direction_scope": "global",
                "direction_index": 6.0,
                "mlp.down_proj.max_weight": 1.2,
            }
            trials = [
                create_trial(
                    state=TrialState.COMPLETE,
                    values=[0.0, 0.2],
                    params=common,
                    distributions=distributions,
                    user_attrs={"constraints": [-0.001]},
                ),
                create_trial(
                    state=TrialState.COMPLETE,
                    values=[0.1, 0.05],
                    params=dict(common, direction_index=7.0),
                    distributions=distributions,
                    user_attrs={"constraints": [-0.002]},
                ),
                create_trial(
                    state=TrialState.COMPLETE,
                    values=[0.0, 0.01],
                    params=dict(common, direction_index=8.0),
                    distributions=distributions,
                    user_attrs={"constraints": [0.02]},
                ),
            ]
            study.add_trials(trials)

            seeds = load_seed_parameters(
                str(path),
                1,
                ["mlp.experts.down_proj"],
                SeedSelection.FIRST_OBJECTIVE,
            )

            self.assertEqual(len(seeds), 1)
            self.assertEqual(seeds[0]["direction_scope"], "global")
            self.assertEqual(seeds[0]["direction_index"], 6.0)
            self.assertEqual(seeds[0]["mlp.experts.down_proj.max_weight"], 1.2)


if __name__ == "__main__":
    unittest.main()
