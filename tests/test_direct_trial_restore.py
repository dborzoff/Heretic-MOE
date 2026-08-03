# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heretic.config import Settings


class DirectTrialRestoreConfigTests(unittest.TestCase):
    def test_accepts_exact_trial_number(self) -> None:
        with patch.object(sys, "argv", ["test-direct-trial-restore"]):
            settings = Settings(model="local/model", restore_trial_number=114)

        self.assertEqual(settings.restore_trial_number, 114)

    def test_restore_trial_number_is_not_archived_in_study_settings(self) -> None:
        with patch.object(sys, "argv", ["test-direct-trial-restore"]):
            settings = Settings(model="local/model", restore_trial_number=114)

        self.assertNotIn("restore_trial_number", settings.model_dump())


if __name__ == "__main__":
    unittest.main()
