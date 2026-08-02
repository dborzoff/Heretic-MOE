# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path

from heretic.config import DatasetSpecification
from heretic.scorers.perplexity import load_perplexity_text


class PerplexityTextTests(unittest.TestCase):
    def test_loads_local_utf8_file_without_dataset_hub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            text_path = Path(temporary_directory) / "ppl.txt"
            text_path.write_text("alpha beta gamma\n", encoding="utf-8")

            loaded = load_perplexity_text(
                DatasetSpecification(dataset=str(text_path))
            )

            self.assertEqual(loaded, "alpha beta gamma\n")


if __name__ == "__main__":
    unittest.main()
