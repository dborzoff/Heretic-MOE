import hashlib
import unittest

from heretic.config import DatasetSpecification
from heretic.scorers.perplexity import (
    load_perplexity_text,
    paired_relative_perplexity_interval,
)


class PerplexityUncertaintyTests(unittest.TestCase):
    def test_identical_windows_are_not_distinguishable(self):
        result = paired_relative_perplexity_interval(
            [2.0, 2.1, 1.9], [2.0, 2.1, 1.9]
        )

        self.assertEqual(result["relative_change_ci95_lower"], 0.0)
        self.assertEqual(result["relative_change_ci95_upper"], 0.0)
        self.assertFalse(result["statistically_distinguishable_from_baseline"])

    def test_consistent_improvement_keeps_negative_interval(self):
        result = paired_relative_perplexity_interval(
            [1.9, 2.0, 1.8, 1.95], [2.0, 2.1, 1.9, 2.05]
        )

        self.assertLess(result["relative_change_ci95_upper"], 0.0)
        self.assertTrue(result["statistically_distinguishable_from_baseline"])

    def test_noisy_small_change_crosses_zero(self):
        result = paired_relative_perplexity_interval(
            [2.02, 2.08, 1.93, 2.01], [2.0, 2.1, 1.9, 2.05]
        )

        self.assertLess(result["relative_change_ci95_lower"], 0.0)
        self.assertGreater(result["relative_change_ci95_upper"], 0.0)
        self.assertFalse(result["statistically_distinguishable_from_baseline"])

    def test_window_count_must_match(self):
        with self.assertRaises(ValueError):
            paired_relative_perplexity_interval([2.0], [2.0, 2.1])

    def test_builtin_corpus_is_frozen_and_available_offline(self):
        text = load_perplexity_text(
            DatasetSpecification(dataset="builtin://perplexity-reference-v1")
        )

        payload = text.encode("utf-8")
        self.assertEqual(len(payload), 241_986)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "49d7e8f6f3eeacc3fd95e8436bb28278746fdfd47994be4d1da46a36a6228fc3",
        )


if __name__ == "__main__":
    unittest.main()
