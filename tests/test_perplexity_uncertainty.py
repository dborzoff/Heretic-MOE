import hashlib
import unittest

from heretic.config import DatasetSpecification
from heretic.scorers.perplexity import (
    load_perplexity_text,
    paired_relative_perplexity_interval,
    symmetric_perplexity_change,
)


class PerplexityUncertaintyTests(unittest.TestCase):
    def test_symmetric_drift_treats_reciprocal_changes_equally(self):
        doubled = symmetric_perplexity_change(160.0, 80.0)
        halved = symmetric_perplexity_change(40.0, 80.0)

        self.assertEqual(doubled["signed_relative_change"], 1.0)
        self.assertEqual(halved["signed_relative_change"], -0.5)
        self.assertAlmostEqual(doubled["symmetric_drift"], 1.0)
        self.assertAlmostEqual(halved["symmetric_drift"], 1.0)

    def test_symmetric_drift_is_zero_at_the_clean_baseline(self):
        result = symmetric_perplexity_change(80.0, 80.0)

        self.assertEqual(result["signed_log_delta"], 0.0)
        self.assertEqual(result["signed_relative_change"], 0.0)
        self.assertEqual(result["symmetric_drift"], 0.0)

    def test_symmetric_drift_rejects_non_positive_perplexity(self):
        with self.assertRaises(ValueError):
            symmetric_perplexity_change(0.0, 80.0)

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
        # The built-in loader deliberately normalizes CRLF to the committed LF
        # representation before hashing and returning the frozen corpus.
        self.assertEqual(len(payload), 241_748)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "1d6f25ca80bd49255212d67d7eff96763ab01abbd472c04b916ec62318857a9d",
        )


if __name__ == "__main__":
    unittest.main()
