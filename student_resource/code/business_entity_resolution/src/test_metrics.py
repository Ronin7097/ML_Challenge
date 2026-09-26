"""Regression checks for competition scoring and threshold selection."""
import unittest

import numpy as np

from train_boost import best_threshold, entity_scores, metrics


class MetricTests(unittest.TestCase):
    def test_singletons_and_missed_links(self):
        result = entity_scores(np.array([0, 0, 0, 2]), np.array([0, 1, 0, 1]), np.array([0, 0, 3, 2]))
        np.testing.assert_allclose(result, [1, 0, 0, 2.5 / 3.5])

    def test_false_negatives_include_unretrieved_truth(self):
        result = metrics(np.array([True]), np.array([1]), np.array([0]), np.array([4, 0]))
        self.assertAlmostEqual(result["macro_f0.5"], (1.25 / 2 + 1) / 2)
        self.assertEqual(result["fn"], 3)

    def test_exact_sweep_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(41)
        for _ in range(20):
            queries = np.repeat(np.arange(20), 6)
            labels = rng.integers(0, 2, size=len(queries))
            truth = np.bincount(queries[labels == 1], minlength=20) + rng.integers(0, 3, size=20)
            scores = np.round(rng.normal(size=len(queries)), 1)
            threshold, best = best_threshold(scores, labels, queries, truth)
            all_thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
            expected = max(metrics(scores >= t, labels, queries, truth)["macro_f0.5"] for t in all_thresholds)
            self.assertAlmostEqual(best, expected)
            self.assertAlmostEqual(metrics(scores >= threshold, labels, queries, truth)["macro_f0.5"], expected)


if __name__ == "__main__":
    unittest.main()
