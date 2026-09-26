"""Candidate context respects entity boundaries and excludes the candidate itself."""
import unittest
import numpy as np

from train_context import CONTEXT_NAMES, context_features


class ContextTests(unittest.TestCase):
    def test_self_exclusion_and_source_agreement(self):
        scores = np.log(np.array([.9, .8, .2]) / np.array([.1, .2, .8]))
        result = context_features(np.zeros((3, 1)), scores, np.zeros(3), np.array([2, 2, 3]))[:, 1:]
        col = lambda name: result[:, CONTEXT_NAMES.index(name)]
        np.testing.assert_allclose(col("other_best_probability"), [.8, .9, .9])
        np.testing.assert_allclose(col("same_source_other_best"), [.8, .9, 0])
        np.testing.assert_allclose(col("opposite_source_best"), [.2, .2, .9])
        np.testing.assert_allclose(col("other_probability_sum"), [1., 1.1, 1.7])

    def test_no_cross_entity_information(self):
        x = np.zeros((4, 2))
        scores = np.array([1., -2., 12., -4.])
        query = np.array([0, 0, 1, 1])
        source = np.array([2, 3, 2, 3])
        combined = context_features(x, scores, query, source)
        for start in [0, 2]:
            np.testing.assert_array_equal(combined[start:start + 2], context_features(x[start:start + 2], scores[start:start + 2], query[start:start + 2], source[start:start + 2]))

    def test_one_candidate_and_tied_scores_are_finite(self):
        x = context_features(np.zeros((1, 2)), np.array([1000.]), np.array([0]), np.array([2]))
        self.assertTrue(np.isfinite(x).all())
        result = context_features(np.zeros((3, 2)), np.ones(3), np.zeros(3), np.array([2, 2, 2]))
        np.testing.assert_allclose(result[:, 2 + CONTEXT_NAMES.index("score_rank")], np.log1p([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
