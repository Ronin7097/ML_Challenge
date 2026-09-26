"""Behavior checks for choosing complete prediction sets per entity."""
import unittest

import numpy as np

from decision_policy import select_matches


class DecisionTests(unittest.TestCase):
    def choose(self, probabilities, query):
        p = np.asarray(probabilities, dtype=float)
        return select_matches(np.log(p / (1 - p)), np.asarray(query), 0, "expected_f0.5")

    def test_empty_singleton_or_confident_match(self):
        np.testing.assert_array_equal(self.choose([.1, .95], [0, 1]), [False, True])

    def test_multiple_matches_and_distractor(self):
        np.testing.assert_array_equal(self.choose([.98, .97, .01], [0, 0, 0]), [True, True, False])

    def test_interleaved_entities_do_not_change_each_other(self):
        p, q = [.98, .1, .97, .15, .01], [7, 4, 7, 4, 7]
        combined = self.choose(p, q)
        for entity in set(q):
            mask = np.asarray(q) == entity
            np.testing.assert_array_equal(combined[mask], self.choose(np.asarray(p)[mask], np.asarray(q)[mask]))

    def test_empty_input(self):
        self.assertEqual(len(self.choose([], [])), 0)

    def test_threshold_compatibility(self):
        np.testing.assert_array_equal(select_matches(np.array([-1., 0., 1.]), np.array([0, 0, 0]), 0), [False, True, True])


if __name__ == "__main__":
    unittest.main()
