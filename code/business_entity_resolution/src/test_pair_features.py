import unittest
import numpy as np
from pair_features import FEATURE_NAMES, normalized, shortlist
from evaluate_tune import entity_scores
from fit_owner import incoming_context, WINNER_NAMES, CONTEXT_NAMES


class FeaturesTest(unittest.TestCase):
    def test_unicode_and_numeric_competition(self):
        self.assertEqual(normalized("Café"), "cafe")
        self.assertIn("ा", normalized("भारत"))
        x = shortlist([("Acme Ltd", "12 Main Road 110001"), ("Acme Ltd", "92 Main Road 110002")],
                      ("Acme Limited", "12 Main Rd 110001"), np.array([.99,.97]), np.array([2,2]), False)
        self.assertEqual(x.shape, (2,len(FEATURE_NAMES)))
        self.assertEqual(x[0,FEATURE_NAMES.index("core_exact")], 1)
        self.assertEqual(x[1,FEATURE_NAMES.index("first_number_conflict")], 1)
        self.assertGreater(x[0,FEATURE_NAMES.index("address_ratio_to_best_rival")], 0)
        self.assertTrue(np.isfinite(shortlist([("", ""), ("A", "")], ("", ""), np.array([.1,0]), [1,1], True)).all())

    def test_macro_includes_empty_queries_and_weights_precision(self):
        gold={"a":set(),"b":{"x"},"c":{"y","z"},"d":{"p"}}
        pred={"a":set(),"b":{"x","wrong"},"c":{"y"},"d":set()}
        scores, counts=entity_scores(gold,pred,list(gold))
        np.testing.assert_allclose(scores,[1,1.25/2.25,1.25/1.5,0])
        self.assertEqual(counts,{"tp":2,"fp":1,"fn":2})

    def test_incoming_support_excludes_current_edge(self):
        x=np.zeros((4,len(WINNER_NAMES)),np.float32)
        x[:,WINNER_NAMES.index('pair_probability')]=[.9,.7,.9,.6]
        x[:,WINNER_NAMES.index('target_source3')]=[0,1,0,0]
        context=incoming_context(np.array([0,0,1,0]),x,2)
        np.testing.assert_allclose(context[:,CONTEXT_NAMES.index('best_other_probability')],[.7,.9,-1,.9])
        np.testing.assert_allclose(context[:,CONTEXT_NAMES.index('other_probability_sum')],[1.3,1.5,0,1.6],atol=1e-6)
        np.testing.assert_allclose(context[:,CONTEXT_NAMES.index('other_opposite_source_count_05')],[1,2,0,1])
        x[:,WINNER_NAMES.index('pair_probability')]=[.9,.9,.7,.7]
        context=incoming_context(np.array([0,0,0,0]),x,1)
        np.testing.assert_allclose(context[:,CONTEXT_NAMES.index('incoming_rank')],[0,0,2,2])


if __name__=="__main__":
    unittest.main()
