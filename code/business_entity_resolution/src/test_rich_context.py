import unittest

import numpy as np

from rich_context import EXTRA_NAMES, PAIR_NAMES, enrich_row, pair_evidence, peer_indices


class RichContextTest(unittest.TestCase):
    def test_number_edit_is_evidence_not_veto(self):
        x=dict(zip(PAIR_NAMES,pair_evidence(('Acme','12 Main Road'),('Acme','102 Main Rd'))))
        self.assertEqual(x['rich_number_subsequence'],1)
        self.assertEqual(x['rich_number_equal'],0)
        self.assertEqual(x['rich_address_alpha_exact'],1)

    def test_distinguishes_nearby_rival(self):
        x=dict(zip(EXTRA_NAMES,enrich_row((('Acme Ltd','12 Main Road'),('Acme Limited','12 Main Rd'),
                          ('Acme Ltd','92 Main Road'),None,-1,0))))
        self.assertEqual(x['rich_number_vs_dense_rival'],1)
        self.assertEqual(x['rich_peer_exists'],0)

    def test_peer_is_other_and_permutation_invariant(self):
        owner=np.array([2,2,3,2]);prob=np.array([.9,.9,.8,.7]);keys=[('B',''),('A',''),('C',''),('D','')]
        peers=peer_indices(owner,prob,keys)
        self.assertTrue(all(j!=i for i,j in enumerate(peers) if j>=0))
        self.assertEqual(peers[2],-1)
        order=np.array([3,1,0,2]);reverse=np.argsort(order)
        other=peer_indices(owner[order],prob[order],[keys[i] for i in order])
        mapped=np.array([order[j] if j>=0 else -1 for j in other])[reverse]
        np.testing.assert_array_equal(peers,mapped)

    def test_unicode_and_empty_records_finite(self):
        for a,b in [(('',''),('','')),(('भारत','१२ रोड'),('Bharat','12 Road')),(('Café','12 rue'),('Cafe','12 rue'))]:
            x=pair_evidence(a,b)
            self.assertEqual(len(x),len(PAIR_NAMES))
            self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':unittest.main()
