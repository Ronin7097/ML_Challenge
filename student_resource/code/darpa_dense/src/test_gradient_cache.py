"""Behavioral check: cached gradients equal retained-graph microbatch gradients."""
import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from train_encoder import embed, gradient_cache


class TinyEncoder(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.tokens = torch.nn.Embedding(50, 12)
        self.projection = torch.nn.Linear(12, 8)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.projection(self.dropout(self.tokens(input_ids))))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for training parity")
class GradientCacheTest(unittest.TestCase):
    def test_retained_graph_and_rng_replay(self):
        for dropout, source_rows in ((0.0,7),(0.2,7),(0.0,14),(0.2,14)):
            torch.manual_seed(731)
            reference = TinyEncoder(dropout).cuda().train()
            cached = copy.deepcopy(reference)
            batches = [{"input_ids":torch.randint(0,50,(n,3),device="cuda")} for n in (7,source_rows)]
            mask = torch.zeros((7,source_rows),dtype=torch.bool,device="cuda")
            mask[0,1] = mask[1,0] = True
            torch.manual_seed(81)
            reps=[]
            for tokens in batches:
                reps.append(torch.cat([embed(reference,{"input_ids":tokens["input_ids"][j:j+3]}) for j in range(0,len(tokens['input_ids']),3)]))
            expected=F.cross_entropy((reps[0]@reps[1].T/0.05).masked_fill(mask,float("-inf")),torch.arange(7,device="cuda"))
            expected.backward()
            torch.manual_seed(81)
            actual=gradient_cache(cached,batches,3,mask)
            self.assertAlmostEqual(actual,float(expected.detach()),places=5)
            for a,b in zip(reference.parameters(),cached.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=0.02,atol=0.002)


if __name__=="__main__":
    unittest.main()
