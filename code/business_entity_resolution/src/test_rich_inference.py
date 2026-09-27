import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fit_owner import FEATURE_NAMES
from predict_rich import read_route_features


class FakePair:
    def predict(self,matrix,num_threads):
        return np.array([.99,1e-9])


class RichInferenceTest(unittest.TestCase):
    def write_group(self,folder,rows=2):
        table=pa.table({'target_row':np.zeros(rows,dtype=np.int64),
                        'owner_row':np.array([3,4][:rows]),
                        **{n:np.zeros(rows,dtype=np.float32) for n in FEATURE_NAMES}})
        pq.write_table(table,folder/'features-00000.parquet')

    def test_preserves_tiny_runner_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write_group(folder)
            raw,runner=read_route_features(folder,np.array([0]),np.array([3]),1,FakePair(),k=2,threads=1)
            self.assertEqual(raw.shape,(1,len(FEATURE_NAMES)))
            self.assertEqual(runner[0],np.float32(1e-9))
            # This score vanishes if reconstructed from two saved float32 values.
            self.assertNotEqual(runner[0],np.float32(.99)-np.float32(.99-1e-9))

    def test_rejects_changed_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write_group(folder)
            with self.assertRaisesRegex(ValueError,'winner differs'):
                read_route_features(folder,np.array([0]),np.array([4]),1,FakePair(),k=2,threads=1)

    def test_requires_complete_rivals(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write_group(folder,rows=1)
            with self.assertRaisesRegex(ValueError,'Incomplete routed'):
                read_route_features(folder,np.array([0]),np.array([3]),1,FakePair(),k=2,threads=1)


if __name__=='__main__':unittest.main()
