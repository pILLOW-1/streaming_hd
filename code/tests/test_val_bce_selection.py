"""VAL BCE resolves metric ties without preferring earlier updates or layers."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.checkpoint_selection import best_snapshot
from common.baseline_training import fit_vector,Progress,seed_all
import tempfile
class BceSelectionTests(unittest.TestCase):
    def test_lower_bce_wins_equal_auc_ap_in_either_arrival_order(self):
        first={'weight':torch.tensor([1.])};second={'weight':torch.tensor([2.])}
        b=best_snapshot((.8,.7,-.5),first,None);b=best_snapshot((.8,.7,-.3),second,b)
        self.assertFalse(b['tie']);self.assertEqual(b['state']['weight'].item(),2.)
        b=best_snapshot((.8,.7,-.3),second,None);b=best_snapshot((.8,.7,-.5),first,b)
        self.assertFalse(b['tie']);self.assertEqual(b['state']['weight'].item(),2.)
    def test_actual_global_mean_callbacks_have_three_val_criteria(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(40,3)).astype(np.float32);v=rng.normal(size=(24,3)).astype(np.float32);y=(x[:,0]>0).astype(int);vy=(v[:,0]>0).astype(int)
        seen=[]
        def record(rank,state,best):seen.append(rank);return best_snapshot(rank,state,best)
        with tempfile.TemporaryDirectory() as d:
            seed_all(42)
            with patch('common.checkpoint_selection.best_snapshot',record):
                fit_vector(dict(method_key='global_mean',level='step',select_weights_by_val_auroc=True,selection_tie_breaker='VAL_BCE'),dict(x=x),dict(x=v),y,vy,'cpu',Progress(Path(d)/'progress.pt',{'test':'VAL_BCE'}))
        self.assertTrue(seen);self.assertTrue(all(len(rank)==3 and np.isfinite(rank[2]) for rank in seen))
if __name__=='__main__':unittest.main()
