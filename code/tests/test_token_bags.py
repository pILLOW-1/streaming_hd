"""HaMI ori/SE compatibility and exact BF16 storage without full-bank copies."""
import sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.baseline_training import token_bags

class TokenBagTests(unittest.TestCase):
    def test_ori_needs_no_semantic_probability(self):
        x=np.arange(12,dtype=np.float32).reshape(4,3)
        bags=token_bags(dict(tokens=x,offsets=np.array([0,1,4]),hami_mode='ori'))
        self.assertEqual(len(bags),2)
        torch.testing.assert_close(bags[0],torch.from_numpy(x[:1])[:,None],rtol=0,atol=0)
        torch.testing.assert_close(bags[1],torch.from_numpy(x[1:])[:,None],rtol=0,atol=0)
        with self.assertRaises(IndexError):bags[2]
    def test_se_retains_original_augmentation(self):
        x=np.array([[1,-2],[3,-4]],dtype=np.float32)
        bags=token_bags(dict(tokens=x,offsets=np.array([0,1,2]),semantic_probability=np.array([.25,.5])))
        torch.testing.assert_close(bags[0],torch.tensor([[[1.25,-2.5]]]),rtol=0,atol=0)
        torch.testing.assert_close(bags[1],torch.tensor([[[4.5,-6.]]]),rtol=0,atol=0)
    def test_bf16_bits_round_trip_and_readonly_mapping(self):
        expected=torch.tensor([[1e8,-1e8,0],[.1,-.25,3]],dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tokens.npy';np.save(path,expected.view(torch.uint16).numpy())
            bank=np.load(path,mmap_mode='r');bags=token_bags(dict(tokens=bank,offsets=np.array([0,1,2]),hami_mode='ori',token_dtype='bfloat16_bits'))
            torch.testing.assert_close(bags[0][:,0],expected[:1].float(),rtol=0,atol=0)
            torch.testing.assert_close(bags[1][:,0],expected[1:].float(),rtol=0,atol=0)
            torch.testing.assert_close(bags[0][:,0],expected[:1].float(),rtol=0,atol=0)
            self.assertFalse(bank.flags.writeable)
if __name__=='__main__':unittest.main()
