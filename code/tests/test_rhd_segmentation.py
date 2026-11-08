import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.rhd.segmentation import split_single_step_spans
from baselines.rhd.kernel import components
class RHDSegmentationTests(unittest.TestCase):
    def test_split_has_no_gap_overlap_or_changed_token(self):
        for end in [8,9]:
            original=[[3,end]];result,changed=split_single_step_spans(original)
            self.assertTrue(changed);self.assertEqual(original,[[3,end]])
            self.assertEqual([i for a,b in result for i in range(a,b)],list(range(3,end)))
            self.assertLessEqual(abs((result[0][1]-result[0][0])-(result[1][1]-result[1][0])),1)
    def test_multiple_steps_unchanged_and_single_token_explicit(self):
        value=[[3,7],[9,13]];result,changed=split_single_step_spans(value);self.assertFalse(changed);self.assertEqual(result,value)
        with self.assertRaisesRegex(ValueError,'SINGLE_TOKEN'):split_single_step_spans([[3,4]])
    def test_two_halves_make_cv_domain_nonempty(self):
        result=components([1.,2.],[.2,.3],[[0.,0.],[.1,0.]])
        self.assertTrue(np.isfinite(result).all());self.assertEqual(result[1],0.);self.assertEqual(result[3],0.)
if __name__=='__main__':unittest.main()
