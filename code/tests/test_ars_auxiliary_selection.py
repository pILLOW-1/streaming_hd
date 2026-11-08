import unittest
from baselines.ars.auxiliary_selection import select_auxiliary


class SelectionTests(unittest.TestCase):
    def test_false_is_preserved_and_only_invalid_auxiliary_position_is_removed(self):
        records = [dict(sample_id='a',status='INVALID_OFFICIAL_JUDGE_OUTPUT',agreement=[True,False,None,True,False],invalid_outputs=[[None,'**']]),
                   dict(sample_id='b',status='PASS',agreement=[True,False,True,False,True],invalid_outputs=[])]
        selected = select_auxiliary(records)
        self.assertEqual(selected['source_positions'],[0,1,3,4,5,6,7,8,9])
        self.assertEqual(selected['original_ids'],[0,0,0,0,1,1,1,1,1])
        self.assertEqual(selected['agreement'],[True,False,True,False,True,False,True,False,True])
        self.assertEqual(selected['supervised_original_count'],2)
        self.assertEqual(selected['isolated_format_variants'][0]['variant_index'],1)

    def test_all_invalid_variants_still_keep_original_supervised_reference(self):
        record=dict(sample_id='a',status='INVALID_OFFICIAL_JUDGE_OUTPUT',agreement=[True,None,None,None,None],invalid_outputs=[[None,'**']]*4)
        selected=select_auxiliary([record])
        self.assertEqual(selected['source_positions'],[0])
        self.assertEqual(selected['supervised_original_count'],1)
        self.assertEqual(selected['retained_generated_variants'],0)

    def test_oom_remains_separate_from_authorized_format_isolation(self):
        record=dict(sample_id='a',status='INVALID_OFFICIAL_JUDGE_OUTPUT',agreement=[True,None,None,False,True],invalid_outputs=[[None,'**'],[None,'[ERROR] CUDA out of memory']])
        selected=select_auxiliary([record])
        self.assertEqual(len(selected['isolated_format_variants']),1)
        self.assertEqual(len(selected['unresolved_computation_variants']),1)

    def test_string_false_and_inexact_failure_mapping_are_rejected(self):
        with self.assertRaises(ValueError):select_auxiliary([dict(sample_id='a',status='PASS',agreement=[True,'false',True,True,True])])
        with self.assertRaises(ValueError):select_auxiliary([dict(sample_id='a',status='INVALID_OFFICIAL_JUDGE_OUTPUT',agreement=[True,None,True,True,True],invalid_outputs=[])])


if __name__=='__main__':
    unittest.main()
