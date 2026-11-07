import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from common.dataset import load_cell, verify_assignment


class DatasetTests(unittest.TestCase):
    def write_fixture(self, root):
        assignments = []
        for i, split in enumerate(('train','val','test')):
            d = root/'model'/'Math'/split
            (d/'generation').mkdir(parents=True)
            (d/'annotations').mkdir()
            identity = dict(sample_id=f's{i}', model='model', domain='Math',
                            canonical_question_id=f'q{i}', question_family_id=f'f{i}')
            generation = dict(identity, original_steps=['a','excluded','b'], retained_original_step_ids=[0,2])
            annotation = dict(identity, labels=dict(step=[True,False],prefix=[1,0],answer=False))
            (d/'generation/records.jsonl').write_text(json.dumps(generation)+'\n')
            (d/'annotations/labels.jsonl').write_text(json.dumps(annotation)+'\n')
            assignments.append(dict(identity, split=split))
        path=root/'assignments.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in assignments))
        return path

    def test_authoritative_layout_and_independent_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); assignment=self.write_fixture(root)
            rows=load_cell({'dataset_root':tmp},'model','Math')
            self.assertEqual(rows['train'][0]['prefix_hallucination'],[1,0])
            self.assertEqual(rows['train'][0]['original_steps'],['a','excluded','b'])
            verify_assignment(rows,'model','Math',assignment)
            rows['test'][0]['sample_id']='wrong'
            with self.assertRaises(ValueError):verify_assignment(rows,'model','Math',assignment)

    def test_group_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.write_fixture(root)
            for sub in ('generation/records.jsonl','annotations/labels.jsonl'):
                p=root/'model'/'Math'/'test'/sub
                row=json.loads(p.read_text());row['question_family_id']='f0';p.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'crosses splits'):
                load_cell({'dataset_root':tmp},'model','Math')

    def test_missing_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.write_fixture(root)
            p=root/'model'/'Math'/'val'/'annotations/labels.jsonl'
            row=json.loads(p.read_text());row['labels']['prefix']=[1];p.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete prefix'):
                load_cell({'dataset_root':tmp},'model','Math')
