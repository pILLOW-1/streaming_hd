import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from baselines.run import load_pack, predict
from common.baseline_training import Progress, fit_vector, seed_all
from common.native_inputs import validate_full_input


class PortableWorkflows(unittest.TestCase):
    def test_saplma_fits_and_checkpoint_reproduces_predictions(self):
        rng = np.random.default_rng(53)
        train = {'x': rng.normal(size=(64, 4)).astype('float32')}
        val = {'x': rng.normal(size=(40, 4)).astype('float32')}
        y = (train['x'][:, 0] > 0).astype(int)
        vy = (val['x'][:, 0] > 0).astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            seed_all(42)
            progress = Progress(Path(tmp) / 'progress.pt', {'test': 'saplma'})
            cfg = dict(method_key='saplma', level='step', select_weights_by_val_auroc=False)
            model, options = fit_vector(cfg, train, val, y, vy, 'cpu', progress)
            scores = predict(model, val, options, 'cpu')
            self.assertEqual(scores.shape, (40,))
            self.assertTrue(np.isfinite(scores).all())
            self.assertGreater(float(np.std(scores)), 0)
            restored, opts = fit_vector(cfg, train, val, y, vy, 'cpu', progress)
            np.testing.assert_array_equal(scores, predict(restored, val, opts, 'cpu'))

    def test_raw_runner_uses_val_without_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                folder = root / 'data/model/Math' / split
                (folder / 'generation').mkdir(parents=True)
                (folder / 'annotations').mkdir()
                generated, annotated, keys = [], [], []
                for i in range(8):
                    identity = dict(sample_id=f'{split}_{i}', model='model', domain='Math',
                                    question_family_id=f'{split}_f{i}', canonical_question_id=f'{split}_q{i}')
                    generated.append(dict(identity, original_steps=['a'], retained_original_step_ids=[0]))
                    annotated.append(dict(identity, labels=dict(step=[i % 2], prefix=[i % 2], answer=i % 2)))
                    keys.append(f'{split}_{i}:0')
                for sub, rows in [('generation/records.jsonl', generated), ('annotations/labels.jsonl', annotated)]:
                    (folder / sub).write_text(''.join(json.dumps(r) + '\n' for r in rows))
                dest = root / 'features/llm_check/model/Math/prefix/layer_1'
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / (split + '.npz')
                np.savez(path, score=np.asarray([.1,.8,.2,.7,.3,.9,.4,.6]), position_id=np.asarray(keys))
                path.with_suffix('.json').write_text(json.dumps(dict(method='llm_check', level='prefix', layer=1, SHA256=hashlib.sha256(path.read_bytes()).hexdigest())))
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps(dict(method_key='llm_check', level='prefix', layers=[1], score_sign=1)))
            path_cfg = root / 'paths.json'
            path_cfg.write_text(json.dumps(dict(dataset_root=str(root/'data'), baseline_feature_root=str(root/'features'), output_root=str(root/'out'))))
            subprocess.run([sys.executable, str(ROOT/'code/baselines/run.py'), '--paths', str(path_cfg), '--recipe', str(recipe), '--model', 'model', '--domain', 'Math', '--device', 'cpu'], check=True, capture_output=True, text=True)
            result = json.loads((root/'out/llm_check/model/Math/prefix/result.json').read_text())
            self.assertEqual(result['TEST']['AUROC'], 1)
            self.assertEqual(result['TEST']['F1'], 1)
            self.assertEqual(result['selected']['layer'], 1)
            pack = root/'features/llm_check/model/Math/prefix/layer_1/val'
            from common.dataset import load_cell
            rows = load_cell({'dataset_root': str(root/'data')}, 'model', 'Math')['val']
            with self.assertRaisesRegex(ValueError, 'method identity'):
                load_pack(pack, rows, 'prefix', 'saplma', 1)
            with self.assertRaisesRegex(ValueError, 'layer identity'):
                load_pack(pack, rows, 'prefix', 'llm_check', 2)
            with pack.with_suffix('.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_pack(pack, rows, 'prefix')

    def test_full_answer_tokenizer_offsets(self):
        sys.path.insert(0, str(ROOT/'code/generation'))
        from prepare_full_inputs import prepare_full_record
        def characters(text, **kwargs):
            return dict(input_ids=list(map(ord,text)), offset_mapping=[(i,i+1) for i in range(len(text))])
        record=dict(sample_id='a',prompt='Q:',reasoning='one;two',answer_separator='\nA:',answer='yes',original_steps=['one','two'])
        n=prepare_full_record(record,characters)
        validate_full_input(n,record)
        self.assertEqual(n['step_token_spans'],[[2,5],[6,9]])
        self.assertEqual(n['answer_token_span'],[12,15])
        self.assertEqual(n['reasoning_start_token'],2)
        self.assertEqual(''.join(map(chr,n['input_ids'])),'Q:one;two\nA:yes')

    def test_full_answer_rejects_wrong_answer_and_tokens(self):
        row = dict(sample_id='a', original_steps=['reason'], answer='answer')
        ids = [1, 2, 3, 4]
        n = dict(row, input_ids=ids, reasoning_start_token=1, step_token_spans=[[1,3]],
                 answer_token_span=[3,4], token_input_SHA256=hashlib.sha256(struct.pack('<4q', *ids)).hexdigest())
        validate_full_input(n, row)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_full_input(dict(n, answer='gold instead of generated'), row)
        with self.assertRaisesRegex(ValueError, 'token hash mismatch'):
            validate_full_input(dict(n, input_ids=[1,2,3,5]), row)


if __name__ == '__main__':
    unittest.main()
