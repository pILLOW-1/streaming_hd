import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/generation'))
from baselines.streaming.run import evaluate_metrics,objectives,choose,evaluate,matrix
from baselines.streaming.stream_metrics import select_threshold
from baselines.streaming.kernel import prefix_objective
from baselines.streaming.observed_lbfgs import ObservedLBFGS
from generation import validate_row
from hash_utils import sha256_tokens

class StreamingTests(unittest.TestCase):
 def test_prefix_objective_and_gradient(self):
  torch.manual_seed(7);x=torch.randn(7,requires_grad=True);y=torch.tensor([0.,1,0,1,1,0,1]);a=torch.sigmoid(torch.randn(7))
  actual=objectives(x,y,a,[3,4]);reference=prefix_objective(x,y,a,[3,4])
  self.assertTrue(torch.equal(actual,reference))
  self.assertTrue(torch.equal(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(reference,x)[0]))
 def test_metrics_pairwise_and_ties(self):
  y=np.asarray([0,1,0,1,1]);s=np.asarray([.1,.2,.2,.9,.9]);m=evaluate_metrics(y,s,.2)
  pair=np.mean([(float(p>n)+.5*float(p==n)) for p in s[y==1] for n in s[y==0]])
  self.assertAlmostEqual(m['AUROC'],pair);self.assertAlmostEqual(m['TPR_at_1pct_FPR'],2/3)
  self.assertAlmostEqual(m['F1'],.8)
 def test_f1_threshold_bruteforce(self):
  rng=np.random.default_rng(42)
  for _ in range(30):
   y=np.r_[0,1,rng.integers(0,2,15)];s=rng.integers(0,5,len(y))/5
   t=select_threshold(y,s);choices=[]
   for v in np.r_[np.unique(s),np.nextafter(s.min(),-np.inf)]:
    m=evaluate_metrics(y,s,v);choices.append((m['F1'],m['precision'],v))
   self.assertEqual(t,max(choices)[2])
 def test_no_layer_tiebreak(self):
  with self.assertRaises(ValueError):choose([dict(AUROC=.8,AP=.6,layer=1),dict(AUROC=.8,AP=.6,layer=2)])
 def test_observer_preserves_lbfgs(self):
  torch.manual_seed(2);x=torch.randn(15,3);y=torch.randint(0,2,(15,)).float();base=torch.nn.Linear(3,1);other=torch.nn.Linear(3,1);other.load_state_dict(base.state_dict());seen=[]
  for model,op in [(base,torch.optim.LBFGS),(other,ObservedLBFGS)]:
   kw=dict(lr=1,max_iter=8,line_search_fn='strong_wolfe')
   if op is ObservedLBFGS:kw['on_accepted']=lambda i:seen.append(i)
   opt=op(model.parameters(),**kw)
   def closure():
    opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)+1e-4*sum(.5*p.square().sum() for p in model.parameters());loss.backward();return loss
   opt.step(closure)
  self.assertTrue(seen);self.assertTrue(all(torch.equal(a,b) for a,b in zip(base.parameters(),other.parameters())))
 def test_detector_only_checkpoints(self):
  manifest=json.loads((ROOT/'experiments/streaming_622_p70/checkpoint/MANIFEST.json').read_text())
  self.assertEqual(len(manifest),30)
  for entry in manifest:
   f=ROOT/entry['path'];self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),entry['SHA256']);d=torch.load(f,map_location='cpu',weights_only=True);self.assertEqual(set(d),{'weight','bias'});self.assertTrue(all(torch.isfinite(v).all() for v in d.values()))
 def test_results_match_weights_and_confusion_counts(self):
  folder=ROOT/'experiments/streaming_622_p70'
  records=json.loads((folder/'result/metrics.json').read_text())
  weights=json.loads((folder/'checkpoint/MANIFEST.json').read_text())
  self.assertEqual(len(records),45)
  self.assertEqual(len({(r['model'],r['domain'],r['level']) for r in records}),45)
  for r in records:
   source='prefix' if r['level']=='answer' else r['level']
   w=next(w for w in weights if (w['model'],w['domain'],w['level'])==(r['model'],r['domain'],source))
   self.assertEqual(r['selected_layer'],w['layer'])
   tp,fp,tn,fn=(r[k] for k in ('TP','FP','TN','FN'))
   self.assertAlmostEqual(r['ACC'],(tp+tn)/(tp+fp+tn+fn))
   self.assertAlmostEqual(r['F1'],2*tp/(2*tp+fp+fn))
   for k in ('AUROC','AP','TPR_at_1pct_FPR'):
    self.assertTrue(0<=r[k]<=1)
 def test_assignment_disjointness_and_count(self):
  rows=[json.loads(x) for x in (ROOT/'experiments/streaming_622_p70/split_assignments.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),15697);self.assertEqual(len({(r['model'],r['domain'],r['sample_id']) for r in rows}),15697)
  counts={}
  for r in rows:counts[r['model'],r['domain'],r['split']]=counts.get((r['model'],r['domain'],r['split']),0)+1
  for e in json.loads((ROOT/'experiments/streaming_622_p70/data_identity.json').read_text()):self.assertEqual(counts[e['model'],e['domain'],e['split']],e['count'])
 def test_native_spans_and_hash(self):
  row=dict(sample_id='x',input_ids=[1,2,3,4],reasoning_start_token=1,retained_original_step_ids=[0,2],step_token_spans=[[1,2],[2,4]])
  validate_row(row);self.assertEqual(len(sha256_tokens(row['input_ids'])),64)
  row['step_token_spans'][1]=[2,5]
  with self.assertRaises(ValueError):validate_row(row)
 def test_answer_reuses_prefix_predictions_and_real_labels(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);scores=np.array([.9,.2,.1,.8]);np.savez(out/'prefix_val_predictions.npz',scores=scores);np.savez(out/'prefix_predictions.npz',scores=scores)
   rows={'val':[dict(retained_original_step_ids=[0,2],answer_hallucination=0),dict(retained_original_step_ids=[1,5],answer_hallucination=1)]}
   rows['test']=rows['val'];result=evaluate(rows,SimpleNamespace(device='cpu'),out,dict(layer=1),dict(width=3),'answer')
   with np.load(out/'answer_predictions.npz') as saved:self.assertEqual(saved['scores'].tolist(),[.2,.8]);self.assertEqual(saved['y'].tolist(),[0,1])
   self.assertEqual(result['AUROC'],1.0)

if __name__=='__main__':unittest.main()
