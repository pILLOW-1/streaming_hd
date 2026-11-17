"""Locate HaMI's highest scoring token in the frozen error-only VAL cohort."""
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from diagnose import digest

domain = sys.argv[1]
home = Path.home()
project = home / 'agent_uncertainty_quantification'
runs = project / 'experiments/streaming_hd/runs'
base = runs / 'baselines_622_p70_20261108'
cell = base / 'fits/qwen3_8_27b' / domain / 'hami/step'
cohort = runs / 'qwen_step_aggr_concentration_20261117/errors_val'
reference = json.loads((cohort / 'result' / (domain + '.json')).read_text())
source = base / 'hami_sources/qwen3_8_27b' / domain / 'reports/FEATURE_MANIFEST.json'
manifest = json.loads(source.read_text())
frozen = json.loads((cell / 'reports/FROZEN_SELECTION.json').read_text())
assert frozen['branch'] == 'ori' and frozen['domain'] == domain
assert digest(source) == frozen['source_manifest_SHA256']
assert socket.gethostname().split('.')[0] == manifest['host'].split('.')[0]
layer = frozen['selected']['layer']
feature_root = Path(manifest['feature_root'])
relative = 'layer_{}/hami_tokens.npy'.format(layer)
feature = feature_root / relative
checkpoint = cell / 'checkpoints' / ('layer_%d.pt' % layer)
pred = cell / 'reports' / ('val_layer_%d.npz' % layer)
assert digest(checkpoint) == frozen['checkpoint_SHA256']
assert digest(pred) == frozen['VAL_predictions_SHA256']
pin = manifest['files'][relative]
assert feature.stat().st_size == pin['bytes'] and digest(feature) == pin['SHA256']
offsets_file = feature_root / 'OFFSETS.json'
assert hashlib.sha256(json.dumps(json.loads(offsets_file.read_text())).encode()).hexdigest() == manifest['identity']['token_offsets_SHA256']
offsets = np.asarray(json.loads(offsets_file.read_text()))
ids = json.loads((feature_root / 'POSITIONS.json').read_text())
a, b = manifest['identity']['ranges']['val']
with np.load(pred, allow_pickle=False) as saved:
    labels = saved['labels']
    keys = saved['position_id'].tolist()
    expected = saved['scores'][labels == 1]
assert ids[a:b] == keys and len(keys) == reference['all_val_positions']
chosen = np.flatnonzero(labels == 1)
pos = [keys[i] for i in chosen]
assert len(pos) == reference['positions']
assert hashlib.sha256(json.dumps(pos).encode()).hexdigest() == reference['error_positions_sha256']

upstream = project / 'workspaces/streaming_hd/baseline/hami/upstream/3d277605534999d2756a6eeef759d0a91199ad58'
sys.path.insert(0, str(upstream))
from hami import HaMI

model = HaMI(5120, 128)
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert state['identity']['source_manifest_SHA256'] == frozen['source_manifest_SHA256']
model.load_state_dict(state['state_dict'])
model.eval()
raw = np.load(feature, mmap_mode='r', allow_pickle=False)
assert raw.shape == (offsets[-1], 5120) and raw.dtype == np.uint16
bins = np.zeros(10, dtype=np.float64)
max_residual = 0.
tied = 0
with torch.inference_mode():
    for start in range(0, len(chosen), 64):
        indices = chosen[start:start+64] + a
        lengths = offsets[indices + 1] - offsets[indices]
        assert np.all(lengths > 0)
        tokens = np.concatenate([raw[offsets[i]:offsets[i+1]] for i in indices])
        x = torch.from_numpy(tokens.view(np.int16)).view(torch.bfloat16).float()
        scores = model.net(x).ravel()
        assert torch.isfinite(scores).all()
        for i, vector in enumerate(scores.split(lengths.tolist()), start):
            n = vector.numel()
            pooled = float(torch.topk(vector, int(n * .1) + 1).values.mean())
            max_residual = max(max_residual, abs(pooled - float(expected[i])))
            winners = torch.where(vector == vector.max())[0].numpy()
            tied += len(winners) > 1
            bins += np.bincount((np.minimum(9, winners * 10 // (n - 1)) if n > 1 else np.array([5])), minlength=10) / len(winners)
assert max_residual < 1e-4 and np.isclose(bins.sum(), len(chosen))
receipt = dict(domain=domain, layer=layer, positions=len(chosen), top_token_fraction=(bins / len(chosen)).tolist(),
               ties=tied, checkpoint_sha256=frozen['checkpoint_SHA256'], feature_sha256=pin['SHA256'],
               max_delta_from_saved_hami_val_score=max_residual, error_positions_sha256=reference['error_positions_sha256'])
out = Path(sys.argv[2])
temp = out.with_suffix('.tmp')
temp.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
temp.replace(out)
print('VERIFIED', domain, layer, len(chosen), max_residual, flush=True)
