"""Bind a published Cetus detector to the corresponding VAL range in its source bank."""
import hashlib
import json
from pathlib import Path
import socket
import sys

domain, split, output = sys.argv[1:]
assert split in ("train","val")
home = Path.home()
runs = home/'agent_uncertainty_quantification/experiments/streaming_hd/runs'
paths = {
    'Science': 'qwen_step_only_legacy_science_20261116',
    'Multi-hop': 'qwen_step_only_legacy_aggr_extra_20261116/Multi-hop',
}
run = runs/paths[domain]
published = home/'streaming_hd/experiments/qwen_step_only_learnable_20261115/result/legacy_recipe'/domain/'aggr.json'
selected = json.loads(published.read_text())
index_bytes = (run/'reports/INPUT.json').read_bytes()
assert hashlib.sha256(index_bytes).hexdigest() == selected['input_sha256']
index = json.loads(index_bytes)
hami = index['hami']
assert hami['identity']['domain'] == domain
assert socket.gethostname().split('.')[0] == hami['host'].split('.')[0]

def lengths(split):
    for item in index['splits'][split]['rows']:
        assert len(item['positions']) == len(item['spans'])
        for lo, hi in item['spans']:
            assert hi > lo
            yield hi-lo

origin = sum(lengths('train')) if split == 'val' else 0
val, positive_keys = [], []
all_count = 0
for row in index['splits'][split]['rows']:
    assert len(row['spans']) == len(row['labels']) == len(row['positions'])
    for (lo, hi), label, key in zip(row['spans'], row['labels'], row['positions']):
        assert label in (0, 1) and hi > lo
        n = hi - lo
        if label == 1:
            val.append([origin, origin+n])
            positive_keys.append(key)
        origin += n
        all_count += 1
assert all_count == hami['identity']['ranges'][split][1] - hami['identity']['ranges'][split][0]
layer = selected['layer']
relative = f'layer_{layer}/hami_tokens.npy'
feature = Path(hami['feature_root'])/relative
pin = hami['files'][relative]
assert feature.stat().st_size == pin['bytes']
checkpoint = run/'arms/aggr'/f'best_{layer:02d}.pt'
spec = dict(domain=domain, split=split, layer=layer, epoch=selected['epoch'],
            target_step_error=1, all_positions=all_count,
            error_positions_sha256=hashlib.sha256(json.dumps(positive_keys).encode()).hexdigest(),
            positions=len(val), spans=val, feature=str(feature), feature_sha256=pin['SHA256'],
            checkpoint=str(checkpoint), checkpoint_sha256=selected['checkpoint_sha256'],
            input_sha256=selected['input_sha256'])
Path(output).write_text(json.dumps(spec)+'\n')
