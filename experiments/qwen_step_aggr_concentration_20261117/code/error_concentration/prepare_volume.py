"""Stage one frozen RunPod VAL layer directly into Cetus job scratch."""
import hashlib
import json
from pathlib import Path
import sys

import boto3
from botocore.config import Config

domain, split, scratch, runs = sys.argv[1:]
assert split in ("train","val")
scratch, runs = Path(scratch), Path(runs)
source = runs / 'qwen_step_only_learnable_20261115/runpod/legacy_recipe' / domain / 'aggr'
selection = json.loads((source/'TEST_COMPLETE.json').read_text())['selection']
assert selection['domain'] == domain and selection['arm'] == 'aggr'
layer = selection['layer']
c = boto3.Session(profile_name='runpod-qwen').client(
    's3', endpoint_url='https://s3api-eu-ro-1.runpod.io', region_name='eu-ro-1',
    config=Config(connect_timeout=12, read_timeout=90, retries={'max_attempts': 2}))
bucket = 'fcsu28qkr3'
root = f'qwen_relation/{domain}/{split}/'
frozen = selection[split]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def data(key):
    with c.get_object(Bucket=bucket, Key=key)['Body'] as stream:
        return stream.read()

def pinned(key, expected):
    raw = data(key)
    assert sha(raw) == expected, f'SHA mismatch: {key}'
    return json.loads(raw)

receipt = pinned(root+'TOKEN_WRITE_RECEIPT.json', frozen['receipt'])
offsets = pinned(root+'TOKEN_OFFSETS.json', frozen['offsets'])
label_file = pinned(root+'labels.b45b2fd.json', frozen['labels'])
assert receipt['domain'] == label_file['domain'] == domain
assert receipt['split'] == label_file['split'] == split
assert len(offsets) == len(label_file['rows']) == receipt['rows']
offset = {row['sample_id']: row for row in offsets}
assert len(offset) == len(offsets) and set(offset) == {x['sample_id'] for x in label_file['rows']}
keys, spans, labels = [], [], []
for row in label_file['rows']:
    sample = row['sample_id']
    entry = offset[sample]
    origin, stop = entry['spans'][0]
    if 'token_sha' in row:
        assert entry['token_SHA256'] == row['token_sha'] and origin == row['origin']
    steps = row['retained_original_step_ids']
    step_spans = row['step_token_spans']
    target = row["step_hallucination"]
    assert len(step_spans) == len(steps) == len(target)
    assert all(x in (0, 1) for x in target)
    for step, (lo, hi), y in zip(steps, step_spans, target):
        assert origin <= lo < hi <= stop
        keys.append(f'{sample}:{step}')
        labels.append(y)
        spans.append([entry['offset'] + lo - origin, entry['offset'] + hi - origin])
assert len(set(keys)) == len(keys) and sha(json.dumps(keys).encode()) == frozen['positions']

weight_key = ('qwen_relation/results/qwen_step_only_learnable_20261115/legacy_recipe/'
              f'{domain}/aggr/upper/{selection["checkpoint"]}')
weight = data(weight_key)
assert sha(weight) == selection['checkpoint_sha256']
checkpoint = scratch/'checkpoint.pt'
checkpoint.write_bytes(weight)

layer_name = f'layer_{layer:02d}/reasoning_answer.bf16.npy'
key = root + layer_name
head = c.head_object(Bucket=bucket, Key=key)
pin = receipt['files'][layer_name]
assert pin['bytes'] == head['ContentLength'] and pin['layer'] == layer
feature = scratch/'val.npy'
h = hashlib.sha256()
count = 0
with c.get_object(Bucket=bucket, Key=key, IfMatch=head['ETag'])['Body'] as stream, feature.open('wb') as f:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        h.update(block)
        f.write(block)
        count += len(block)
assert count == pin['bytes']
after = c.head_object(Bucket=bucket, Key=key)
assert (after['ETag'], after['ContentLength'], after['LastModified']) == (
    head['ETag'], head['ContentLength'], head['LastModified'])
all_count = len(spans)
positive_keys = [key for key, y in zip(keys, labels) if y == 1]
spans = [span for span, y in zip(spans, labels) if y == 1]
spec = dict(domain=domain, split=split, layer=layer, epoch=selection['epoch'],
            target_step_error=1, all_positions=all_count,
            error_positions_sha256=sha(json.dumps(positive_keys).encode()),
            positions=len(spans), spans=spans, feature=str(feature), feature_sha256=h.hexdigest(),
            source_key=key, source_etag=head['ETag'], source_bytes=count,
            feature_receipt_sha256=frozen['receipt'], positions_sha256=frozen['positions'],
            checkpoint=str(checkpoint), checkpoint_sha256=selection['checkpoint_sha256'])
(scratch/'input.json').write_text(json.dumps(spec)+'\n')
