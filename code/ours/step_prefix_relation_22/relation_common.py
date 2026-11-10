"""Frozen Step--Prefix experiment interfaces; no data resplitting or label repair."""
import hashlib
import json
import os
from pathlib import Path

IDENTITY = 'step_prefix_relation_22_qwen_20261110'
PROJECT = Path('/shared/homes/u26487468/agent_uncertainty_quantification')
WORK = PROJECT / 'workspaces/streaming_hd/ours/step_prefix_relation_22'
STATE = PROJECT / 'state/streaming_hd' / IDENTITY
RUN = PROJECT / 'experiments/streaming_hd/runs' / IDENTITY
REPO = Path('/shared/homes/u26487468/streaming_hd')
DATA = PROJECT / 'assets/streaming_hd_new'
BASE = PROJECT / 'experiments/streaming_hd/runs/baselines_622_p70_20261108'
SCRATCH = Path('/scratch/u26487468/agent_uncertainty_quantification') / IDENTITY
READOUTS = ('last_token', 'global_mean', 'step_exp', 'prefix_exp')
DOMAINS = ('Factuality', 'Logic', 'Multi-hop', 'Science', 'Math')
MODEL = 'qwen3_8_27b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), f.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def config():
    return json.loads((WORK / 'config.json').read_text())


def dataset(domain, splits):
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / 'code'))
    from common.dataset import load_split
    # This module is loaded as relation_common to avoid shadowing common.dataset.
    result = {}
    for split in splits:
        rows = load_split(DATA, MODEL, domain, split)
        keys, ys, cs = [], [], []
        for row in rows:
            for step, y, c in zip(row['retained_original_step_ids'], row['step_hallucination'], row['prefix_hallucination']):
                keys.append(row['sample_id'] + ':' + str(step)); ys.append(y); cs.append(c)
        y, c = np.asarray(ys, dtype=np.int64), np.asarray(cs, dtype=np.int64)
        assert len(keys) == len(set(keys)) and len(keys) > 0
        assert not np.any(y > c), 'Conflicting labels: stop; never rewrite or drop positions'
        assert set(y) == set(c) == {0, 1}, 'Both task classes required for VAL AUROC selection'
        result[split] = dict(rows=rows, keys=keys, y=y, c=c, k=np.where(y == 1, 0, np.where(c == 1, 1, 2)))
    seen = set()
    for split, item in result.items():
        families = {row['question_family_id'] for row in item['rows']}
        assert not seen.intersection(families), 'Question family crosses splits'
        seen.update(families)
    return result


def data_hashes(domain, splits=('train', 'val', 'test')):
    return {split + '/' + name: sha(DATA / MODEL / domain / split / name)
            for split in splits for name in ('generation/records.jsonl', 'annotations/labels.jsonl')}


def source(domain, readout):
    if readout in ('last_token', 'global_mean'):
        receipt = BASE / 'sources' / MODEL / domain / 'reports/FEATURE_MANIFEST.json'
        name = 'saplma' if readout == 'last_token' else 'global_mean'
    else:
        receipt = RUN / 'reports/features' / domain / 'FEATURE_MANIFEST.json'
        name = readout
    x = json.loads(receipt.read_text())
    assert x['status'] == 'COMPLETE'
    return x, name, sha(receipt)


def arm_name(domain, readout, head, loss):
    return '.'.join((domain, readout, head, loss))


def arm_paths(arm):
    domain, readout, head, loss = arm.split('.')
    relative = Path(domain) / readout / head / loss
    return RUN / 'reports' / relative, RUN / 'checkpoints' / relative
