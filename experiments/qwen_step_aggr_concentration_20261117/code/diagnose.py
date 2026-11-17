"""Measure paired VAL Step concentration under learned and positional pooling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def measures(weights):
    length = len(weights)
    return (1 / (length * np.dot(weights, weights)),
            float(weights.max()), float(np.sort(weights)[-3:].sum()),
            float(weights[-((length + 4) // 5):].sum()),
            float(weights.argmax() / (length - 1)) if length > 1 else np.nan)


def quartiles(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [.25, .5, .75]).tolist() if values.size else None


@torch.inference_mode()
def diagnose(raw, spans, scorer):
    result = np.empty((len(spans), 2, 5), dtype=np.float64)
    for start in range(0, len(spans), 64):
        batch = spans[start:start + 64]
        lengths = np.diff(batch, axis=1).ravel()
        tokens = np.concatenate([raw[lo:hi] for lo, hi in batch])
        assert len(tokens) == lengths.sum()
        inputs = torch.from_numpy(tokens.view(np.int16)).view(torch.bfloat16).float()
        logits = scorer(inputs).reshape(-1)
        assert torch.isfinite(logits).all()
        for row, step_logits in enumerate(logits.split(lengths.tolist()), start):
            fixed = torch.linspace(0, 3, step_logits.numel(), dtype=torch.float32)
            for arm, scores in enumerate((step_logits, fixed)):
                weights = scores.softmax(dim=0).double().numpy()
                assert abs(weights.sum() - 1) < 1e-6
                result[row, arm] = measures(weights)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    spec = json.loads(args.input.read_text())
    assert spec['split'] == 'val'
    feature = Path(spec['feature'])
    checkpoint = Path(spec['checkpoint'])
    for path, key in ((feature, 'feature_sha256'), (checkpoint, 'checkpoint_sha256')):
        assert digest(path) == spec[key], f'SHA mismatch: {path}'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['selection']['layer'] == spec['layer']
    assert saved['selection']['epoch'] == spec['epoch']
    detector = saved['model']
    scorer = torch.nn.Sequential(torch.nn.Linear(5120,256), torch.nn.BatchNorm1d(256),
                                 torch.nn.ReLU(), torch.nn.Dropout(.3), torch.nn.Linear(256,1))
    scorer.load_state_dict({k[7:]: v for k,v in detector.items() if k.startswith('scorer.')})
    scorer.eval()
    raw = np.load(feature, mmap_mode='r', allow_pickle=False)
    spans = np.asarray(spec['spans'], dtype=np.int64)
    lengths = (spans[:,1] - spans[:,0])
    assert raw.ndim == 2 and raw.shape[1] == 5120 and raw.dtype == np.dtype('<u2')
    assert lengths.size == spec['positions'] and lengths.min() > 0
    assert spans.min() >= 0 and spans.max() <= len(raw)
    scores = diagnose(raw, spans, scorer)
    names = ('effective_ratio', 'top1_mass', 'top3_mass', 'last20_mass', 'argmax_relative')
    bins = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, int(max(257, lengths.max())))]
    summary = {}
    for label, subset in [('all', lengths > 0)] + [(f'{lo}-{hi}', (lengths >= lo) & (lengths <= hi)) for lo,hi in bins]:
        pair = scores[subset]
        summary[label] = dict(positions=len(pair), metrics={
            name: dict(learned=quartiles(pair[:,0,k]), fixed=quartiles(pair[:,1,k]),
                       paired_delta=quartiles(pair[:,0,k]-pair[:,1,k]))
            for k,name in enumerate(names)
        }, argmax_deciles={
            arm: np.histogram(pair[np.isfinite(pair[:,j,4]),j,4], bins=np.linspace(0,1,11))[0].tolist()
            for j,arm in enumerate(('learned','fixed'))
        })
    receipt = {k: spec[k] for k in ('domain','split','layer','epoch','positions','feature_sha256','checkpoint_sha256')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(receipt, summary=summary), indent=2, allow_nan=False)+'\n')
    tmp.replace(args.out)


if __name__ == '__main__':
    main()
