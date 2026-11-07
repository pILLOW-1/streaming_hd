#!/usr/bin/env python3
"""Extract SAPLMA/Global Mean Answer features from full native generated inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--layers', required=True)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import numpy as np
    from common.dataset import load_cell, read_jsonl
    from common.native_backend import NativeBackend
    from common.native_inputs import validate_full_input, verify_snapshot
    paths = json.loads(args.paths.read_text())
    for key in ('native_full_input_root', 'baseline_feature_root'):
        if not paths.get(key):
            parser.error('Set ' + key)
    model_path = paths['model_paths'][args.model]
    verify_snapshot(ROOT, model_path, args.model)
    rows = load_cell(paths, args.model, args.domain)
    backend = NativeBackend(model_path, dtype=args.dtype, device=args.device)
    layers = list(range(1, backend.text_config.num_hidden_layers + 1)) if args.layers == 'all' else list(map(int, args.layers.split(',')))
    if not layers or len(set(layers)) != len(layers) or min(layers) < 1 or max(layers) > backend.text_config.num_hidden_layers:
        raise ValueError('Use unique native hidden_states indices 1..L')
    destinations = {}
    for method in ('saplma', 'global_mean'):
        for layer in layers:
            dest = Path(paths['baseline_feature_root']) / method / args.model / args.domain / 'answer' / f'layer_{layer}'
            if dest.exists():
                raise FileExistsError(dest)
            destinations[method, layer] = dest
    for dest in destinations.values():
        dest.mkdir(parents=True)
    for split, records in rows.items():
        source = Path(paths['native_full_input_root']) / args.model / args.domain / (split + '.native.jsonl')
        native = {row['sample_id']: row for row in read_jsonl(source)}
        if native.keys() != {row['sample_id'] for row in records}:
            raise ValueError('Native input coverage differs')
        vectors = {key: [] for key in destinations}
        for row in records:
            n = native[row['sample_id']]
            validate_full_input(n, row)
            states, _ = backend.forward(n['input_ids'], attention=False)
            for layer in layers:
                vectors['saplma', layer].append(states[layer][-1].float().cpu().numpy())
                vectors['global_mean', layer].append(states[layer][n['reasoning_start_token']:].double().mean(0).float().cpu().numpy())
            del states
        for (method, layer), dest in destinations.items():
            target = dest / (split + '.npz')
            np.savez(target, x=np.stack(vectors[method, layer]), position_id=np.asarray([r['sample_id'] + ':answer' for r in records]))
            receipt = dict(method=method, level='answer', layer=layer, dtype=args.dtype,
                           SHA256=hashlib.sha256(target.read_bytes()).hexdigest(),
                           native_input_SHA256=hashlib.sha256(source.read_bytes()).hexdigest())
            target.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
