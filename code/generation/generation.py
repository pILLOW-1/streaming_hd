#!/usr/bin/env python3
"""Native BF16 whole-trajectory hidden states and Streaming pooled features."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def layer_list(value: str, count: int) -> list[int]:
    if value == 'all':
        return list(range(1, count + 1))
    layers = sorted({int(i) for i in value.split(',')})
    if not layers or layers[0] < 1 or layers[-1] > count:
        raise ValueError('Layers must be native hidden_states indices1..L;0 is the embedding output.')
    return layers


def validate_row(row: dict) -> None:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_id']):
        raise ValueError('sample_id must be a safe single filename component.')
    ids, spans, retained = row['input_ids'], row['step_token_spans'], row['retained_original_step_ids']
    if not ids or not all(type(i) is int and i >= 0 for i in ids):
        raise ValueError('input_ids must be nonempty integer token IDs.')
    if not spans or len(spans) != len(retained):
        raise ValueError('Every retained Step must have exactly one token span.')
    if not all(type(i) is int and i >= 0 for i in retained) or retained != sorted(set(retained)):
        raise ValueError('Retained Step IDs must be unique and increasing.')
    if not 0 <= row['reasoning_start_token'] <= spans[0][0]:
        raise ValueError('Invalid reasoning start; it must exclude the prompt.')
    previous_end = 0
    for start, end in spans:
        if not 0 <= start < end <= len(ids) or end < previous_end:
            raise ValueError('Invalid or out-of-order Step span.')
        previous_end = end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Model directory or supported Hub ID.')
    parser.add_argument('--inputs', required=True, type=Path, help='Frozen native-input JSONL.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layers', default='all', help='all, or comma-separated native layer indices.')
    parser.add_argument('--pool-only', action='store_true', help='Save Step/Prefix pools without raw token tensors.')
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from feature_pooling import BothPools
    from hash_utils import sha256_record, sha256_tokens, sha256_file, sha256_array
    from token_count import count_native
    if not torch.cuda.is_available():
        raise RuntimeError('Run this script inside your GPU allocation.')
    config = transformers.AutoConfig.from_pretrained(args.model)
    text_config = getattr(config, 'text_config', config)
    count, width = text_config.num_hidden_layers, text_config.hidden_size
    layers = layer_list(args.layers, count)
    torch.backends.cuda.matmul.allow_tf32 = False

    class ExportingPools(BothPools):
        export = None

        def _hook(self, layer):
            original = super()._hook(layer)

            def hook(module, arguments, output):
                if self.export is not None and layer in self.active:
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    root, row, name = self.export
                    start, end = row['reasoning_start_token'], row['step_token_spans'][-1][1]
                    path = root / f'{name}layer_{layer}.pt'
                    temp = path.with_suffix('.tmp')
                    torch.save({'hidden_states': hidden[0, start:end].detach().cpu(),
                                'layer': layer, 'token_start': start, 'token_end': end,
                                'input_ids': row['input_ids'],
                                'step_token_spans': row['step_token_spans'],
                                'retained_original_step_ids': row['retained_original_step_ids']}, temp)
                    temp.replace(path)
                return original(module, arguments, output)
            return hook

    engine = ExportingPools(args.model, count, width)
    checked = False
    try:
        seen = set()
        with args.inputs.open() as source:
            for line in source:
                row = json.loads(line)
                validate_row(row)
                if 'token_input_SHA256' in row and row['token_input_SHA256'] != sha256_tokens(row['input_ids']):
                    raise ValueError('Frozen token-ID hash mismatch.')
                sid = row['sample_id']
                if sid in seen:
                    raise ValueError('Duplicate sample_id in input stream.')
                seen.add(sid)
                target = args.output / sid
                target.mkdir(parents=True, exist_ok=False)
                if not checked:
                    # Native API identity at identical input, then fixed-shape causal check.
                    probe = row['input_ids'][:256]
                    if len(probe) < 4:
                        raise ValueError('At least4input tokens required for the numerical gate.')
                    cut = len(probe) // 2
                    engine.run(probe, layers, raw_span=(cut - 1, cut), api=True)
                    _, before = engine.run(probe, layers, raw_span=(cut - 1, cut))
                    changed = probe.copy()
                    changed[cut:] = [1] * (len(probe) - cut)
                    _, after = engine.run(changed, layers, raw_span=(cut - 1, cut))
                    if not all(torch.equal(before[l], after[l]) for l in layers):
                        raise RuntimeError('Native fixed-shape causal identity gate failed.')
                    checked = True
                engine.local_layers = engine.global_layers = set(layers)
                engine.export = None if args.pool_only else (target, row, '')
                gp, lp = engine.both(row, layers, row['reasoning_start_token'])
                corrected = []
                for position, override in row.get('step_causal_overrides', {}).items():
                    position = int(position)
                    if not 0 <= position < len(row['step_token_spans']):
                        raise ValueError('Invalid causal override position.')
                    start = row['reasoning_start_token']
                    if override['input_ids'][:start] != row['input_ids'][:start]:
                        raise ValueError('Causal override changed the prompt.')
                    causal = dict(row, input_ids=override['input_ids'],
                                  step_token_spans=[override['span']],
                                  retained_original_step_ids=[row['retained_original_step_ids'][position]])
                    validate_row(causal)
                    engine.export = None if args.pool_only else (target, causal, f'override_{position}_')
                    cg, cl = engine.both(causal, layers, start)
                    for l in layers:
                        gp[l][position], lp[l][position] = cg[l][0], cl[l][0]
                    corrected.append(position)
                for l in layers:
                    for kind, value in [('local_exp', lp[l]), ('global_exp', gp[l])]:
                        if value.dtype != np.float32 or value.shape != (len(row['step_token_spans']), width) or not np.isfinite(value).all():
                            raise RuntimeError('Invalid pooled features.')
                        np.save(target / f'{kind}.layer_{l}.npy', value)
                output_hashes = {f.name: {'SHA256': sha256_file(f), 'bytes': f.stat().st_size}
                                 for f in sorted(target.iterdir()) if f.suffix in ['.pt', '.npy']}
                array_hashes = {str(l): {'local_exp': sha256_array(lp[l]), 'global_exp': sha256_array(gp[l])}
                                for l in layers}
                result = {'sample_id': sid, 'model': args.model, 'layers': layers,
                          'backbone_dtype': 'bfloat16', 'pooled_dtype': 'float32',
                          'global_exp_accumulator_dtype': 'float64',
                          'token_input_SHA256': sha256_tokens(row['input_ids']),
                          'native_input_record_SHA256': sha256_record(row),
                          'token_counts': count_native(row),
                          'native_prompt_SHA256': row.get('native_prompt_SHA256'),
                          'full_text_SHA256': row.get('full_text_SHA256'),
                          'files': output_hashes, 'pooled_array_SHA256': array_hashes,
                          'retained_original_step_ids': row['retained_original_step_ids'],
                          'native_API_identity': True, 'fixed_shape_causal_identity': True,
                          'causal_override_positions': corrected,
                          'torch': torch.__version__, 'transformers': transformers.__version__}
                (target / 'MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
                engine.export = None
                print(json.dumps({'sample_id': sid, 'status': 'COMPLETE'}), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
