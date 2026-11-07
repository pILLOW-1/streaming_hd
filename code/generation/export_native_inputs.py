#!/usr/bin/env python3
"""Export hash-verified native model inputs from a local JSONL sidecar."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def native_record(row: dict) -> dict:
    ids = row['input_ids']
    actual = hashlib.sha256(b''.join(struct.pack('<q', value) for value in ids)).hexdigest()
    if actual != row['token_input_SHA256']:
        raise ValueError('Frozen input token hash mismatch: ' + row['sample_id'])
    spans = row['step_token_spans']
    if len(spans) != len(row['retained_original_step_ids']):
        raise ValueError('Retained Step/span count mismatch')
    if not 0 <= row['reasoning_start_token'] <= spans[0][0] < len(ids):
        raise ValueError('Invalid reasoning start')
    for start, end in spans:
        if not 0 <= start < end <= len(ids):
            raise ValueError('Invalid retained Step token span')
    result = {key: row[key] for key in [
        'sample_id', 'input_ids', 'reasoning_start_token', 'step_token_spans',
        'retained_original_step_ids', 'native_prompt_SHA256', 'full_text_SHA256',
        'token_input_SHA256', 'model_repo_id', 'model_revision',
        'tokenizer_repo_id', 'tokenizer_revision',
    ]}
    overrides = json.loads(row['step_causal_overrides_json'])
    if overrides:
        result['step_causal_overrides'] = overrides
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Local records with frozen native token fields.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as source:
        data = [json.loads(line) for line in source if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for row in data:
            output.write(json.dumps(native_record(row), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
