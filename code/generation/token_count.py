#!/usr/bin/env python3
"""Count native input, reasoning-history, Step and Prefix tokens from frozen spans."""
import argparse
import json
from pathlib import Path
from generation import validate_row


def count_native(row: dict) -> dict:
    validate_row(row)
    start = row['reasoning_start_token']
    spans = row['step_token_spans']
    intervals = sorted(spans)
    unique, left, right = 0, intervals[0][0], intervals[0][1]
    for a, b in intervals[1:]:
        if a <= right:
            right = max(right, b)
        else:
            unique += right - left
            left, right = a, b
    unique += right - left
    return {'input_tokens': len(row['input_ids']),
            'reasoning_history_tokens': len(row['input_ids']) - start,
            'retained_steps': len(spans),
            'per_step_tokens': [b - a for a, b in spans],
            'per_prefix_history_tokens': [b - start for a, b in spans],
            'retained_step_token_sum': sum(b - a for a, b in spans),
            'retained_step_unique_tokens': unique,
            'causal_override_input_tokens': {k: len(v['input_ids']) for k, v in row.get('step_causal_overrides', {}).items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import sys
    output = args.output.open('x') if args.output else sys.stdout
    try:
        with args.input.open() as stream:
            for line in stream:
                row = json.loads(line)
                output.write(json.dumps({'sample_id': row['sample_id'], **count_native(row)}) + '\n')
    finally:
        if args.output:
            output.close()


if __name__ == '__main__':
    main()
