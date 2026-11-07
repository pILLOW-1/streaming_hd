#!/usr/bin/env python3
"""Explicit SHA-256 conventions for text, JSON records, token IDs and files."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256_text(text: str) -> str:
    """Hash exact UTF-8 bytes, without whitespace or Unicode normalization."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_record(record: dict) -> str:
    """Match source-record hashing: sorted keys, Unicode, default JSON separators."""
    return sha256_text(json.dumps(record, sort_keys=True, ensure_ascii=False))


def sha256_tokens(ids: list[int]) -> str:
    """SHA-256 of contiguous signed int64 little-endian token-ID bytes."""
    return hashlib.sha256(struct.pack('<' + str(len(ids)) + 'q', *ids)).hexdigest()


def sha256_array(array) -> str:
    """Hash contiguous array value bytes, matching production pooled-row hashes."""
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='JSONL records; output one hash report per record.')
    source.add_argument('--file', type=Path, help='Hash raw file bytes.')
    source.add_argument('--text', help='Hash this exact UTF-8 string.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import sys
    output = args.output.open('x') if args.output else sys.stdout
    try:
        if args.file:
            output.write(json.dumps({'file_SHA256': sha256_file(args.file)}) + '\n')
        elif args.text is not None:
            output.write(json.dumps({'text_SHA256': sha256_text(args.text)}) + '\n')
        else:
            with args.input.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    result = {'sample_id': row.get('sample_id'), 'input_record_SHA256': sha256_record(row)}
                    if 'input_ids' in row:
                        result['token_input_SHA256'] = sha256_tokens(row['input_ids'])
                    for key, field in [('prompt', 'native_prompt_SHA256'), ('reasoning', 'reasoning_text_SHA256')]:
                        if key in row:
                            result[field] = sha256_text(row[key])
                    output.write(json.dumps(result) + '\n')
    finally:
        if args.output:
            output.close()


if __name__ == '__main__':
    main()
