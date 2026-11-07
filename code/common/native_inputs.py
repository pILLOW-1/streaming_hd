"""Identity gates for exact generated-answer feature inputs."""
import hashlib
import json
from pathlib import Path
import struct


def verify_snapshot(root, model_path, model):
    pins = json.loads((Path(root) / 'configs/model_revisions.json').read_text())['models'][model]
    for name, info in pins['snapshot_files'].items():
        if hashlib.sha256((Path(model_path) / name).read_bytes()).hexdigest() != info['SHA256']:
            raise ValueError('Native model/tokenizer snapshot identity mismatch: ' + name)


def validate_full_input(native, row):
    for key in ('sample_id', 'original_steps', 'answer'):
        if native[key] != row[key]:
            raise ValueError('Full native input identity mismatch: ' + key)
    ids = native['input_ids']
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('Invalid native input IDs')
    digest = hashlib.sha256(struct.pack('<' + str(len(ids)) + 'q', *ids)).hexdigest()
    if native['token_input_SHA256'] != digest:
        raise ValueError('Native full input token hash mismatch')
    spans = native['step_token_spans']
    if not spans or len(spans) != len(row['original_steps']):
        raise ValueError('Full input requires every original Step span')
    if not 0 <= native['reasoning_start_token'] <= spans[0][0]:
        raise ValueError('Invalid reasoning start')
    previous = 0
    for start, end in spans:
        if not 0 <= start < end <= len(ids) or end < previous:
            raise ValueError('Invalid original Step span')
        previous = end
    start, end = native['answer_token_span']
    if not spans[-1][1] <= start < end == len(ids):
        raise ValueError('Full native input must end at the generated Answer')
