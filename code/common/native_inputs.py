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


def exact_decoded_text(tokenizer, native):
    """Recover saved original Unicode spellings only when both frozen hashes pass.

    Tokenizer NFC normalization can make decode(ids) differ from the original
    text even though the token sequence is reversible. Saved Step/Answer text
    supplies the spellings; the original full-text checksum is the authority.
    """
    ids=native['input_ids']
    text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    expected=native.get('full_text_SHA256')
    def digest(value):return hashlib.sha256(value.encode()).hexdigest()
    if expected and digest(text)!=expected:
        normalizer=tokenizer.backend_tokenizer.normalizer
        if normalizer is None:raise ValueError('Exact source text unavailable without a declared normalization')
        for chunk in [native.get('question',''),native.get('options_or_context','')]:
            if not chunk:continue
            normalized=normalizer.normalize_str(chunk)
            if normalized!=chunk and normalized in text:text=text.replace(normalized,chunk,1)
        approximate=len(tokenizer.decode(ids[:native['step_token_spans'][0][0]],skip_special_tokens=False,clean_up_tokenization_spaces=False))
        cursor=max(0,approximate-128);parts=[]
        for step in native['original_steps']:
            normalized=normalizer.normalize_str(step);start=text.find(normalized,cursor)
            if start<cursor:raise ValueError('Cannot locate a normalized original Step')
            parts.extend([text[:start] if not parts else text[cursor:start],step]);cursor=start+len(normalized)
        parts.append(text[cursor:]);text=''.join(parts)
        answer=native.get('answer','');normalized=normalizer.normalize_str(answer)
        if answer and normalized!=answer and text.endswith(normalized):text=text[:-len(normalized)]+answer
        if digest(text)!=expected:raise ValueError('Recovered text does not match frozen full-text checksum')
    if tokenizer(text,add_special_tokens=False)['input_ids']!=ids:
        raise ValueError('Recovered text changed frozen native token IDs')
    return text
