#!/usr/bin/env python3
"""Tokenize an exact saved prompt/reasoning/Answer into full native sidecars."""
import argparse
import json
from pathlib import Path
from hash_utils import sha256_tokens, sha256_text, sha256_record


def prepare_full_record(record, tokenizer):
    # A separator is required, including an explicit empty separator. Never infer
    # chat delimiters, reconstruct reasoning with joins, or insert reference text.
    prompt, reasoning, separator, answer = (record[k] for k in ('prompt', 'reasoning', 'answer_separator', 'answer'))
    if not all(isinstance(v, str) for v in (prompt, reasoning, separator, answer)) or not answer:
        raise ValueError('Exact saved text fields and a nonempty generated Answer required')
    text = prompt + reasoning + separator + answer
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc['offset_mapping']
    def tokens(start, end):
        indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError('Noncontiguous or empty native span')
        return [indices[0], indices[-1] + 1]
    spans = []
    cursor = 0
    for step in record['original_steps']:
        if not step:
            raise ValueError('Empty original Step')
        start = reasoning.find(step, cursor)
        if start < 0:
            raise ValueError('Original Steps do not match saved reasoning')
        cursor = start + len(step)
        spans.append(tokens(len(prompt) + start, len(prompt) + cursor))
    if not spans:
        raise ValueError('No original reasoning Steps')
    answer_span = tokens(len(prompt) + len(reasoning) + len(separator), len(text))
    if spans[-1][1] > answer_span[0] or answer_span[1] != len(enc['input_ids']):
        raise ValueError('Ambiguous reasoning/Answer boundary or trailing native tokens')
    result = {k: record[k] for k in ('sample_id', 'original_steps', 'answer')}
    result.update(input_ids=enc['input_ids'], step_token_spans=spans,
                  reasoning_start_token=tokens(len(prompt), len(prompt) + len(reasoning))[0],
                  answer_token_span=answer_span, token_input_SHA256=sha256_tokens(enc['input_ids']),
                  full_text_SHA256=sha256_text(text), source_record_SHA256=sha256_record(record))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.input.open() as source, a.output.open('x') as output:
        for line in source:
            if line.strip():
                output.write(json.dumps(prepare_full_record(json.loads(line), tokenizer), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
