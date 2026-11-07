#!/usr/bin/env python3
"""Tokenize exact prompt + saved reasoning and preserve retained Step spans."""
import argparse
import hashlib
import json
from pathlib import Path
from hash_utils import sha256_text, sha256_record, sha256_tokens
from format_prompt import render_prompt
from token_count import count_native


def prepare_record(record: dict, tokenizer, model_family: str | None = None) -> dict:
    family = model_family or record.get('model_family')
    prompt = record['prompt'] if 'prompt' in record else render_prompt(tokenizer, record['messages'], family)
    reasoning = record['reasoning']
    steps, retained = record['original_steps'], record['retained_original_step_ids']
    if not retained or retained != sorted(set(retained)):
        raise ValueError('Retained Step IDs must be nonempty, unique and increasing.')
    spans, cursor = [], 0
    for step in steps:
        if not isinstance(step, str) or not step:
            raise ValueError('Every original Step must be a nonempty string.')
        start = reasoning.find(step, cursor)
        if start < 0:
            raise ValueError('Original Steps must occur in order in the exact saved reasoning.')
        cursor = start + len(step)
        spans.append((start, cursor))
    if not 0 <= retained[0] <= retained[-1] < len(steps):
        raise ValueError('Retained Step ID outside original_steps.')
    last = spans[retained[-1]][1]
    text = prompt + reasoning[:last]
    enc = tokenizer(text, add_special_tokens=False, truncation=False,
                    padding=False, return_offsets_mapping=True)
    offsets, token_spans, overrides = enc['offset_mapping'], [], {}
    for position, sid in enumerate(retained):
        start, end = (len(prompt) + x for x in spans[sid])
        indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError('Step has no contiguous native tokens.')
        token_spans.append([indices[0], indices[-1] + 1])
        if text[end:offsets[indices[-1]][1]].strip():
            before = tokenizer(text[:end], add_special_tokens=False, truncation=False,
                               padding=False, return_offsets_mapping=True)
            ids = [i for i, (a, b) in enumerate(before['offset_mapping'])
                   if a < end and b > start and b > a]
            if not ids:
                raise ValueError('Empty corrected causal Step token span.')
            overrides[str(position)] = {'input_ids': before['input_ids'],
                                        'span': [ids[0], ids[-1] + 1]}
    start = len(prompt) + spans[0][0]
    reasoning_tokens = [i for i, (a, b) in enumerate(offsets) if b > start and b > a]
    if not reasoning_tokens or reasoning_tokens[0] > token_spans[0][0]:
        raise ValueError('Cannot locate exact reasoning start.')
    result = {'sample_id': record['sample_id'], 'input_ids': enc['input_ids'],
              'step_token_spans': token_spans, 'reasoning_start_token': reasoning_tokens[0],
              'retained_original_step_ids': retained,
              'native_prompt_SHA256': hashlib.sha256(prompt.encode()).hexdigest(),
              'full_text_SHA256': hashlib.sha256(text.encode()).hexdigest()}
    if overrides:
        result['step_causal_overrides'] = overrides
    result.update(source_record_SHA256=sha256_record(record),
                  reasoning_text_SHA256=sha256_text(reasoning[:last]),
                  token_input_SHA256=sha256_tokens(enc['input_ids']))
    counts = count_native(result)
    counts.update(prompt_tokens_standalone=len(tokenizer(prompt, add_special_tokens=False)['input_ids']),
                  reasoning_tokens_standalone=len(tokenizer(reasoning[:last], add_special_tokens=False)['input_ids']),
                  prompt_tokens_in_input=sum(a < len(prompt) and b > 0 and b > a for a, b in offsets),
                  reasoning_tokens_in_input=sum(a < len(text) and b > len(prompt) and b > a for a, b in offsets),
                  shared_prompt_reasoning_boundary_tokens=sum(a < len(prompt) < b for a, b in offsets))
    result['token_counts'] = counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Model directory or Hugging Face model ID.')
    parser.add_argument('--model-family', required=True, choices=['deepseek', 'qwen', 'gemma'])
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    cls = PreTrainedTokenizerFast if args.model_family == 'deepseek' else AutoTokenizer
    tokenizer = cls.from_pretrained(args.model, use_fast=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open('x') as output:
        for line in source:
            row = prepare_record(json.loads(line), tokenizer, args.model_family)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
