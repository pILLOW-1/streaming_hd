#!/usr/bin/env python3
"""Render the exact model-native prompt formatting used by extraction preparation."""
import argparse
import json
from pathlib import Path
from hash_utils import sha256_text


def render_prompt(tokenizer, messages: list[dict], family: str) -> str:
    if family not in ['deepseek', 'qwen', 'gemma']:
        raise ValueError('Unknown model family.')
    kwargs = {'enable_thinking': True, 'reasoning_effort': 'medium'} if family == 'qwen' else {}
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, **kwargs)
    if family == 'deepseek':
        prompt += '<think>\n'
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-family', required=True, choices=['deepseek', 'qwen', 'gemma'])
    parser.add_argument('--input', required=True, type=Path, help='JSONL with original messages.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    cls = PreTrainedTokenizerFast if args.model_family == 'deepseek' else AutoTokenizer
    tokenizer = cls.from_pretrained(args.model, use_fast=True)
    with args.input.open() as source, args.output.open('x') as output:
        for line in source:
            row = json.loads(line)
            prompt = render_prompt(tokenizer, row['messages'], args.model_family)
            if 'prompt' in row and row['prompt'] != prompt:
                raise ValueError('Existing exact prompt differs from this native template.')
            row.update(prompt=prompt, native_prompt_SHA256=sha256_text(prompt),
                       model_family=args.model_family)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
