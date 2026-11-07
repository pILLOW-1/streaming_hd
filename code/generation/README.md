# Native inputs and features

These utilities preserve native prompt formatting, token IDs, reasoning boundaries and original Step indices. They do not regenerate answers or change the dataset split.

## Reasoning inputs

Use saved exact `prompt` and `reasoning` strings with `sample_id`, `original_steps`, and `retained_original_step_ids`. Alternatively, supply the original `messages` to render the model's chat template. Never reconstruct saved reasoning by joining retained Steps: discarded Steps remain part of the causal context.

```bash
python code/generation/prepare_inputs.py --model /your/local/model \
  --model-family deepseek --input /your/exact_saved_records.jsonl \
  --output /your/native_inputs.jsonl
```

Store one sidecar per split at `<native_input_root>/<model>/<domain>/<split>.native.jsonl`. The sidecar contains exact prompt-plus-reasoning `input_ids`, `reasoning_start_token`, retained `step_token_spans`, `retained_original_step_ids`, token counts and SHA-256 receipts. Genuine tokens crossing a future Step boundary receive explicit corrected causal input overrides. Native layer indices are `hidden_states[1..L]`.

`format_prompt.py` preserves native chat-template formatting; `hash_utils.py` hashes exact UTF-8 text, sorted-key JSON, or signed little-endian int64 token bytes. `token_count.py` counts full inputs and spans; standalone token counts need not add up because tokenization can merge across boundaries.

## Hidden states and pooled features

```bash
python code/generation/generation.py --model /your/local/model \
  --inputs /your/native_inputs.jsonl --output /your/cache/model --layers all
```

One whole-trajectory BF16 causal forward extracts all requested native layers. Raw token tensors and local/global exponential pools are saved with hashes. Add `--pool-only` when only Streaming pools are needed. The higher-level `code/tools/extract_features.py` validates and reuses existing pools. Model/tokenizer snapshot identities are recorded in `configs/model_revisions.json`; the files themselves are external.

## Full generated-Answer inputs

SAPLMA/Global Mean Answer features and ReDe need the actual generated Answer endpoint. Supply exact `prompt`, `reasoning`, `answer_separator`, `answer`, `original_steps`, and `sample_id`; `answer_separator` must reproduce the saved text, including any native channel delimiters. No reference answer is inserted.

```bash
python code/generation/prepare_full_inputs.py --model /your/local/model \
  --input /your/exact_full_records.jsonl --output /your/full_native_inputs.jsonl
```

Store these at `<native_full_input_root>/<model>/<domain>/<split>.native.jsonl`. All original Step spans, the Answer span, token hash and generated text identity are checked. Ambiguous reasoning/Answer token boundaries are rejected. The dataset's generation/annotation files are unchanged.

ICR additionally requires exact `core_positions` (`user_prompt_start`, `user_prompt_end`, `response_start`) from the original native prompt. These are token offsets, not guessed text lengths.
