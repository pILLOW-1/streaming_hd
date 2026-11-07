# Data

Dataset link: [pillow-streaming/streaming-hd-v1](https://huggingface.co/datasets/pillow-streaming/streaming-hd-v1).

The agreed local dataset format and recorded experiment assignment are authoritative. The Hugging Face repository can be updated independently; these programs do not download it or depend on its current contents. This directory contains documentation only.

## Local layout

Set `dataset_root` in your private path configuration to a directory with:

```text
<model>/<domain>/<train|val|test>/
  generation/records.jsonl
  annotations/labels.jsonl
```

Model keys are `deepseek_r1_distill_llama_8b`, `qwen3_8_27b`, and `gemma_4_31b_it`. Domain keys are `Math`, `Logic`, `Science`, `Multi-hop`, and `Factuality`.

Generation records contain `sample_id`, `model`, `domain`, `canonical_question_id`, `question_family_id`, `question`, `options_or_context`, `answer`, `original_steps`, and `retained_original_step_ids`. The original reasoning Steps remain in their original order; retained indices are zero-based indices into `original_steps`.

Annotation records use the same identity fields and a nested `labels` object:

```json
{"step": [0, 1, 0], "prefix": [0, 1, 0], "answer": 0}
```

This synthetic example illustrates the three label fields, not an actual dataset record. Step and Prefix lists align with the retained original Step indices. Answer is one independent binary label. One means hallucination and zero means no hallucination. **Do not compute Prefix labels by cumulatively OR-ing Step labels.**

The loader joins generation and annotation records by `sample_id`, checks complete label coverage, and rejects sample IDs or question families that cross splits within a model/domain. It never rewrites the source data or reassigns a split.

For the archived experiment, `experiments/streaming_622_p70/split_assignments.jsonl` records the exact IDs and ordering. Its TEST target is p≥70%, with the documented cell-specific relaxations. Archived detector evaluation requires this assignment. Use `dataset_roots[model][domain]` to point a cell at a separate authoritative root when necessary.

## Feature inputs

Feature extraction additionally takes frozen native inputs from the separately configured `native_input_root`:

```text
<model>/<domain>/<train|val|test>.native.jsonl
```

Each native record binds `sample_id`, full prompt-plus-reasoning `input_ids`, `reasoning_start_token`, retained `step_token_spans` (half-open), `retained_original_step_ids`, and token/text hashes. Exact model/tokenizer revisions are recorded in `configs/model_revisions.json`. Genuine token-boundary corrections are stored in `step_causal_overrides`. Do not infer the reasoning start from the first retained Step when earlier Steps were excluded from supervision.

Gold answers, future Steps, and final Answer text must not enter Step/Prefix features. Original excluded Steps remain in the causal context. Streaming Answer uses the final Prefix score against the independent Answer label.
