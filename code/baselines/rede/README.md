# ReDe + supervised probing

Source: [Reasoning Denoiser, arXiv:2607.22098v1](https://arxiv.org/html/2607.22098v1), equations 1–3, algorithms 2–3 and Appendix A.4. This is a paper-derived implementation; no official implementation revision is asserted.

```bash
python code/baselines/rede/run.py --paths configs/paths.local.json \
  --recipe configs/baselines/rede.answer.json \
  --model deepseek_r1_distill_llama_8b --domain Math
```

The pipeline fits the projection on TRAIN, filters reasoning Steps, re-forwards the filtered question/reasoning/generated-Answer sequence, trains the downstream probe, selects on VAL, and evaluates TEST after freezing. Filtering is not replaced by averaging stale pre-filter states.

Set `native_full_input_root` to an auxiliary sidecar directory with `<model>/<domain>/<split>.native.jsonl`. Each record binds `sample_id`, exact full `input_ids` through the generated Answer, `reasoning_start_token`, token spans for **all original** reasoning Steps, `original_steps`, `answer`, `token_input_SHA256`, and `answer_token_span`. These are feature-construction inputs, not a replacement dataset format. Generated Answer text must match the data; gold/reference answers never enter the backbone.

Install the pinned ARS probe dependency following `THIRD_PARTY.md` before running ReDe. The fixed paper optimizer/probe settings are implemented. Projection widths, epochs, candidate grid, ratio rounding and short-trace behavior were not fully fixed by the paper. The example recipe explicitly supplies project defaults; it is not advertised as the authors' exact selected configuration. Expand its candidates before running, using TRAIN/VAL only. The supplied short-trace extension keeps all Steps when k=15 is undefined; choose `error` to enforce strict rejection instead. Traces with fewer than two informative/noisy proxy Steps do not contribute an undefined projection loss but remain in downstream data.

Text filtering removes exact Step substrings and retains original separators, prompt and Answer, then retokenizes. A tokenizer round-trip check must pass first. The native vocabulary-head check must also pass; no incompatible model-head semantics are silently accepted.
