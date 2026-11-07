# Current research state

## Research objective

Study hallucination detection across individual reasoning Steps, cumulative Prefixes and final Answers, including how errors persist and are corrected. Establish baseline behavior on the retained dataset before finalizing a proposed method.

## Settled decisions

- Retain the three-generator, five-domain dataset and the selected **6:2:2, TEST p ≥ 70% target** assignment. The target has recorded per-cell relaxations; the display name does not mean every cell achieves 70%.
- `p` denotes the fraction of TEST trajectories with at least two Prefix 1→0 transitions. Original Step positions, independent three-level labels and within-model split separation are preserved.
- Fit each generator/domain independently. Select detector layer and weights using complete own VAL AUROC, then VAL AP. No preference for smaller layers or earlier updates resolves an exact tie.
- Use own-level VAL F1max thresholds, with precision and largest-threshold tie rules and strict score > threshold. Report AUROC, AP and empirical TPR@1%FPR as primary metrics; ACC/F1 are secondary.
- Use native whole-trajectory hidden states, retaining genuine causal token-boundary corrections. Reuse accepted feature caches.
- Streaming Answer scores are the final retained Prefix scores of the frozen Prefix detector, evaluated against independent Answer labels.
- Publish only the selected Streaming reproduction in `experiments/`. Earlier experiments remain part of the research history, not current accepted results for this split.
- **The proposed method is not finalized. `code/ours/` contains only a README.** Previous learnable-aggregation, representation, relation and joint-training trials do not constitute a selected method.
- Theory publication is deferred. Data format is defined locally; Hugging Face is an access link and may be updated separately.

## Current evidence

[The selected experiment](../experiments/streaming_622_p70/README.md) binds the split IDs, data hashes, metrics and detector weights. [The result table](../experiments/streaming_622_p70/result/RESULTS.md) reports per-domain Step, Prefix and Answer performance.

The split was chosen after examining alternative splits and outcomes. It is a deliberate evaluation design, not an untouched random estimate of the source population. Historical trial results must not be merged into the selected result table.

[VALIDATION.json](VALIDATION.json) describes repository checks and their limits. Synthetic CPU tests and dependency checks do not certify every native GPU backend or constitute new method benchmarks. [VALIDATION_INITIAL_RELEASE.json](VALIDATION_INITIAL_RELEASE.json) describes the earlier release before the user deferred Ours; its tests are historical.

## Next work

1. Obtain the agreed local dataset and exact backbone/tokenizer snapshots; configure private paths.
2. Verify assignment and label hashes when reproducing archived detector results.
3. Reuse native features where identity and coverage match. Install separately obtained upstream dependencies only under their author terms.
4. Run authorized baseline experiments, record their full protocols and evidence, and evaluate on TEST only after VAL selection is frozen.
5. Discuss and finalize the proposed method before introducing implementation under `code/ours/`.

Old experiment deletion remains pending approval. Do not delete original data, models, the selected split, or reusable feature/native-input sources merely because an old path name appears in the trace.
