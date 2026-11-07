# Overnight scorer campaign — 2026-09-08

Source experiment: `experiments/streaming_hd/runs/step_scorer_overnight_frozen_test_20260908`

- Validation ranking: `mean_context`, `linear`, `pointwise_adapter`, `vector_attention`.
- Frozen common candidate: `mean_context`.
- Frozen test inference count: 12; test-time fit count: 0; test-time selection: false.
- `mean_context` versus `S_original` across AUROC/accuracy/F1 × three models: 6 strict wins, 0 ties, 3 losses.
- Predeclared 9/9 strict-win target: not achieved.
- Final independent audit: PASS, SHA-256 `f16e7032b6b5eb20334628918c8c15045cf6981cc073547bf431588f3355ddb6`.

Grounding:

- `results/VALIDATION_SELECTION.json` records the complete validation-only ordering and selected candidate.
- `results/nine_metric_gate.json` records the frozen nine-cell comparisons and hard-goal result.
- `audits/FINAL_VALIDATION.json` records the independent NumPy64 replay, metric/bootstrap checks, artifact hashes, and no-test-reselection assertion.
