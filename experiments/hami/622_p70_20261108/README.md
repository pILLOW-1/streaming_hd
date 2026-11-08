# HaMI (ori)

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | step | 8 | 82.91 | 52.16 | 14.29 | 72.97 | 48.89 |
| gemma_4_31b_it | Factuality | step | 40 | 79.46 | 21.42 | 12.10 | 93.98 | 25.43 |
| qwen3_8_27b | Factuality | step | 46 | 87.35 | 41.74 | 13.99 | 85.32 | 46.90 |

Detector weights are not included in this incremental release.
