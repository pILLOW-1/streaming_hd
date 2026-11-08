# ARS

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | answer | 20 | 68.35 | 83.00 | 7.86 | 70.50 | 82.39 |
| qwen3_8_27b | Factuality | answer | 32 | 89.84 | 89.68 | 60.00 | 72.91 | 73.43 |
| qwen3_8_27b | Logic | answer | 37 | 87.03 | 44.29 | 4.35 | 81.00 | 50.00 |

Detector weights are not included in this incremental release.
