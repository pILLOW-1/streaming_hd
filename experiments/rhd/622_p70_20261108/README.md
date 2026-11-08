# RHD

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | answer | 28 | 61.35 | 76.66 | 2.86 | 70.00 | 82.35 |
| deepseek_r1_distill_llama_8b | Logic | answer | 28 | 45.81 | 39.99 | 5.63 | 34.50 | 51.30 |
| deepseek_r1_distill_llama_8b | Multi-hop | answer | 28 | 54.20 | 69.65 | 3.40 | 64.19 | 78.19 |
| gemma_4_31b_it | Logic | answer | 53 | 41.88 | 12.82 | 0.00 | 16.50 | 25.78 |
| gemma_4_31b_it | Science | answer | 60 | 58.73 | 37.81 | 3.23 | 42.11 | 47.16 |
| qwen3_8_27b | Factuality | answer | 56 | 72.60 | 62.57 | 2.35 | 51.23 | 62.07 |
| qwen3_8_27b | Logic | answer | 64 | 64.63 | 18.83 | 0.00 | 68.00 | 25.58 |

Detector weights are not included in this incremental release.
