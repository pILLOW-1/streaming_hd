# ReDe

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | answer | 27 | 72.85 | 84.45 | 7.14 | 72.50 | 83.08 |
| deepseek_r1_distill_llama_8b | Logic | answer | 16 | 82.19 | 65.88 | 5.63 | 72.50 | 69.27 |
| deepseek_r1_distill_llama_8b | Math | answer | 20 | 90.09 | 86.12 | 0.00 | 80.00 | 82.85 |
| deepseek_r1_distill_llama_8b | Multi-hop | answer | 31 | 79.51 | 87.26 | 11.56 | 72.93 | 81.44 |
| gemma_4_31b_it | Factuality | answer | 41 | 87.13 | 79.68 | 37.29 | 84.50 | 66.67 |
| gemma_4_31b_it | Logic | answer | 34 | 94.71 | 84.52 | 63.33 | 92.50 | 75.41 |
| gemma_4_31b_it | Math | answer | 59 | 86.85 | 71.45 | 3.80 | 84.00 | 82.02 |
| gemma_4_31b_it | Multi-hop | answer | 24 | 91.61 | 78.11 | 32.56 | 87.39 | 70.10 |
| qwen3_8_27b | Factuality | answer | 51 | 92.85 | 92.44 | 48.24 | 76.35 | 76.47 |
| qwen3_8_27b | Logic | answer | 49 | 89.17 | 44.56 | 4.35 | 84.50 | 55.07 |
| qwen3_8_27b | Multi-hop | answer | 36 | 88.44 | 73.28 | 28.30 | 79.31 | 61.90 |
| qwen3_8_27b | Science | answer | 41 | 71.60 | 59.11 | 14.49 | 71.43 | 54.41 |

Detector weights are not included in this incremental release.
