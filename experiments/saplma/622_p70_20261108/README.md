# SAPLMA

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 14 | 78.78 | 89.56 | 30.28 | 72.03 | 79.55 |
| deepseek_r1_distill_llama_8b | Factuality | step | 5 | 79.13 | 48.11 | 9.52 | 75.68 | 49.14 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 32 | 82.82 | 82.41 | 12.12 | 73.13 | 77.09 |
| deepseek_r1_distill_llama_8b | Multi-hop | step | 15 | 82.73 | 47.59 | 14.52 | 84.76 | 48.44 |
| gemma_4_31b_it | Factuality | prefix | 37 | 68.31 | 43.34 | 2.70 | 56.38 | 52.87 |
| gemma_4_31b_it | Factuality | step | 39 | 80.40 | 23.27 | 13.52 | 89.68 | 28.81 |
| gemma_4_31b_it | Logic | prefix | 43 | 76.81 | 32.43 | 8.34 | 76.11 | 36.81 |
| gemma_4_31b_it | Logic | step | 35 | 84.36 | 22.77 | 18.22 | 96.96 | 21.05 |
| gemma_4_31b_it | Math | prefix | 38 | 80.19 | 74.57 | 8.87 | 70.17 | 72.88 |
| gemma_4_31b_it | Math | step | 37 | 81.76 | 30.88 | 12.50 | 88.94 | 35.17 |
| gemma_4_31b_it | Multi-hop | prefix | 33 | 76.96 | 63.62 | 10.11 | 66.58 | 64.77 |
| gemma_4_31b_it | Multi-hop | step | 34 | 84.72 | 48.49 | 29.38 | 92.97 | 46.63 |
| gemma_4_31b_it | Science | prefix | 43 | 81.20 | 64.59 | 8.31 | 69.66 | 63.26 |
| gemma_4_31b_it | Science | step | 35 | 84.32 | 25.18 | 13.63 | 92.29 | 31.40 |
| qwen3_8_27b | Factuality | prefix | 47 | 78.42 | 73.44 | 7.95 | 70.03 | 71.92 |
| qwen3_8_27b | Factuality | step | 36 | 86.07 | 40.72 | 12.10 | 87.28 | 44.25 |
| qwen3_8_27b | Multi-hop | prefix | 46 | 80.49 | 79.54 | 9.52 | 72.94 | 76.63 |
| qwen3_8_27b | Multi-hop | step | 47 | 90.22 | 58.47 | 31.43 | 92.33 | 56.89 |

Detector weights are not included in this incremental release.
