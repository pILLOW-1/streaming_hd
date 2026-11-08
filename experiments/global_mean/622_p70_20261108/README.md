# Global Mean

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 14 | 80.54 | 90.37 | 32.70 | 72.50 | 79.93 |
| deepseek_r1_distill_llama_8b | Factuality | step | 15 | 74.77 | 41.40 | 8.84 | 69.10 | 43.44 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 32 | 85.67 | 86.29 | 23.27 | 75.76 | 78.44 |
| deepseek_r1_distill_llama_8b | Multi-hop | step | 14 | 68.39 | 26.20 | 4.57 | 79.25 | 28.83 |
| gemma_4_31b_it | Factuality | prefix | 58 | 66.44 | 38.04 | 0.75 | 60.33 | 52.93 |
| gemma_4_31b_it | Factuality | step | 41 | 61.05 | 10.37 | 6.41 | 91.46 | 15.49 |
| gemma_4_31b_it | Logic | prefix | 53 | 76.68 | 27.34 | 4.50 | 69.78 | 36.84 |
| gemma_4_31b_it | Logic | step | 28 | 66.92 | 5.40 | 3.39 | 88.46 | 7.09 |
| gemma_4_31b_it | Math | prefix | 60 | 79.10 | 74.48 | 9.18 | 69.17 | 72.61 |
| gemma_4_31b_it | Math | step | 58 | 66.09 | 14.19 | 3.96 | 81.71 | 20.70 |
| gemma_4_31b_it | Multi-hop | prefix | 53 | 71.57 | 55.49 | 7.81 | 69.00 | 68.69 |
| gemma_4_31b_it | Multi-hop | step | 53 | 72.63 | 24.20 | 9.95 | 91.58 | 25.83 |
| gemma_4_31b_it | Science | prefix | 57 | 79.60 | 63.20 | 9.25 | 67.96 | 62.15 |
| gemma_4_31b_it | Science | step | 58 | 66.67 | 8.62 | 2.30 | 83.68 | 14.07 |

Detector weights are not included in this incremental release.
