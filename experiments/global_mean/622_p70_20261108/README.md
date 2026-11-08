# Global Mean

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 14 | 80.54 | 90.37 | 32.70 | 72.50 | 79.93 |
| deepseek_r1_distill_llama_8b | Factuality | step | 15 | 74.77 | 41.40 | 8.84 | 69.10 | 43.44 |
| deepseek_r1_distill_llama_8b | Logic | prefix | 18 | 87.46 | 74.57 | 13.48 | 80.67 | 69.69 |
| deepseek_r1_distill_llama_8b | Logic | step | 19 | 77.52 | 12.83 | 5.34 | 88.16 | 19.85 |
| deepseek_r1_distill_llama_8b | Math | prefix | 21 | 83.01 | 91.58 | 14.48 | 79.82 | 86.84 |
| deepseek_r1_distill_llama_8b | Math | step | 14 | 63.66 | 14.76 | 2.72 | 71.45 | 20.95 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 32 | 85.67 | 86.29 | 23.27 | 75.76 | 78.44 |
| deepseek_r1_distill_llama_8b | Multi-hop | step | 14 | 68.39 | 26.20 | 4.57 | 79.25 | 28.83 |
| deepseek_r1_distill_llama_8b | Science | prefix | 17 | 86.56 | 94.91 | 16.33 | 84.95 | 90.91 |
| deepseek_r1_distill_llama_8b | Science | step | 16 | 59.63 | 13.82 | 2.36 | 67.22 | 20.39 |
| gemma_4_31b_it | Factuality | prefix | 58 | 66.44 | 38.04 | 0.75 | 60.33 | 52.93 |
| gemma_4_31b_it | Factuality | step | 41 | 61.05 | 10.37 | 6.41 | 91.46 | 15.49 |
| gemma_4_31b_it | Logic | prefix | 53 | 76.68 | 27.34 | 4.50 | 69.78 | 36.84 |
| gemma_4_31b_it | Logic | step | 28 | 66.92 | 5.40 | 3.39 | 88.46 | 7.09 |
| gemma_4_31b_it | Math | prefix | 60 | 79.10 | 74.48 | 9.18 | 69.17 | 72.61 |
| gemma_4_31b_it | Math | step | 58 | 66.09 | 14.19 | 3.96 | 81.71 | 20.70 |
| gemma_4_31b_it | Multi-hop | answer | 58 | 80.33 | 51.31 | 9.30 | 76.52 | 46.00 |
| gemma_4_31b_it | Multi-hop | prefix | 53 | 71.57 | 55.49 | 7.81 | 69.00 | 68.69 |
| gemma_4_31b_it | Multi-hop | step | 53 | 72.63 | 24.20 | 9.95 | 91.58 | 25.83 |
| gemma_4_31b_it | Science | prefix | 57 | 79.60 | 63.20 | 9.25 | 67.96 | 62.15 |
| gemma_4_31b_it | Science | step | 58 | 66.67 | 8.62 | 2.30 | 83.68 | 14.07 |
| qwen3_8_27b | Factuality | prefix | 49 | 82.26 | 79.44 | 12.06 | 72.42 | 74.07 |
| qwen3_8_27b | Factuality | step | 51 | 69.79 | 21.16 | 5.67 | 76.63 | 27.06 |
| qwen3_8_27b | Logic | prefix | 64 | 81.00 | 37.63 | 6.51 | 75.14 | 42.09 |
| qwen3_8_27b | Logic | step | 64 | 67.05 | 3.24 | 0.65 | 86.47 | 6.70 |
| qwen3_8_27b | Math | prefix | 51 | 71.39 | 53.99 | 6.30 | 64.71 | 54.78 |
| qwen3_8_27b | Math | step | 64 | 61.67 | 8.03 | 1.59 | 74.99 | 13.81 |
| qwen3_8_27b | Multi-hop | prefix | 64 | 83.81 | 82.77 | 8.90 | 75.04 | 78.29 |
| qwen3_8_27b | Multi-hop | step | 51 | 67.38 | 16.12 | 4.15 | 64.75 | 21.64 |

Detector weights are not included in this incremental release.
