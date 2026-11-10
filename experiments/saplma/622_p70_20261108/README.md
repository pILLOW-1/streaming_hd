# SAPLMA

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | answer | 27 | 74.04 | 86.45 | 10.71 | 71.50 | 82.99 |
| deepseek_r1_distill_llama_8b | Factuality | prefix | 14 | 78.78 | 89.56 | 30.28 | 72.03 | 79.55 |
| deepseek_r1_distill_llama_8b | Factuality | step | 5 | 79.13 | 48.11 | 9.52 | 75.68 | 49.14 |
| deepseek_r1_distill_llama_8b | Logic | prefix | 18 | 87.75 | 76.54 | 16.58 | 80.88 | 70.51 |
| deepseek_r1_distill_llama_8b | Logic | step | 17 | 90.24 | 36.07 | 21.95 | 94.28 | 37.36 |
| deepseek_r1_distill_llama_8b | Math | prefix | 14 | 80.24 | 90.00 | 12.10 | 78.55 | 86.25 |
| deepseek_r1_distill_llama_8b | Math | step | 17 | 83.17 | 39.07 | 15.10 | 87.64 | 41.65 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 32 | 82.82 | 82.41 | 12.12 | 73.13 | 77.09 |
| deepseek_r1_distill_llama_8b | Multi-hop | step | 15 | 82.73 | 47.59 | 14.52 | 84.76 | 48.44 |
| deepseek_r1_distill_llama_8b | Science | prefix | 17 | 84.37 | 93.96 | 14.73 | 84.41 | 90.56 |
| deepseek_r1_distill_llama_8b | Science | step | 18 | 82.82 | 38.84 | 14.11 | 87.92 | 41.37 |
| gemma_4_31b_it | Factuality | answer | 33 | 88.05 | 82.62 | 50.85 | 85.00 | 67.39 |
| gemma_4_31b_it | Factuality | prefix | 37 | 68.31 | 43.34 | 2.70 | 56.38 | 52.87 |
| gemma_4_31b_it | Factuality | step | 39 | 80.40 | 23.27 | 13.52 | 89.68 | 28.81 |
| gemma_4_31b_it | Logic | answer | 28 | 88.88 | 65.71 | 26.67 | 91.00 | 70.00 |
| gemma_4_31b_it | Logic | prefix | 43 | 76.81 | 32.43 | 8.34 | 76.11 | 36.81 |
| gemma_4_31b_it | Logic | step | 35 | 84.36 | 22.77 | 18.22 | 96.96 | 21.05 |
| gemma_4_31b_it | Math | answer | 38 | 87.93 | 71.96 | 1.27 | 86.00 | 84.44 |
| gemma_4_31b_it | Math | prefix | 38 | 80.19 | 74.57 | 8.87 | 70.17 | 72.88 |
| gemma_4_31b_it | Math | step | 37 | 81.76 | 30.88 | 12.50 | 88.94 | 35.17 |
| gemma_4_31b_it | Multi-hop | answer | 23 | 90.25 | 72.87 | 23.26 | 87.39 | 67.42 |
| gemma_4_31b_it | Multi-hop | prefix | 33 | 76.96 | 63.62 | 10.11 | 66.58 | 64.77 |
| gemma_4_31b_it | Multi-hop | step | 34 | 84.72 | 48.49 | 29.38 | 92.97 | 46.63 |
| gemma_4_31b_it | Science | prefix | 43 | 81.20 | 64.59 | 8.31 | 69.66 | 63.26 |
| gemma_4_31b_it | Science | step | 35 | 84.32 | 25.18 | 13.63 | 92.29 | 31.40 |
| qwen3_8_27b | Factuality | answer | 49 | 91.15 | 91.70 | 45.88 | 68.97 | 71.49 |
| qwen3_8_27b | Factuality | prefix | 47 | 78.42 | 73.44 | 7.95 | 70.03 | 71.92 |
| qwen3_8_27b | Factuality | step | 36 | 86.07 | 40.72 | 12.10 | 87.28 | 44.25 |
| qwen3_8_27b | Logic | answer | 60 | 85.53 | 39.85 | 8.70 | 85.50 | 50.85 |
| qwen3_8_27b | Logic | prefix | 27 | 80.78 | 36.38 | 6.12 | 75.65 | 42.68 |
| qwen3_8_27b | Logic | step | 41 | 88.22 | 20.02 | 20.32 | 95.96 | 27.39 |
| qwen3_8_27b | Math | prefix | 47 | 73.98 | 57.18 | 7.10 | 65.67 | 57.90 |
| qwen3_8_27b | Math | step | 47 | 80.46 | 23.76 | 10.77 | 89.61 | 30.49 |
| qwen3_8_27b | Multi-hop | prefix | 46 | 80.49 | 79.54 | 9.52 | 72.94 | 76.63 |
| qwen3_8_27b | Multi-hop | step | 47 | 90.22 | 58.47 | 31.43 | 92.33 | 56.89 |
| qwen3_8_27b | Science | answer | 56 | 69.80 | 52.00 | 8.70 | 67.74 | 46.15 |
| qwen3_8_27b | Science | prefix | 47 | 75.70 | 67.29 | 5.07 | 66.26 | 68.02 |
| qwen3_8_27b | Science | step | 49 | 82.55 | 23.79 | 12.30 | 91.08 | 29.62 |

Detector weights are not included in this incremental release.
