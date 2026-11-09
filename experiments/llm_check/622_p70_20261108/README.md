# LLM-Check

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 23 | 80.89 | 91.00 | 27.85 | 71.09 | 81.86 |
| deepseek_r1_distill_llama_8b | Logic | prefix | 32 | 79.16 | 68.06 | 18.17 | 75.84 | 59.30 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 29 | 70.71 | 73.33 | 9.59 | 53.05 | 68.23 |
| deepseek_r1_distill_llama_8b | Science | prefix | 1 | 82.24 | 92.63 | 8.91 | 83.47 | 90.09 |
| gemma_4_31b_it | Factuality | prefix | 46 | 55.00 | 31.32 | 0.17 | 47.38 | 48.52 |
| gemma_4_31b_it | Logic | prefix | 60 | 72.48 | 25.21 | 6.37 | 69.18 | 32.05 |
| gemma_4_31b_it | Math | prefix | 48 | 73.55 | 64.13 | 2.28 | 66.72 | 70.92 |
| gemma_4_31b_it | Multi-hop | prefix | 60 | 70.38 | 57.25 | 11.03 | 61.12 | 61.22 |
| gemma_4_31b_it | Science | prefix | 12 | 69.24 | 45.87 | 1.11 | 60.15 | 54.92 |
| qwen3_8_27b | Factuality | prefix | 12 | 77.90 | 74.12 | 5.64 | 69.59 | 69.33 |
| qwen3_8_27b | Logic | prefix | 56 | 77.30 | 27.85 | 0.24 | 66.87 | 37.32 |
| qwen3_8_27b | Math | prefix | 8 | 62.88 | 42.78 | 1.84 | 47.21 | 51.79 |
| qwen3_8_27b | Multi-hop | prefix | 60 | 71.91 | 72.08 | 6.74 | 64.19 | 70.63 |
| qwen3_8_27b | Science | prefix | 32 | 66.26 | 55.47 | 2.21 | 54.74 | 63.87 |

Detector weights are not included in this incremental release.
