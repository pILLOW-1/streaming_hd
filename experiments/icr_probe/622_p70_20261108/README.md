# ICR Probe

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 0 | 63.26 | 81.25 | 22.66 | 65.45 | 77.21 |
| deepseek_r1_distill_llama_8b | Logic | prefix | 0 | 84.33 | 72.59 | 18.50 | 77.69 | 67.77 |
| deepseek_r1_distill_llama_8b | Multi-hop | prefix | 0 | 81.28 | 80.94 | 9.10 | 72.34 | 75.80 |
| gemma_4_31b_it | Multi-hop | prefix | 0 | 71.94 | 54.28 | 2.11 | 62.24 | 60.81 |
| qwen3_8_27b | Factuality | prefix | 0 | 77.21 | 73.94 | 13.43 | 69.19 | 69.78 |
| qwen3_8_27b | Logic | prefix | 0 | 78.49 | 33.17 | 5.24 | 68.84 | 38.33 |

Detector weights are not included in this incremental release.
