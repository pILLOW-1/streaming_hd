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

Detector weights are not included in this incremental release.
