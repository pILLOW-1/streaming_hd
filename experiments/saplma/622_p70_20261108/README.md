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

Detector weights are not included in this incremental release.
