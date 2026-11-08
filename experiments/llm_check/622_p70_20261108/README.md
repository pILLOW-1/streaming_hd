# LLM-Check

Baseline evaluation on the selected 6:2:2 assignment. TEST targets trajectories with repeated Prefix 1→0 transitions (target p ≥ 70%, with the frozen domain-specific relaxations).

Each model/domain/level is independent. We select layers and detector weights on complete VAL AUROC, breaking ties by VAL AP, then freeze a VAL F1max threshold and evaluate TEST once. TPR@1%FPR is the maximum empirical TEST ROC TPR at FPR ≤ 1%, without interpolation. Higher scores indicate hallucination.

## Completed results

Metrics below are percentages. JSON preserves unrounded values, thresholds and confusion counts. New rows are added only after saved-score checks pass.

| Model | Domain | Level | Layer | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| deepseek_r1_distill_llama_8b | Factuality | prefix | 23 | 80.89 | 91.00 | 27.85 | 71.09 | 81.86 |
| gemma_4_31b_it | Factuality | prefix | 46 | 55.00 | 31.32 | 0.17 | 47.38 | 48.52 |
| qwen3_8_27b | Logic | prefix | 56 | 77.30 | 27.85 | 0.24 | 66.87 | 37.32 |

Detector weights are not included in this incremental release.
