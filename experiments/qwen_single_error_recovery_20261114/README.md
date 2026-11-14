# Qwen single-error recovery diagnostic

Evaluate whether the representation of a known earlier erroneous step helps distinguish inherited from recovered positions. Use frozen Logic and Factuality TRAIN/VAL token banks. Require Y=0 at the current step and exactly one earlier Y=1 on a contiguous observed label history. Set the target to current C. The earlier error location is supplied by the annotation for this diagnostic.

| Configuration | Input | Head |
|---|---|---|
| A | Streaming exponential Prefix through current step | Linear |
| B | Current step last token | Linear |
| C | Current and earlier error step last tokens | Linear |
| D | C plus elementwise product and absolute difference | MLP, 64 ReLU units |

| Setting | Value |
|---|---|
| Native layers | 16, 32, 48, 64 |
| Training | AdamW, BCE, seed 42, learning rate 0.001, weight decay 0.0001, batch 128, 20 epochs |
| Normalization | L2-normalize each input block per position |
| Selection | Highest VAL within-trajectory macro AUROC, then within-pair and pooled AUROC |
| Evaluation | VAL only. Report pooled, within-pair and within-trajectory macro AUROC and coverage. |

C vs B isolates access to the annotated error representation under a linear head. D vs C changes both input features and head capacity. A null result is specific to these readouts, layers and cohort.

## VAL results

All eight configurations completed. Each cell gives pooled / within-trajectory pair-weighted / within-trajectory macro AUROC, followed by the selected native layer.

| Configuration | Logic | Factuality |
|---|---|---|
| A | 0.5861 / 0.5332 / 0.6440 (L64) | 0.5070 / 0.5350 / 0.4949 (L64) |
| B | 0.5077 / 0.4953 / 0.6157 (L64) | 0.5484 / 0.4979 / 0.6491 (L64) |
| C | 0.4373 / 0.4901 / 0.6053 (L64) | 0.5381 / 0.4979 / 0.6473 (L64) |
| D | 0.4429 / 0.5063 / 0.5975 (L64) | 0.5581 / 0.5144 / 0.4996 (L32) |

Logic VAL has 999 positions across 43 trajectories, with 25 mixed-label trajectories. Factuality VAL has 369 positions across 60 trajectories, with 19 mixed-label trajectories. VAL is used for selection and reporting.
