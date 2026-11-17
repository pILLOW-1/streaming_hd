# Qwen conditional Prefix ranking

Test whether direct within-trajectory ordinal supervision improves error-recovery discrimination while preserving Prefix detection across all positions.

| Configuration | Conditional Prefix head | Training objective |
| --- | --- | --- |
| BCE control | Linear on original, unnormalized Streaming Prefix features | All-position composed Prefix BCE |
| BCE + trajectory ranking | Same linear head and features | BCE plus 0.1 times the trajectory-macro logistic pair loss |

| Parameter | Value |
| --- | --- |
| Step probability | Original published Streaming scores, frozen |
| Optimization | Legacy Adam 0.001, weight decay 0.0005, seed 42, identical 128-position BCE batches |
| Ranking batches | At most 32 uniformly sampled trajectories per update, all eligible positive–negative pairs per sampled trajectory |
| Training length | Match the existing BCE control's number of optimizer updates at each layer |
| Selection | VAL Prefix AUROC, then AP, then lower BCE across the 64 layers and candidate epochs |
| Evaluation | One frozen TEST pass for the ranking arm. Reuse the control's saved predictions. |

Use labeled positions with no current Step error and at least one earlier labeled position establishing an error. Pair every inherited positive with every recovered clean negative on the same trajectory, allowing multiple earlier errors. Apply the ranking loss to conditional logits and average within trajectory before averaging across trajectories. For recovery comparisons, reconstruct the control's conditional probability from its saved composed score and the frozen Step probability. Compare both arms on the same recoverable pairs and report coverage. The time-only comparator assigns higher risk to positions closer to the first observed error.

## Step TEST AUROC

| Configuration | Logic | Factuality |
| --- | --- | --- |
| BCE control | | |
| BCE + trajectory ranking | | |

Step scores are frozen. Step metrics are therefore omitted.

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
| --- | --- | --- |
| BCE control | 0.831632 | 0.818123 |
| BCE + trajectory ranking | 0.835883 | |

## Conditional recovery AUROC (trajectory macro)

| Configuration | Logic VAL | Factuality VAL | Logic TEST | Factuality TEST |
| --- | --- | --- | --- | --- |
| BCE control | 0.563189 | | 0.542960 | |
| BCE + trajectory ranking | 0.571456 | | 0.555914 | |
| Time-only comparator | 0.468605 | | 0.479852 | |

The BCE-control Prefix scores reuse the verified legacy results. Completed ranking arms are listed in the result index.
