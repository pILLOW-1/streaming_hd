# Step training protocols

The two recipes use the same frozen Qwen features and Step labels. The published C18 fixed readout shares the token pool and Step loss, while using joint Step/Prefix selection. See [C18](../../../qwen_c18_error_memory_20261114/README.md) and the [learnable A2 comparison](../../../qwen_c18_a2_series_20261115/README.md).

## Trajectory recipe

| Training parameter | Setting |
|---|---|
| Backbone and data | Frozen Qwen3.8-27B BF16 token features, original Logic/Factuality splits and position index |
| Head and loss | Linear Step head with binary cross-entropy on the Step labels |
| Optimizer | Adam, seed 42, learning rate 0.001, weight decay 0.0001 |
| Batching | Same trajectory grouping and 256-supervised-position / 24,000-token batch targets as the learnable A2 experiments |
| Candidate layers | Native layers 1–64, split into ranges 1–32 and 33–64 per domain and configuration |
| Selection | Up to 100 epochs, patience 10. Maximize VAL Step AUROC, then AP, then minimize BCE. Evaluate the frozen winner once on TEST. |

## Legacy recipe

| Training parameter | Legacy recipe |
|---|---|
| Optimizer and initialization | Adam, learning rate 0.001, weight decay 0.0005, seed 42 |
| Batching | Shuffle individual labeled Steps each epoch. Train batch 128, evaluation batch 256. Pool only the tokens of each labeled Step. |
| Objective | Unweighted Step BCE with logits |
| Learning-rate schedule | ReduceLROnPlateau on mean training loss, factor 0.1, patience 20, minimum learning rate 1e-6 |
| Checkpoint | At most 20 epochs, patience 5, full VAL Step AUROC improvement greater than 1e-7 |
| Layer and TEST | Select from layers 1–64 by VAL Step AUROC, then AP, then lower BCE. Evaluate the frozen winner once on TEST. |

An independent CPU pass recomputes TEST metrics from saved predictions after checking all 64 VAL candidates.
