# Standalone Qwen Step detectors

**RunPod status (2026-11-16 AEDT):** The researcher cancelled the remaining RunPod work and shut down the instances. The tables retain published, independently verified results.

Measure Step detection when each learnable Step readout is optimized directly for the Step label. Include Streaming fixed exponential aggregation with frozen token values as the reference. Compare the frozen TEST scores with the corresponding learnable A2 runs in the [series comparison](../qwen_c18_a2_series_20261115/README.md).

| Configuration | Token weights | Pooled values |
|---|---|---|
| Streaming fixed aggregation | Fixed exponential weights, normalized within each Step | Frozen token features |
| Learnable aggregation | Content-only scorer with within-step softmax | Frozen token features |
| Learnable representation | Fixed exponential weights | Linear-256, BatchNorm and ReLU |
| Aggregation and representation | Linear score on the shared representation with within-step softmax | Shared learned representation |

| Training parameter | Setting |
|---|---|
| Backbone and data | Frozen Qwen3.8-27B BF16 token features, original Logic/Factuality splits and position index |
| Head and loss | Linear Step head with binary cross-entropy on the Step labels |
| Optimizer | Adam, seed 42, learning rate 0.001, weight decay 0.0001 |
| Batching | Same trajectory grouping and 256-supervised-position / 24,000-token batch targets as the learnable A2 experiments |
| Candidate layers | Native layers 1–64, split into ranges 1–32 and 33–64 per domain and configuration |
| Selection | Up to 100 epochs, patience 10. Maximize VAL Step AUROC, then AP, then minimize BCE. Evaluate the frozen winner once on TEST. |

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Streaming fixed aggregation | | |
| Learnable aggregation |  | 0.8768 |
| Learnable representation | 0.8666 | 0.8807 |
| Aggregation and representation | 0.8752 | 0.8740 |

Each populated cell was independently recomputed from frozen TEST predictions on CPU. Selection was checked against all 64 candidate VAL reports. Pending configurations have empty cells.

Streaming uses within-Step weights proportional to `exp(3u)`, where `u` runs uniformly from zero to one across the Step tokens. Fixed pooled vectors are reused across detector epochs.

The fixed Streaming arm shares the fixed Step pool and Step loss in [C18 (A1)](../qwen_c18_error_memory_20261114/README.md). The C18 three-class objective factorizes into Step BCE and conditional Prefix BCE masked to Step-negative positions. The Step-only run keeps trajectory batching and VAL Step-only checkpoint and layer selection. C18 uses shuffled Step examples and joint Step/Prefix selection.


## Legacy training recipe on the current data

This rerun uses the same frozen Logic and Factuality splits, three learnable readouts and linear Step head. It measures the effect of restoring the pre-dataset-change training recipe.

| Training parameter | Legacy recipe |
|---|---|
| Optimizer and initialization | Adam, learning rate 0.001, weight decay 0.0005, seed 42 |
| Batching | Shuffle individual labeled Steps each epoch. Train batch 128, evaluation batch 256. Pool only the tokens of each labeled Step. |
| Objective | Unweighted Step BCE with logits |
| Learning-rate schedule | ReduceLROnPlateau on mean training loss, factor 0.1, patience 20, minimum learning rate 1e-6 |
| Checkpoint | At most 20 epochs, patience 5, full VAL Step AUROC improvement greater than 1e-7 |
| Layer and TEST | Select from layers 1–64 by VAL Step AUROC, then AP, then lower BCE. Evaluate the frozen winner once on TEST. |

**Streaming + learned score residual.** Pool frozen token features with $w_{t,i}=\mathrm{softmax}_{i}(3u_{t,i}+s_{t,i})$. The fixed positional score and the learnable aggregation scorer output are added before one softmax within each Step. Zero-initialize the scorer output layer to recover fixed Streaming pooling at initialization. Use the same linear Step head, training recipe and layer selection.

### Legacy recipe Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| [Original Streaming](../streaming_622_p70/result/metrics.json) | 0.8716 | 0.8720 |
| Learnable aggregation | 0.8871 | 0.8816 |
| Learnable representation | 0.8785 | 0.8767 |
| Aggregation and representation | 0.8550 | 0.8705 |
| Streaming + learned score residual | 0.8882 | 0.8742 |

Fill each cell after independent verification from the saved TEST predictions.
