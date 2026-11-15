# Standalone Qwen Step detectors

Measure Step detection when each learnable Step readout is optimized directly for the Step label. Compare the frozen TEST scores with the corresponding learnable A2 runs in the [series comparison](../qwen_c18_a2_series_20261115/README.md).

| Configuration | Token weights | Pooled values |
|---|---|---|
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
| Learnable aggregation | | |
| Learnable representation | | |
| Aggregation and representation | | |

The run uses the existing frozen feature volume and index. The learner computes only the Step logit and Step loss. Published results require independent recomputation from saved predictions.
