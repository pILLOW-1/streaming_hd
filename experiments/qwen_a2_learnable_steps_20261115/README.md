# Qwen A2 learnable Step readouts

These runs keep the historical Step representation trainable through the A2 history term and detach the historical Step probability. Logic and the detached-history comparison are in progress.

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation | | 0.8702 |
| Learnable representation | | 0.8800 |
| Aggregation and representation | | 0.8766 |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation | | 0.8156 |
| Learnable representation | | 0.8402 |
| Aggregation and representation | | 0.8252 |

All published metrics were independently recomputed on CPU from the frozen TEST predictions. Selection was checked against all 64 candidate VAL reports.
