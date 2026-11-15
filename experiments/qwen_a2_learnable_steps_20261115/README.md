# Qwen A2 learnable Step readouts

These runs keep the historical Step representation trainable through the A2 history term and detach the historical Step probability. Logic and the detached-history comparison are in progress.

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation |  | 0.8702 |
| Learnable representation | 0.8681 | 0.8800 |
| Aggregation and representation | 0.8684 | 0.8766 |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation |  | 0.8156 |
| Learnable representation | 0.8335 | 0.8402 |
| Aggregation and representation | 0.8395 | 0.8252 |

Each populated cell was independently recomputed from frozen TEST predictions on CPU. Selection was checked against all 64 candidate VAL reports. Pending configurations have empty cells.
