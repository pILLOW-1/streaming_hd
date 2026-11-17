# GRU + memory κ with frozen Learnable aggregation

Aggregate Step tokens with the frozen scorer at the fixed Prefix v1 layer. Train a projection, causal GRU memory, and a conditional head that predicts κ. Compose the Prefix probability as `q + (1 - q) * κ`, where `q` comes from the frozen Step VAL winner. Apply Prefix BCE to the composed probability at all TRAIN positions.

| Domain | Frozen Step probability checkpoint | Frozen Step representation checkpoint |
| --- | --- | --- |
| Logic | L35, epoch 3 | L39, epoch 4 |
| Factuality | L49, epoch 1 | L49, epoch 1 |

Choose the checkpoint by Prefix VAL AUROC and the threshold by Prefix VAL F1. Evaluate the frozen checkpoint on TEST. The summary table reports recovery AUROC for the composed Prefix score. The result records also report recovery AUROC for κ.
