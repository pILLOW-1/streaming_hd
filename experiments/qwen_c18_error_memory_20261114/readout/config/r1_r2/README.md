# A2 Step and memory readouts

R0 reuses the original A2 Step exponential pool. R1 replaces both the Step input and memory content with the current candidate layer's last Step token. R2 concatenates that last token with the original Step exponential pool. Prefix input remains the original raw Prefix exponential pool. All earlier steps contribute to causal memory, weighted by detached predicted Step probabilities.

| Parameter | Setting |
|---|---|
| Objective | Original C18 three-class CE; Step and conditional Prefix heads trained jointly |
| Optimizer | Adam, LR 0.001, weight decay 0.0001, seed 42 |
| Budget | Position batch 256, max 100 epochs, patience 10; no scheduler |
| Selection | Layers 1–64; joint mean Step/Prefix VAL AUROC, mean AP, then lower mean BCE |
| Evaluation | One frozen selected TEST; independent CPU metrics from saved predictions |
| Normalization | None |

Factuality R1 and R2 both select layer 47, epoch 9. See [frozen configuration](config.json) and [executed worker snapshot](source/worker.py). The snapshots retain their original private-runtime helper imports; they document the executed source rather than bundling data or runtime environments.
