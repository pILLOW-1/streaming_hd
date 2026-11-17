# Qwen Step-only hard Top-10% learnable aggregation

Test whether restricting learned pooling to the strongest token scores improves Step error detection.

| Configuration | Token attention |
| --- | --- |
| Legacy learnable aggregation | Content score followed by full within-Step softmax |
| Learnable aggregation, hard Top-10% | Keep the highest ceil(0.10 × Step length) scores, set the remaining scores to negative infinity, and apply one within-Step softmax |

Ties resolve in token order. A Step with one retained token gives it weight one, and its pooling softmax contributes zero gradient to the scorer. The original untransformed token vectors are pooled.

| Training setting | Value |
| --- | --- |
| Qwen domains | Logic and Factuality |
| Splits | Frozen legacy TRAIN, VAL and TEST |
| Head and optimizer | Original Step-only learned scorer and linear head, Step BCE, Adam at 0.001, weight decay 0.0005 |
| Training | Seed 42, random 128-position batches, 20 epoch cap, patience 5 |
| Selection | Evaluate native layers 1–64. Select by VAL Step AUROC, then AP, then lower BCE. |
| TEST | One evaluation of each frozen selected checkpoint, with CPU metric verification from saved predictions |

## Step TEST AUROC

| Configuration | Logic | Factuality |
| --- | --- | --- |
| Legacy learnable aggregation | 0.8871 | 0.8816 |
| Learnable aggregation, hard Top-10% | | |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
| --- | --- | --- |
| Legacy learnable aggregation | | |
| Learnable aggregation, hard Top-10% | | |

The Prefix metric is outside this Step-only experiment. Dense control values reuse the published legacy Step-only scores. Top-10% results are pending.
