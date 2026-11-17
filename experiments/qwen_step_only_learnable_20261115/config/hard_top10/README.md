# Learnable aggregation, hard Top-10%

Use the [learnable aggregation scorer](../learnable_aggregation/README.md). Retain the highest `ceil(0.10 × Step length)` learned scores, breaking ties in token order. Set all others to negative infinity and apply one within-Step softmax in TRAIN, VAL, and TEST. When only one token is retained, its weight is one and the softmax supplies zero gradient to the scorer. Apply the [legacy recipe](../protocol/README.md) on Logic and Factuality. See the [mask implementation](../../code/aggr_top10/topk.py).
