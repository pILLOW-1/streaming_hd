# Learnable aggregation

Score each frozen token with the learned scorer, apply one within-Step softmax, and pool the original token vectors. Train the scorer and linear Step head jointly. See the [shared protocol](../protocol/README.md).
