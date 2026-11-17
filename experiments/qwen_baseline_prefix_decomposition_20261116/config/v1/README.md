# v1

Use original frozen Streaming Step probabilities and an unnormalized exponential Prefix representation. Train a linear conditional head on all TRAIN positions using Adam, seed 42, learning rate 0.001, weight decay 0.0005, random batches of 128 positions, a 20-epoch cap, patience 5, and the legacy plateau schedule. Select by VAL Prefix AUROC, then AP, then lower BCE. Evaluate the frozen winner once on TEST. See the [shared protocol](../protocol/README.md) and [training source](../../train.py).
