# v1 + GRU + memory κ

Pool each observed Step with the fixed Streaming exponential weights. Project it into 128 dimensions and update a GRU state with LayerNorm. Four-head causal attention retrieves Step representations through the current Step, with a learned null key and an exact zero value. A gated residual and a linear readout produce the κ logit. The detector reads histories of complete trajectories, including visible unlabelled steps.

Use the exact frozen original Streaming Step probabilities for TRAIN, VAL and TEST. Fit the new κ branch with ordinary all-position BCE on q + (1 − q) κ. The Adam parameters, per-layer v1 epoch budgets and VAL selection rule are shared with v1. Shuffle complete trajectories and batch around 128 labelled positions to preserve recurrence. Freeze the VAL winner before one TEST per domain. Compare Prefix AUROC/AP/BCE and matched within-trajectory recovery AUROC with the saved v1 predictions.
