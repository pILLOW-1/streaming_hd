# Qwen conditional Prefix persistence probe

Estimate the probability that an earlier error remains active after a step that introduces no new error. Include positions where the immediately preceding original step has C=1 and the current step has Y=0. The target is current C. Reset history at trajectory boundaries and exclude positions with an unlabeled predecessor.

| Configuration | Domains | Readout | Head |
|---|---|---|---|
| Last token | Logic, Factuality | Last token in the current step | Linear, one logit |
| Streaming Step | Logic, Factuality | Current-step local exponential pooling | Linear, one logit |
| Last token + SAPLMA | Logic, Factuality | Last token in the current step | SAPLMA MLP, one logit |
| Streaming Prefix | Logic, Factuality | Exponential pooling over the reasoning prefix through the current step | Linear, one logit |

| Training setting | Value |
|---|---|
| Backbone | Qwen3.8-27B, frozen BF16 feature bank, native layers 1–64 |
| Input | FP32 readout, per-position L2 normalization for linear probes and raw features for SAPLMA |
| Streaming Step optimization | Seed 42, full-batch BCE plus 1e-4 times half squared parameter norm, observed L-BFGS, strong Wolfe, learning rate 1, at most 100 iterations |
| Streaming Prefix pooling | Normalize weights proportional to exp(0.003 × token index). Use the same optimization as the Streaming Step linear probe. |
| SAPLMA optimization | Seed 42, widths 256/128/64 with ReLU, Xavier initialization, BCE, Keras Adam at 0.001 with epsilon 1e-7, batch 32, five epochs |
| Selection | Highest eligible VAL AUROC across layers and accepted updates or epochs, then AP |

Fit on TRAIN and report eligible VAL AUROC. The same positions are used for both readouts. VAL is used for selection, so the reported values are diagnostic.

| Readout | Logic VAL AUROC | Factuality VAL AUROC |
|---|---:|---:|
| Last token | 0.5903 | 0.7727 |
| Streaming Step | 0.5853 | 0.7778 |
| Last token + SAPLMA | 0.5850 | 0.7674 |
| Streaming Prefix | 0.6193 | 0.8286 |


Within-trajectory VAL AUROC compares eligible positive and negative positions in the same trajectory, using the same selected model. Cells show pair-weighted AUROC followed by the equal-weight trajectory mean. Parentheses give contributing trajectories over all trajectories with eligible positions.

| Readout | Logic | Factuality |
|---|---:|---:|
| Last token | 0.5373 / 0.5664 (40/70) | 0.4965 / 0.4976 (38/90) |
| Streaming Step | 0.5419 / 0.5301 (40/70) | 0.5325 / 0.4861 (38/90) |
| Last token + SAPLMA | 0.5264 / 0.5370 (40/70) | 0.5174 / 0.5637 (38/90) |
| Streaming Prefix | 0.5098 / 0.5188 (40/70) | 0.6252 / 0.5285 (38/90) |
