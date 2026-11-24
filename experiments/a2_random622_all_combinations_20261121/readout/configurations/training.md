# Shared training and evaluation

| Parameter | Value |
|---|---|
| Backbone | Frozen Qwen3.8-27B, native BF16 source features |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Split | Original whole-question-family random 6:2:2, seed 42 |
| Logic/Factuality split artifact SHA-256 | `0c8d4f1f316d147906986fa8179418841f9977c63bb9bc4bf7023b1f6a5998eb` |
| Detector / seed | FP32 / 42 |
| Optimizer | Adam, learning rate 0.001, weight decay 0.0001 |
| Batch | 256 supervised positions, with complete causal original-Step histories |
| Budget | 100 epochs, patience 10, no scheduler |
| Candidate layers | 1–64; independent detector and optimizer per layer |
| Selection | Joint mean Step/Prefix VAL AUROC, then mean AP, then lower mean marginal BCE |
| Threshold | Each output's own VAL F1max threshold |
| TEST | One frozen selected evaluation; CPU checks saved predictions, no model replay |

Retain all earlier original Steps, including Steps without supervised retained positions. Do not truncate token histories or change the detach rule. Checkpoint selection is joint; Step and Prefix share the selected layer and epoch within each configuration/domain. Raw inputs, feature banks, prediction arrays and weights remain private on RunPod/Cetus.

Only threshold-independent AUROC and AP are published here; operating-point metrics and raw prediction arrays are not included.
