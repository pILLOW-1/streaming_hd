# Training and evaluation

| Parameter | Value |
|---|---|
| Backbone | Frozen Qwen3.8-27B, BF16 source features; FP32 detector |
| Model/tokenizer revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Split | Whole-question-family random 6:2:2, seed 42 |
| Split SHA-256 | `0c8d4f1f316d147906986fa8179418841f9977c63bb9bc4bf7023b1f6a5998eb` |
| Readouts/history/loss | Original A2 fixed exponential pools, causal predicted-error memory, three-class CE |
| Training weighting | Original supervised-position weighting, unchanged |
| Optimizer | Adam, lr 0.001, weight decay 0.0001; no scheduler |
| Batch / seed | 256 supervised positions / 42 |
| Budget | Maximum 100 epochs, patience 10 |
| Candidate layers | Native 1–64, independent fresh detectors |
| Checkpoint and layer | Joint trajectory-equal weighted Step/Prefix VAL AUROC, then AP, then lower BCE |
| Thresholds | Original own-output VAL F1max rule, unchanged |
| TEST | One selected frozen evaluation/domain; ordinary position-level AUROC |
| Acceptance | Independent Cetus CPU checks of all candidate VAL predictions, selection, IDs/labels/coverage, selected hashes and TEST metrics; no model replay |

Both domains are complete and accepted. Private source, predictions and detector weights remain on Cetus/RunPod. This directory contains aggregate records and configuration only.
