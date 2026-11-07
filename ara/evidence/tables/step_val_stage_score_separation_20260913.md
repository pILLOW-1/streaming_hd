# Frozen VAL stage score diagnostic, 2026-09-13

Source: `experiments/streaming_hd/runs/step_val_stage_score_separation_20260913/tables/val_stage_score_diagnostic.csv` (SHA-256 `720ca1b3b69682e487dcf98f89c8f313f652a77fe662ccdc213bf0b7cea0b513`); independent audit: `experiments/streaming_hd/runs/step_val_stage_score_separation_20260913/audits/INDEPENDENT_VALIDATION.json` (SHA-256 `f277e79da1ccc10b854c29ab73070316a5a4cb87efeaa0252f3bae76c0884c76`).

| Model | Early AUROC | Middle AUROC | Late AUROC | Late minus Early [95% trajectory-bootstrap CI] | Early to Late normal-logit median | Early to Late positive prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.9544 | 0.8499 | 0.7923 | -0.1621 [-0.179, -0.143] | -4.988 to -0.930 | 12.9% to 53.8% |
| Llama | 0.9172 | 0.8403 | 0.8018 | -0.1154 [-0.139, -0.093] | -4.145 to -0.604 | 17.2% to 59.6% |
| DeepSeek | 0.9544 | 0.9423 | 0.9406 | -0.0138 [-0.030, 0.002] | -4.759 to -4.280 | 14.3% to 13.7% |

Nine model-stage cells sum to 57,801 canonical VAL Steps; all FN/FP/FNR/FPR counts match the predecessor exactly. Original logits were used for score distributions and AUROC, probabilities for Brier/calibration summaries. The independently reproduced intervals use 500 seed-42 resamples of complete trajectories; no replicate lacked a class. No training, inference, threshold tuning, or TEST access.
