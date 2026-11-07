# Frozen VAL score-density comparison (2026-09-14)

Source: `experiments/streaming_hd/runs/step_val_score_density_20260914/audits/source_audit.json`, `audits/INDEPENDENT_VALIDATION.json`, and `results/auroc_summary.csv`. All 57,801 paired Step IDs, labels, trajectory positions, and source scores match the frozen VAL originals in order; six AUROCs were independently recomputed and agree with the frozen source comparison to <1e-12.

| Model | Normal | Hallucinated | Original Streaming AUROC | Learned-feature weighted AUROC |
|---|---:|---:|---:|---:|
| Qwen | 9,959 | 5,362 | 0.8676227231387592 | 0.8914472406617224 |
| Llama | 9,482 | 6,081 | 0.8701091043256611 | 0.8784157406614446 |
| DeepSeek | 23,234 | 3,683 | 0.9383237022077455 | 0.9465218296021510 |

The figure is a descriptive 2×3 boundary-reflected, normalized KDE of scores on [0,1], with one common 0.035 score-unit bandwidth and shared y scale. It is not a new fit, significance test, or TEST result. Figure and source-hash provenance are in the experiment directory.
