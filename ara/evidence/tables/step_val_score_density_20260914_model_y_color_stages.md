# Four versioned frozen-VAL score-density figures (2026-09-14)

Sources: `experiments/streaming_hd/runs/step_val_score_density_20260914/results/score_density_model_y_color_24_panels.csv`, `results/paired_val_predictions_with_stage_model_y_color.csv`, and `audits/score_density_2x3_model_y_color_INDEPENDENT_VALIDATION.json`. The independently implemented audit passed on all 57,801 VAL Steps, four 2×3 PNG/PDF pairs, 24 panel rows, distinct colors, model-wise shared vertical y limits, unchanged original hashes, and nine predecessor stage populations and learned-feature AUROCs.

| Model | Stage | Original Streaming AUROC (new VAL replay) | Learned-feature AUROC (prior stage report replay) |
|---|---|---:|---:|
| Qwen | Early | 0.934235 | 0.954447 |
| Qwen | Middle | 0.815989 | 0.849879 |
| Qwen | Late | 0.763823 | 0.792329 |
| Llama | Early | 0.905945 | 0.917184 |
| Llama | Middle | 0.836406 | 0.840296 |
| Llama | Late | 0.784651 | 0.801768 |
| DeepSeek | Early | 0.942631 | 0.954412 |
| DeepSeek | Middle | 0.932449 | 0.942280 |
| DeepSeek | Late | 0.938318 | 0.940610 |

Stage AUROC uses the frozen source logits; KDE curves use the corresponding frozen probabilities, retaining 1024 bins, Gaussian bandwidth 0.035, [0,1] reflection, and per-class unit integral. The exact counts, 24 AUROCs, hashes and figure paths are in the experiment directory. Qwen/Llama show a larger descriptive Early-to-Late AUROC drop than DeepSeek on this VAL population; this is not a new TEST or causal result. Across-stage density peak heights are not directly comparable because y-axis limits vary between figures. Local Windows `D:` transfer remains unverified from the remote host; all eight figure files are staged under the project reports mirror for local transfer.
