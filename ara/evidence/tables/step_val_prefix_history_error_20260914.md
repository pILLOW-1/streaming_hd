# Frozen VAL previous released-Prefix-label diagnostic

Source: `experiments/streaming_hd/runs/step_val_prefix_history_error_20260914/results/conditional_error_table.csv`; independent replay: `experiments/streaming_hd/runs/step_val_prefix_history_error_20260914/audits/INDEPENDENT_VALIDATION.json` (SHA-256 `70c09c807d13a860cc3cd87806b406685f4f5c476941857daa79dbe3968e3929`). The table SHA-256 is `502152e8f67eb9fa44f1dcb3e6ec361aae0d2e8a8c960ac05a84a675cd64e254`.

All 57,801 frozen VAL Steps and 2,355 trajectories passed original-prediction score/label replay, strict canonical Step-to-previous-Prefix join, stage rederivation, fixed `>0.5` threshold, and independent 2,000-replicate trajectory-cluster bootstrap replay. No fitting, TEST access, GPU, or PBS submission.

| Model | n Steps | n trajectories | Δ_FP pp [95% CI] | Δ_FN pp [95% CI] | released Prefix vs Step-OR mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 15,321 | 854 | +14.13 [12.16, 16.24] | +9.90 [6.38, 13.45] | 1,418 |
| Llama | 15,563 | 659 | +21.75 [18.52, 25.12] | +16.24 [12.14, 20.52] | 447 |
| DeepSeek | 26,917 | 842 | +10.16 [8.07, 12.62] | +18.96 [13.28, 24.21] | 4,814 |

`Δ_FP=FPR(G10)-FPR(G00)` and `Δ_FN=FNR(G01)-FNR(G11)`; the first group digit is the previous **released Prefix** label and the second is the current Step label. Released Prefix `cumulative_hallucination` is not identical to a Step-label OR (Qwen 1,418, Llama 447, DeepSeek 4,814 mismatches), so `G01` does not necessarily mean first hallucinated Step. This is a conditional association, not a causal or TEST inference. Full counts, stage rows, CI values, and representative mismatching identities are in the cited run files.
