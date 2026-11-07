# Original Streaming frozen VAL prior-Prefix-label diagnostic

Source: `experiments/streaming_hd/runs/step_val_prefix_history_error_streaming_20260914/results/streaming_conditional_error_table.csv` (SHA-256 `1dddcd632093fac170e16587e1f2a23acb3ac2381ce3857fbfbdee8d74cecd8d`); paired same-population table: `results/paired_streaming_vs_learned_overall.csv` (SHA-256 `063956da1ffa2c39f4b5d28cf5c50332d5195bb7f53ecd512aa63758f23c7e36`); independent audit: `audits/INDEPENDENT_VALIDATION.json` (SHA-256 `572b37206522637f85cbd516e8fc7ff3256160c3f76ddca17587fc97e05a2c94`).

Original Streaming's frozen `S_prob > 0.5` prediction and larger-score-positive direction were verified against original archived VAL scores/logits/predictions and source code. The unchanged predecessor's 57,801 Step group memberships and 2,355 complete trajectories were reused. All 12 Overall/Early/Middle/Late rows, source identities/hashes, 2,000 seed-42 trajectory-cluster bootstrap CIs, and 18 stage/12 ECDF figure-data cells passed independent replay. No new fit, TEST, GPU or PBS action.

| Model | n Steps | Δ_FP percentage points [95% CI] | Δ_FN percentage points [95% CI] |
| --- | ---: | ---: | ---: |
| Qwen | 15,321 | +21.93 [19.42, 24.70] | +22.52 [19.11, 26.08] |
| Llama | 15,563 | +20.15 [17.17, 23.18] | +32.95 [28.82, 37.01] |
| DeepSeek | 26,917 | +9.01 [7.03, 11.38] | +20.55 [15.65, 25.49] |

Group definitions use previous *released Prefix* label, not Step-derived OR. `G01` does not necessarily mean onset. The paired table preserves all four group rates and both contrasts for original Streaming and learned-feature no-L2 on identical model-specific VAL populations; no uniform method advantage or causal interpretation is claimed.
