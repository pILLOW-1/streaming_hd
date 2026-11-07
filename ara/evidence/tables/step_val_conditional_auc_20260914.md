# Frozen VAL prior-Prefix conditional AUROC

Run: `experiments/streaming_hd/runs/step_val_conditional_auc_20260914/`. Main table: `results/conditional_auc_main.csv` (SHA-256 `121095c1263c3ab9dcc1b0866e4f8d78c479731d3804ffd103a64428e706ffcc`); paired method differences: `results/paired_method_auc_differences.csv` (SHA-256 `ef2a8de44e892d1c11e1ba9f6605f64b05a0f16872cb1450509ddbd46c29cffd`); independent audit: `audits/INDEPENDENT_VALIDATION.json` (SHA-256 `4a726694c6618e840464bf915cbaf2b564ba9c9c19aba6e723d0a42d58446042`, status PASS). Full 2,000-replicate paired trajectory-bootstrap trace is `results/trajectory_bootstrap_auc.csv`; the bounded comparison figure is `figures/conditional_auc_comparison.png` and `.pdf`; full limitations are in `reports/RESULTS.md`.

The same 57,801 frozen VAL Steps and actual previous released-Prefix groups were matched between original Streaming (S) and learned-feature weighted no-L2 (L). `AUC_clean` compares G00 normal versus G01 hallucinated, `AUC_error` G10 normal versus G11 hallucinated; larger score denotes hallucination. An independent implementation replayed all source joins, six methods/models, all 12,000 stored bootstrap rows, percentile CIs, pair differences and the exact four-pair decomposition of overall AUROC. Degenerate replicates: zero. No fitting, TEST, GPU or PBS action.

| Model | S clean | L clean | S error | L error | Paired L−S clean [95% CI] | Paired L−S error [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | .8612 | .9068 | .7720 | .8060 | +.0456 [.0387, .0528] | +.0340 [.0250, .0437] |
| Llama | .8339 | .8805 | .7981 | .8020 | +.0466 [.0357, .0578] | +.0039 [−.0053, .0130] |
| DeepSeek | .9270 | .9446 | .8705 | .8874 | +.0176 [.0119, .0238] | +.0170 [.0121, .0226] |

The cross-state comparisons G00:G11 and G10:G01 account for, respectively, 58.7%/57.7%/68.3% and 5.1%/3.8%/3.0% of all positive-negative pairs in Qwen/Llama/DeepSeek. Conditional-only AUCs therefore do not explain overall AUC alone. Released Prefix labels are not Step-label OR; G01 is not necessarily onset. These VAL AUROCs describe ranking, not calibration, causality or TEST performance. Llama's Error paired CI includes zero.
