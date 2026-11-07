# Zero-margin hinge Global Rank: frozen TEST evidence

Source of record: `experiments/streaming_hd/runs/step_global_hinge_rank_20260914/audits/TEST_FINAL_VALIDATION.json` (SHA-256 `e4f07332d58d5f3e7d928c09008d3fb04619cf97fca493d016a43fc213567513`), `results/test/comparison.csv`, `results/test/deltas_pp.csv`, and `results/test/hinge_activity.json`. The separate CPU-only repair audit is `audits/POST_TEST_AUDIT_REPAIR.json`.

| Model | TEST N | Hinge AUROC | AUPRC | ACC | F1 | Precision | Recall | ΔAUROC vs all-pairs softplus (pp) | Hinge-active TRAIN pair fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 10,047 | 0.898101 | 0.829994 | 0.813676 | 0.749263 | 0.739360 | 0.759435 | +0.136173 | 0.072150 |
| Llama | 12,533 | 0.900439 | 0.835244 | 0.821032 | 0.755744 | 0.748975 | 0.762637 | +0.000262 | 0.071670 |
| DeepSeek | 19,066 | 0.945773 | 0.789112 | 0.929193 | 0.712644 | 0.760909 | 0.670136 | +0.295011 | 0.038778 |

Three seed-42 fits and three one-time TEST inferences completed after VAL checkpoint freeze. Original PBS `87219.hpc-head01` ran on `hpc-exec09` and ended `F/exit 1` because the finalizer attempted to JSON-encode a NumPy `int64` sampled replay index after all scientific checks and report generation. Its output artifacts were synced. The CPU-only post-TEST repair independently rechecked ordered canonical IDs, labels and trajectory fields, exact source/prediction/checkpoint hashes, all 18 metric rows, all 15 percentage-point comparator rows, hinge activity and 21 final run-relative artifact hashes. It did not retrain, infer, select on TEST, or alter any original result file. The original finalizer's sampled NumPy64 logit assertions passed by code-path evidence before serialization, but their numerical error values were not durably recorded; the repair did not rerun the heavy token-cache replay. The PBS failure and monitor `FAILED_JOB` remain historical facts. TEST comparisons are exploratory because earlier experiments already viewed TEST.
