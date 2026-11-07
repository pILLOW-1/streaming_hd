# Frozen hard-tail TEST CPU feasibility and validated result (2026-09-14)

Source of record: `experiments/streaming_hd/runs/step_hard_tail_rank_cpu_benchmark_20260914/`. Pilot PBS `87205.hpc-head01` F/exit0 on hpc-exec16, 4 CPU/24 GB, 67 s walltime; full PBS `87210.hpc-head01` F/exit0 on hpc-exec16, 4 CPU/24 GB, 78 s walltime. Exact frozen VAL-selected source/checkpoint identity: `TASK.md` and predecessor `test_extension/TEST_FREEZE.json` SHA-256 `9056ac64652df853ff5e96b4a0a3d8e0c7955bf9df84cf21dce12a057c0b44ad`. No training or TEST tuning.

| Model | Pilot Steps/s | Pilot tokens/s | Pilot projected full inference s | Actual full inference s | TEST AUROC | ACC | F1 | AUPRC | Δ AUROC vs parent | Δ AUROC vs all-pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 5480.59 | 115713.33 | 1.83 | 2.13 | 0.897191 | 0.818851 | 0.754386 | 0.829370 | +0.000110 | +0.000451 |
| Llama | 3478.32 | 108140.29 | 3.60 | 4.27 | 0.900746 | 0.823586 | 0.752933 | 0.837409 | -0.000396 | +0.000310 |
| DeepSeek | 4055.31 | 105992.46 | 4.70 | 5.39 | 0.941503 | 0.926781 | 0.678637 | 0.776923 | -0.004456 | -0.001320 |

Pilot projected pure inference sum 10.14 s; actual pure CPU full-inference sum 11.79 s. Pilot process max RSS 787,776 KB; full job PBS max memory 5,126,976 KB, well inside 24 GB. Most end-to-end walltime was staging/hashing of the 1.5/3.2/3.9-GB token files and full independent replay/metric checks, not neural-network forward. The full audit `audits/FULL_FINAL_VALIDATION.json` is PASS, SHA-256 `8b6e86b0d36cef8381c49d613256c0020b0d171b2c81811b7b25d44619113064`, with all 10 listed durable artifact hashes independently rechecked. Full CPU logit replay had zero max error for all models; independent sampled NumPy64 errors were at most 6.83e-7.

The original GPU TEST job `87196.hpc-head01` produced all three prediction CSVs but exited 1 when its validator attempted direct `.numpy()` on BF16 token data. It has no original final PASS audit. Its unchanged predictions were separately checked against complete ordered TEST IDs/labels, full CPU model replay and sampled NumPy64 logits, and its AUROC/ACC/F1/AUPRC match the CPU results to reporting precision. `audits/GPU_PREDICTIONS_RECHECK.json` is PASS, SHA-256 `c5c3da14fa017b8cb47e2f6c6dea38600353191e3ab83e22f915545d7a2e5458`; max GPU-vs-CPU logit difference 2.861023e-6. This independent external recheck does not change the historical GPU PBS exit code or claim that its original audit passed.

Underlying files: `results/PILOT_SUMMARY.json`, `results/{qwen,llama,deepseek}/FULL_INFERENCE.json`, `results/full_cpu_comparison.csv`, `results/gpu_prediction_recheck.csv`, `audits/PILOT_FINAL_VALIDATION.json`, `audits/FULL_FINAL_VALIDATION.json`, `audits/GPU_PREDICTIONS_RECHECK.json`, and `reports/FULL_RESULTS.md` under the CPU run root. All CPU and GPU raw predictions remain in separate run directories.
