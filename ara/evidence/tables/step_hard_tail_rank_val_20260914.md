# Hard-tail Step rank, frozen VAL (2026-09-14)

Source: `experiments/streaming_hd/runs/step_hard_tail_rank_20260914/reports/RESULTS.md`; independent replay and artifact hashes: `experiments/streaming_hd/runs/step_hard_tail_rank_20260914/audits/VAL_FINAL_VALIDATION.json`. Current-run PBS job `87165.hpc-head01` finished with exit 0. Exactly three seed-42 fits, one per model; no TEST access.

| Model | Hard-tail VAL AUROC | Δ vs frozen parent | Δ vs all-pairs rank | ACC | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.892403 | +0.000956 | +0.000704 | 0.809608 | 0.730879 | 0.811017 |
| Llama | 0.879000 | +0.000584 | +0.000320 | 0.795412 | 0.734755 | 0.813548 |
| DeepSeek | 0.950099 | +0.003577 | +0.002714 | 0.932273 | 0.717145 | 0.814630 |

The independent audit reports PASS: matching one-variable configuration, TRAIN/VAL disjointness, exact sampler replay, best-checkpoint selection, full VAL logits, and 67 artifact hashes verified again after shared sync. This is single-seed VAL evidence only; stage metrics are diagnostic and do not reselect checkpoints. The first allocated job `87129.hpc-head01` exited before fitting on occupied GPUs and is retained as an operational failure, not an extra fit.
