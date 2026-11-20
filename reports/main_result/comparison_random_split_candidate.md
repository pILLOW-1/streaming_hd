# Random 6:2:2 split: candidate comparison

This report tracks the current random-split experiments. Only completed results independently verified from saved TEST predictions are shown; empty cells are unfinished. Bold identifies the best verified method within the same generator and domain, using full precision.

## Protocol

- Split whole question families independently within each generator/domain: sort family IDs, shuffle once with Python `Random(42)`, then cut at `floor(0.6 G)` and `floor(0.8 G)`. No label/recovery stratification or seed search; all trajectories are retained.
- Train fresh detectors. Reuse only source-bound native backbone features. Preserve the original recovery-enriched p70 split.
- Streaming uses its original training recipe and independent Step/Prefix VAL selection. A2 uses its original three-class CE and joint mean Step/Prefix VAL selection. Thresholds come from VAL; each frozen selected TEST model is evaluated once. Saved-prediction CPU acceptance does not replay the models.
- This is an exploratory re-split of previously inspected data, not a newly collected blind test. Direct comparisons below use the same random split, not the historical p70 TEST set.
- [Exact split IDs and portable reader](https://huggingface.co/datasets/pillow-streaming/streaming-hd-v1/tree/1b86c98bca0e59d67b5c2d208d1d56b3db0e129e/random_622_seed42) (private dataset access required).

## Step TEST AUROC

| Model / method | Logic | Factuality | Science | Math | Multi-hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek / Streaming |  |  |  |  |  |
| Gemma / Streaming |  |  |  |  |  |
| Qwen / Streaming | 0.8773 | 0.8963 |  |  | 0.9334 |
| Qwen / A2 | **0.8890** | **0.9024** |  |  |  |

## Prefix TEST AUROC

| Model / method | Logic | Factuality | Science | Math | Multi-hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek / Streaming |  |  |  |  |  |
| Gemma / Streaming |  |  |  |  |  |
| Qwen / Streaming | 0.8707 | 0.8966 |  |  | 0.9024 |
| Qwen / A2 | **0.8869** | **0.9148** |  |  |  |

## Recovery Prefix TEST AUROC: pooled

Positive positions are inherited errors: Step clean, Prefix erroneous. Negative positions are recovered: Step clean, Prefix clean, with at least one earlier retained erroneous Prefix. Rank by the final Prefix score, with higher scores indicating error. Pool these positions across all TEST trajectories; no within-trajectory restriction.

| Model / method | Logic | Factuality | Science | Math | Multi-hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek / Streaming |  |  |  |  |  |
| Gemma / Streaming |  |  |  |  |  |
| Qwen / Streaming | 0.7002 | 0.7577 |  |  | 0.7955 |
| Qwen / A2 | **0.7339** | **0.7907** |  |  |  |

## Recovery Prefix TEST AUROC: within trajectory

Use the same positive/negative definitions, compute AUROC separately in each trajectory containing both classes, then average trajectories equally. No cross-trajectory pairs. Eligible trajectories: Qwen Logic 74, Factuality 67, Multi-hop 107.

| Model / method | Logic | Factuality | Science | Math | Multi-hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek / Streaming |  |  |  |  |  |
| Gemma / Streaming |  |  |  |  |  |
| Qwen / Streaming | 0.5028 | 0.4832 |  |  | 0.5661 |
| Qwen / A2 | **0.5576** | **0.5164** |  |  |  |

## Verification and status

- Qwen Logic/Factuality: Streaming and A2 independently accepted on Cetus; RunPod instances stopped.
- Qwen Multi-hop: Streaming Step and Prefix independently accepted; PBS job `342179.hpc-head01` finished with exit 0. Selected Step L51/update 28; Prefix L64/epoch 1.
- Qwen Science/Math: still running. DeepSeek/Gemma results remain empty until source recovery and independent acceptance are complete.
- Private evidence identities: `qwen_random622_streaming_a2_20261120` and `streaming_random622_all_combinations_20261120`. Acceptance receipts, prediction/checkpoint hashes and frozen split provenance stay with their corresponding Cetus runs. No samples, feature banks, predictions or trained weights are included in this report.
