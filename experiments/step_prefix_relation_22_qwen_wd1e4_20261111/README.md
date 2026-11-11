# Step–Prefix Joint Detection with Fixed Readouts


This experiment supersedes [step_prefix_relation_22_qwen_20261110](../step_prefix_relation_22_qwen_20261110/). All 80 probes are trained from fresh initializations with Adam weight decay 1e-4. The previous run is retained as historical evidence. Hash-verified frozen backbone features may be reused.

## Motivation

Step and Prefix detectors are often trained separately, leaving their logical relationship unused. We model three mutually exclusive states—new error, inherited error still active, and clean—with one softmax head. Step probability is p(new); Prefix probability is p(new) + p(inherited).

This experiment compares fixed readouts, head capacity, and training objectives. It does not yet establish an advantage over independently trained detectors.

## Configurations

Each row runs independently on Qwen3.8-27B in **Math, Logic, Science, Multi-hop, and Factuality**: **16 configurations × 5 domains = 80 runs**. Answer-level detection is excluded.

| ID | Fixed readout | Prediction head | Loss |
|---|---|---|---|
| C01 | Last token | Linear | Marginal BCE |
| C02 | Last token | Linear | Three-class CE |
| C03 | Last token | MLP | Marginal BCE |
| C04 | Last token | MLP | Three-class CE |
| C05 | Global mean | Linear | Marginal BCE |
| C06 | Global mean | Linear | Three-class CE |
| C07 | Global mean | MLP | Marginal BCE |
| C08 | Global mean | MLP | Three-class CE |
| C09 | Step exponential | Linear | Marginal BCE |
| C10 | Step exponential | Linear | Three-class CE |
| C11 | Step exponential | MLP | Marginal BCE |
| C12 | Step exponential | MLP | Three-class CE |
| C13 | Prefix exponential | Linear | Marginal BCE |
| C14 | Prefix exponential | Linear | Three-class CE |
| C15 | Prefix exponential | MLP | Marginal BCE |
| C16 | Prefix exponential | MLP | Three-class CE |

All four readouts use raw FP32 vectors without L2 normalization. Global mean and the two exponential readouts follow Streaming's original token scope and pooling: Step weights ∝ exp(3u), u ∈ [0,1]; Prefix weights ∝ exp(0.003j). Only the native causal context visible through the current Step is used; pooling excludes the prompt and final Answer while preserving original reasoning tokens and boundary overrides.

## Training Parameters

| Parameter | Setting |
|---|---|
| Backbone | Frozen native BF16 Qwen3.8-27B; no truncation or quantization |
| Data | Existing 622_p70 TRAIN/VAL/TEST assignment and labels, unchanged; model/domain fits independent |
| Candidate layers | Native layers 1–64; each configuration trains all candidates independently |
| Linear head | D → 3 |
| MLP head | D → 256 → 128 → 64 → 3; ReLU; no Dropout or extra normalization |
| Outputs | Softmax(new, inherited, clean); Step = p(new), Prefix = p(new) + p(inherited) |
| Marginal BCE | 0.5 × [BCE(Y, p(new)) + BCE(C, p(new) + p(inherited))] |
| Three-class CE | (Y,C): (1,1) → new; (0,1) → inherited; (0,0) → clean |
| Optimizer | Adam; learning rate 1e-3; weight decay 1e-4 |
| Batch | 256 paired Step/Prefix positions; uniform position weighting; no class resampling; retain final partial batch |
| Epoch sampling | Shuffle all TRAIN positions without replacement; same seeded order across configurations |
| Budget | At most 100 epochs; early stopping after 10 epochs without VAL improvement |
| Seed | 42 |
| Checkpoint/layer selection | Maximize mean of Step/Prefix VAL AUROC; ties: mean AP, then lower mean marginal BCE; unresolved exact ties stop selection |
| Thresholds | Separate own-VAL F1-max thresholds; ties: precision, then larger threshold; strict score > threshold |
| TEST | Frozen selected checkpoint once; independent CPU checks from saved predictions; no extra model replay |

The MLP architecture follows SAPLMA; the unified optimizer settings and stopping budget are experiment-design choices, not a full SAPLMA recipe reproduction.

## Results

**CPU-verified: 0 / 80.** Each accepted run is added immediately to `result/<domain>/<readout>/<head>/<loss>.json`, with `result/index.json` refreshed in the same concrete Git commit. Pending entries contain no result values. Result tables use configurations C01–C16 as rows and the five domains as columns. Step and Prefix have separate AUROC tables; `result/step_auroc.csv`, `prefix_auroc.csv`, `step_ap.csv`, and `prefix_ap.csv` preserve the same matrix layout. `result/summary.csv` uses Step AUROC / Prefix AUROC in each domain cell.

Each result records both tasks' AUROC, AP, TPR@1%FPR, ACC, F1, precision, recall, FPR, confusion counts and coverage, plus selected layer/epoch/threshold and configuration, data, checkpoint and prediction hashes. Raw data, feature banks and trained weights are not published.

Frozen settings: [config.json](config.json). Source: [code/ours/step_prefix_relation_22_qwen_wd1e4_20261111](../../code/ours/step_prefix_relation_22_qwen_wd1e4_20261111/).

## Step AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| C01 |  |  |  |  |  |
| C02 |  |  |  |  |  |
| C03 |  |  |  |  |  |
| C04 |  |  |  |  |  |
| C05 |  |  |  |  |  |
| C06 |  |  |  |  |  |
| C07 |  |  |  |  |  |
| C08 |  |  |  |  |  |
| C09 |  |  |  |  |  |
| C10 |  |  |  |  |  |
| C11 |  |  |  |  |  |
| C12 |  |  |  |  |  |
| C13 |  |  |  |  |  |
| C14 |  |  |  |  |  |
| C15 |  |  |  |  |  |
| C16 |  |  |  |  |  |

## Prefix AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| C01 |  |  |  |  |  |
| C02 |  |  |  |  |  |
| C03 |  |  |  |  |  |
| C04 |  |  |  |  |  |
| C05 |  |  |  |  |  |
| C06 |  |  |  |  |  |
| C07 |  |  |  |  |  |
| C08 |  |  |  |  |  |
| C09 |  |  |  |  |  |
| C10 |  |  |  |  |  |
| C11 |  |  |  |  |  |
| C12 |  |  |  |  |  |
| C13 |  |  |  |  |  |
| C14 |  |  |  |  |  |
| C15 |  |  |  |  |  |
| C16 |  |  |  |  |  |

No accepted results yet. Pending configurations have no result values.
