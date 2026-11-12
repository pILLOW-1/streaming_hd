# Step–Prefix Joint Detection with Fixed and Paired Readouts


This experiment supersedes [step_prefix_relation_22_qwen_20261110](../step_prefix_relation_22_qwen_20261110/). The original 80 fixed-readout probes are trained from fresh initializations with Adam weight decay 1e-4. The previous run is retained as historical evidence. Hash-verified frozen backbone features may be reused.

## Motivation

Step and Prefix detectors are often trained separately, leaving their logical relationship unused. We model three mutually exclusive states—new error, inherited error still active, and clean—with one softmax head. Step probability is p(new); Prefix probability is p(new) + p(inherited).

This experiment compares fixed readouts, head capacity, and training objectives. It does not yet establish an advantage over independently trained detectors.

## Configurations

C01–C16 run in Math, Logic, Science, Multi-hop, and Factuality. C17–C20 run in Logic, Multi-hop, and Factuality. The study contains 92 configured runs. Answer-level detection is excluded.

| ID | Readout | Prediction head | Loss |
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
| C17 | Step exponential + Prefix exponential | Two linear scalar heads | Marginal BCE |
| C18 | Step exponential + Prefix exponential | Two linear scalar heads | Three-class CE |
| C19 | Step exponential + Prefix exponential | Two MLP scalar heads | Marginal BCE |
| C20 | Step exponential + Prefix exponential | Two MLP scalar heads | Three-class CE |

All four readouts use raw FP32 vectors without L2 normalization. Global mean and the two exponential readouts follow Streaming's original token scope and pooling: Step weights ∝ exp(3u), u ∈ [0,1]; Prefix weights ∝ exp(0.003j). Only the native causal context visible through the current Step is used; pooling excludes the prompt and final Answer while preserving original reasoning tokens and boundary overrides.

C17–C20 feed Step exponential pooling to the Step head and Prefix exponential pooling to the conditional Prefix head. The two heads use scalar outputs with the same hidden-layer widths as the original head family. With sigmoid outputs q and r, the ordered state probabilities are q, (1−q)r, and (1−q)(1−r). Step and Prefix probabilities are q and q + (1−q)r. See the [paired-readout configuration](config.ordered_dual.json) and [Factuality scope](config.ordered_dual.factuality.json).

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

**CPU-verified: 40 / 92.** Each accepted run is added immediately to `result/<domain>/<readout>/<head>/<loss>.json`, with `result/index.json` refreshed in the same concrete Git commit. Pending entries contain no result values. Result tables use configurations C01–C20 as rows and the five domains as columns. Step and Prefix have separate AUROC tables. The highest AUROC in each domain column is bolded. Exact ties share bold formatting. `result/step_auroc.csv`, `prefix_auroc.csv`, `step_ap.csv`, and `prefix_ap.csv` preserve the same matrix layout. `result/summary.csv` uses Step AUROC / Prefix AUROC in each domain cell.

The [paired-readout result index](result/ordered_dual/index.json) links the individual C17–C20 reports. All four Logic runs cover 15,233 TEST positions. The Factuality runs reuse the BF16 per-token bank and cover 5,489 TEST positions. Feature readback and pooled-feature comparison were waived for this RunPod run.

Each result records both tasks' AUROC, AP, TPR@1%FPR, ACC, F1, precision, recall, FPR, confusion counts and coverage, plus selected layer/epoch/threshold and configuration, data, checkpoint and prediction hashes. Raw data, feature banks and trained weights are not published.

Frozen settings: [config.json](config.json). Source: [code/ours/step_prefix_relation_22_qwen_wd1e4_20261111](../../code/ours/step_prefix_relation_22_qwen_wd1e4_20261111/).

## Step AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| C01 |  | 0.8853 |  |  | 0.8639 |
| C02 |  | 0.8897 |  |  | 0.8620 |
| C03 |  | 0.8865 |  |  | 0.8600 |
| C04 |  | 0.8870 |  |  | 0.8585 |
| C05 |  | 0.7441 |  |  | 0.6829 |
| C06 |  | 0.7441 |  |  | 0.6853 |
| C07 |  | 0.7532 |  |  | 0.6927 |
| C08 |  | 0.7465 |  |  | 0.6821 |
| C09 |  | 0.8833 |  |  | 0.8757 |
| C10 |  | 0.8907 |  |  | 0.8770 |
| C11 |  | 0.8801 |  |  | 0.8677 |
| C12 |  | 0.8899 |  |  | 0.8667 |
| C13 |  | 0.7846 |  |  | 0.7160 |
| C14 |  | 0.7829 |  |  | 0.7154 |
| C15 |  | 0.7751 |  |  | 0.7008 |
| C16 |  | 0.7849 |  |  | 0.7048 |
| C17 |  | **0.8927** |  |  | 0.8818 |
| C18 |  | 0.8775 |  |  | **0.8820** |
| C19 |  | 0.8718 |  |  | 0.8783 |
| C20 |  | 0.8658 |  |  | 0.8776 |

## Prefix AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| C01 |  | 0.8169 |  |  | 0.7917 |
| C02 |  | 0.8196 |  |  | 0.7897 |
| C03 |  | 0.8208 |  |  | 0.7905 |
| C04 |  | 0.8232 |  |  | 0.7843 |
| C05 |  | 0.8232 |  |  | 0.8118 |
| C06 |  | 0.8231 |  |  | 0.8078 |
| C07 |  | 0.8250 |  |  | 0.8003 |
| C08 |  | 0.8231 |  |  | 0.8153 |
| C09 |  | 0.8241 |  |  | 0.8044 |
| C10 |  | 0.8282 |  |  | 0.8053 |
| C11 |  | 0.8268 |  |  | 0.7980 |
| C12 |  | 0.8290 |  |  | 0.7979 |
| C13 |  | 0.8284 |  |  | 0.8131 |
| C14 |  | 0.8291 |  |  | 0.8114 |
| C15 |  | 0.8282 |  |  | 0.8042 |
| C16 |  | 0.8382 |  |  | 0.8146 |
| C17 |  | 0.8370 |  |  | 0.8133 |
| C18 |  | 0.8341 |  |  | 0.8106 |
| C19 |  | 0.8371 |  |  | **0.8178** |
| C20 |  | **0.8418** |  |  | 0.8097 |
