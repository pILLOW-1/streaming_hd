# Streaming reproduction: 6:2:2, TEST p ≥ 70%

## Method

Three generator models, five domains, independent detector fits per model/domain. Split grouping is within each model; cross-model assignment equality is not required. Exact membership and order are in `split_assignments.jsonl`. p is the fraction of trajectories with at least two Prefix 1→0 transitions. The result report lists actual p/c and authorized target relaxations, including the retained TRAIN/VAL minimum transition proportions for sparse cells.

A complete native BF16 causal trajectory forward produces per-layer pools. Step uses local exponential pooling; Prefix uses cumulative reasoning-only exponential pooling. Pools are FP32 and detector features are row-L2-normalized. Genuine cross-boundary token overrides are retained.

Step fits a linear detector with BCE plus L2 coefficient 1e-4 using LBFGS (lr=1, max_iter=100, strong-Wolfe). Every committed parameter update is a candidate. Prefix begins with linear BCE+L2 LBFGS (no line search), then trains 20 AdamW epochs (lr=.001, weight_decay=.01, batch=64 trajectories) with final-position anchor weight 30 and teacher synchronization coefficient 120. Step teachers are frozen. See `config.json` and the attributed source for the full identity.

Select layers and weights on complete own-domain VAL AUROC, breaking ties by VAL AP. Exact unresolved ties do not prefer earlier updates or smaller layers. Select each level's threshold by VAL F1max, then precision, then largest threshold, using strict score > threshold. Answer shares the selected Prefix model and takes its last retained score; it has a separate Answer VAL threshold.

## Results

[RESULTS.md](result/RESULTS.md) contains all 45 model/domain/level results and actual TEST composition. [metrics.json](result/metrics.json) preserves unrounded metrics, selected layers, selected weights and thresholds. Primary metrics: AUROC, AP and empirical TPR at FPR ≤ 1%; secondary: ACC and F1. Units in JSON are fractions.

The original 30 Step/Prefix cells passed independent saved-score validation. The 15 Answer cells reuse those Prefix scores and passed independent final-position, label, threshold and metric checks. The repository's portable wrapper is validated separately; it does not claim that new training was performed during publication.

## Checkpoints

`checkpoint/<model>/<domain>/step.pt` and `prefix.pt` contain only the detector state_dict (`weight`, `bias`), with no optimizer, RNG, backbone or token features. `checkpoint/MANIFEST.json` records SHA-256 and source identities. Answer reuses `prefix.pt`.

## Run

Follow the [code quick start](../../code/README.md). Use `--mode evaluate` with the included weights, or `--mode train` for fresh selection. All external directories are set in your private copy of `configs/paths.example.json`. To reproduce one model/domain, prepare its rows and matching native feature cache; never include TEST rows in either training or selection.

## Interpretation

This split was retained after comparing five split designs. It targets trajectories with repeated correction transitions and is a deliberately shifted evaluation setting. It is not a random representative estimate of the original source populations. Per-domain exceptions and label rates are available in the results table.
