# Qwen ordered Step and Prefix detectors

This experiment fits two scalar heads with separate Step and Prefix readouts. The ordered three-state parameterization keeps the Step error probability at or below the Prefix error probability.

| Configuration | Head | Objective | Step input | Prefix input |
|---|---|---|---|---|
| linear / marginal_bce | linear | marginal_bce | step_exp | prefix_exp |
| linear / ce | linear | ce | step_exp | prefix_exp |
| mlp / marginal_bce | mlp | marginal_bce | step_exp | prefix_exp |
| mlp / ce | mlp | ce | step_exp | prefix_exp |

## Training parameters

| Parameter | Value |
|---|---|
| Backbone | Qwen3.8-27B, BF16 |
| Candidate layers | 1–64 |
| Detector dtype | FP32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Epoch limit | 100 |
| Early stopping patience | 10 |
| Seed | 42 |
| Feature normalization | No L2 normalization |

## Selection and evaluation

Each configuration uses the frozen 6:2:2 split. Layer and checkpoint selection ranks the mean VAL AUROC, then mean VAL AP, then lower mean VAL marginal BCE. Step and Prefix thresholds are selected independently on VAL by maximum F1. Each frozen detector was evaluated once on TEST. CPU acceptance used the saved predictions. The Logic features were derived from stored BF16 per-token activations. The feature readback and pooled-feature comparison were waived for this run.

Four Logic configurations have completed TEST and CPU acceptance. Each covers 15,233 positions. Other domain cells are pending or outside the configured scope.

### Step AUROC

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | **0.892730** | | | |
| linear / ce | | 0.877511 | | | |
| mlp / marginal_bce | | 0.871752 | | | |
| mlp / ce | | 0.865768 | | | |

### Step AP

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | **0.202211** | | | |
| linear / ce | | 0.162792 | | | |
| mlp / marginal_bce | | 0.173614 | | | |
| mlp / ce | | 0.160720 | | | |

### Step TPR_at_1pct_FPR

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | **0.200000** | | | |
| linear / ce | | 0.170968 | | | |
| mlp / marginal_bce | | 0.190323 | | | |
| mlp / ce | | 0.193548 | | | |

### Step F1

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | **0.280000** | | | |
| linear / ce | | 0.254499 | | | |
| mlp / marginal_bce | | 0.226054 | | | |
| mlp / ce | | 0.249240 | | | |

### Prefix AUROC

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | 0.836986 | | | |
| linear / ce | | 0.834137 | | | |
| mlp / marginal_bce | | 0.837086 | | | |
| mlp / ce | | **0.841833** | | | |

### Prefix AP

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | 0.395954 | | | |
| linear / ce | | 0.391522 | | | |
| mlp / marginal_bce | | 0.410970 | | | |
| mlp / ce | | **0.420628** | | | |

### Prefix TPR_at_1pct_FPR

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | 0.060206 | | | |
| linear / ce | | 0.063632 | | | |
| mlp / marginal_bce | | **0.093000** | | | |
| mlp / ce | | 0.080764 | | | |

### Prefix F1

| Configuration | Factuality | Logic | Math | Multi-hop | Science |
|---|---|---|---|---|---|
| linear / marginal_bce | | **0.452495** | | | |
| linear / ce | | 0.450307 | | | |
| mlp / marginal_bce | | 0.443308 | | | |
| mlp / ce | | 0.451300 | | | |

[Result index](result/index.json)
