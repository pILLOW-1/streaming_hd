# Qwen C18 with independent head selection

Test whether independent layer and epoch selection improves the ordered dual-readout detector on Logic and Factuality.

| Configuration | Architecture and loss | VAL selection |
|---|---|---|
| C17 | Original ordered linear heads, marginal BCE | Original joint selection |
| C18 | Original ordered linear heads, three-class CE | Original joint selection |
| C18 independent | Same heads and three-class CE | Step AUROC and conditional within-trajectory macro AUROC, independently |

| Parameter | Setting |
|---|---|
| Layer candidates | Native 1–64 |
| Training | Adam, LR 0.001, weight decay 0.0001, batch 256, seed 42 |
| Budget | Up to 100 epochs per layer, stop when both heads have gone 10 epochs without improvement |
| Features | Original FP32 step_exp and prefix_exp derived from frozen BF16 token banks |

The Step head ranks all VAL positions. The conditional head ranks inherited versus recovered positions with Y=0 and earlier observed C=1. Ties use AP for Step and pair-weighted then pooled AUROC for the conditional head, then earliest epoch and lowest layer. Combine the selected checkpoints by position ID with p = q + (1-q) kappa. Report aggregate Step and Prefix AUROC and conditional within-trajectory AUROC. Use original saved predictions for baseline comparison. Evaluate the frozen independent pair once on TEST after VAL selection.

Status: verified on VAL and TEST.

## Results

AUROC values are computed from saved predictions on identical position IDs.
Recovery uses Y=0 and an earlier observed C=1.

| Domain | Step layer / epoch | Conditional Prefix layer / epoch |
|---|---:|---:|
| Logic | 42 / 12 | 33 / 79 |
| Factuality | 50 / 5 | 44 / 28 |

### VAL Step

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.9151 | 0.8931 |
| C18 | 0.9158 | 0.8922 |
| C18 independent | 0.9158 | 0.8922 |

### VAL Prefix

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.8952 | 0.9379 |
| C18 | 0.8946 | 0.9379 |
| C18 independent | 0.8924 | 0.9297 |

### VAL Conditional recovery macro

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.5747 | 0.5922 |
| C18 | 0.5637 | 0.6017 |
| C18 independent | 0.6205 | 0.6452 |

### VAL Prefix recovery macro

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.5550 | 0.5983 |
| C18 | 0.5448 | 0.5807 |
| C18 independent | 0.5910 | 0.5782 |

### TEST Step

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.8927 | 0.8818 |
| C18 | 0.8775 | 0.8820 |
| C18 independent | 0.8775 | 0.8820 |

### TEST Prefix

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.8370 | 0.8133 |
| C18 | 0.8341 | 0.8106 |
| C18 independent | 0.8297 | 0.8006 |

### TEST Conditional recovery macro

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.5417 | 0.5441 |
| C18 | 0.5470 | 0.5367 |
| C18 independent | 0.5408 | 0.5251 |

### TEST Prefix recovery macro

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.5498 | 0.5323 |
| C18 | 0.5492 | 0.5338 |
| C18 independent | 0.5453 | 0.5147 |
