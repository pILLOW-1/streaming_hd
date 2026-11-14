# Qwen C18 error-memory readouts

Test whether predicted errors in earlier reasoning steps improve the C18 conditional Prefix readout in Logic and Factuality.

| Arm | Conditional Prefix logit |
|---|---|
| A1 | Original C18, reused frozen results |
| A2 | Original logit plus a linear projection of error memory |
| A3 | Original logit plus a 64-dimensional bilinear interaction between error memory and the current step |

| Training parameter | Value |
|---|---|
| Features | Frozen Qwen BF16 token banks, original FP32 exponential readouts |
| Memory | Exclusive prefix sum of earlier step representations weighted by detached current Step probabilities |
| Initialization | A2 added weight zero. A3 W randomly initialized and V zero |
| Training | C18 three-class CE, Adam, seed 42, LR 0.001, weight decay 0.0001, batch 256 |
| Candidates | Layers 1–64, at most 100 epochs, patience 10 |
| Selection | Original joint mean Step/Prefix VAL AUROC, AP, then lower marginal BCE |

Train on TRAIN, select on VAL and evaluate each frozen arm once on TEST. Report overall Step and Prefix AUROC and inherited-versus-recovered within-trajectory AUROC on the matched recovery cohort. Compute memory from all preceding steps and restrict supervised positions to the frozen C18 labels.

Status: independently verified on VAL and TEST.

## Results

Frozen selections use the original C18 joint VAL rule.

| Configuration | Logic layer / epoch | Factuality layer / epoch |
|---|---:|---:|
| A1 | 42 / 12 | 50 / 5 |
| A2 | 42 / 12 | 47 / 5 |
| A3 | 42 / 5 | 34 / 6 |
| C17 | 35 / 6 | 50 / 5 |

### VAL Step AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.9158 | 0.8922 |
| A2 | 0.9158 | 0.8918 |
| A3 | 0.9120 | 0.8833 |
| C17 | 0.9151 | 0.8931 |

### VAL Prefix AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8946 | 0.9379 |
| A2 | 0.9062 | 0.9395 |
| A3 | 0.8825 | 0.9368 |
| C17 | 0.8952 | 0.9379 |

### VAL Conditional recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5637 | 0.6017 |
| A2 | 0.5652 | 0.5958 |
| A3 | 0.5589 | 0.6069 |
| C17 | 0.5747 | 0.5922 |

### VAL Prefix recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5448 | 0.5807 |
| A2 | 0.5546 | 0.5967 |
| A3 | 0.5546 | 0.5977 |
| C17 | 0.5550 | 0.5983 |

### TEST Step AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8775 | 0.8820 |
| A2 | 0.8775 | 0.8826 |
| A3 | 0.8816 | 0.8753 |
| C17 | 0.8927 | 0.8818 |

### TEST Prefix AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8341 | 0.8106 |
| A2 | 0.8377 | 0.8367 |
| A3 | 0.8129 | 0.8158 |
| C17 | 0.8370 | 0.8133 |

### TEST Conditional recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5470 | 0.5367 |
| A2 | 0.5583 | 0.5361 |
| A3 | 0.5363 | 0.5469 |
| C17 | 0.5417 | 0.5441 |

### TEST Prefix recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5492 | 0.5338 |
| A2 | 0.5636 | 0.5226 |
| A3 | 0.5360 | 0.5289 |
| C17 | 0.5498 | 0.5323 |
