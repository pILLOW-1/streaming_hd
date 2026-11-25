# Qwen A2 loss exploration on the random 6:2:2 split

This is the current unconstrained whole-question-family random 6:2:2 split (seed 42), separate from the historical TEST-p70 split. The data had been inspected previously; this is exploratory evaluation.

| Configuration | Design |
|---|---|
| [Original A2](../README.md) | Original three-class cross-entropy. |
| [A2 + post-error ranking](configurations/post_error_ranking.md) | Add four TRAIN pairwise ranking terms; keep the readout and history unchanged. |

[Shared training and selection](configurations/training.md) · [Result index](result/index.json)

AUROC (%), Qwen only. Bold marks the highest available value in each column before rounding; exact ties are all bold. Best available baseline excludes Streaming and A2 and shows its method. [Reference values and hashes](result/comparison_references.json). Empty cells have no independently accepted result. Logic, Factuality and Multi-hop each completed 192 candidates and one selected TEST; all displayed results passed independent Cetus saved-prediction CPU verification. Other domains remain pending. Pairwise AUROCs and counts are retained in each aggregate result file; these comparisons alone do not establish a recovery mechanism.

## Step TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Best available baseline | **81.03** (SAPLMA) | **83.05** (SAPLMA) | **89.18** (SAPLMA) | 93.09 (SAPLMA) | 90.00 (HaMI) |
| Original Streaming | 78.93 | 82.09 | 87.73 | 93.34 | 89.63 |
| Original A2 | 78.64 | 82.28 | 88.90 | 93.94 | **90.24** |
| A2 + post-error ranking |  |  | 88.48 | **94.05** | 90.20 |

## Prefix TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Best available baseline | **78.70** (SAPLMA) | 76.94 (Global Mean) | 86.80 (Global Mean) | 89.25 (Global Mean) | 89.59 (Global Mean) |
| Original Streaming | 78.20 | **77.14** | 87.07 | 90.24 | 89.66 |
| Original A2 | 74.12 | 76.42 | **88.69** | 91.83 | 91.48 |
| A2 + post-error ranking |  |  | 87.27 | **92.48** | **91.49** |
