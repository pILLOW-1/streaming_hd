# Qwen A2 mechanism exploration on the random 6:2:2 split

Test changes to training or selection while retaining original A2 as the reference.

| Configuration | Design |
|---|---|
| [Original A2](../README.md) | Position-weighted training and ordinary joint VAL selection. |
| [Trajectory-equal VAL selection](configurations/trajectory_equal_val.md) | Fresh original A2 training; inverse trajectory-length VAL weights select checkpoint and layer. |

[Training and selection](configurations/training.md) · [Result index](result/index.json)

AUROC (%), Qwen only. Five-domain layout matches readout; unfinished cells are empty. Column maxima are bold, compared before rounding, with exact ties retained. Best available baseline follows the readout comparator set; Streaming is listed separately. [Exact references](result/comparison_references.json). Both new domain results passed independent Cetus CPU verification. These results isolate a selection change, not a causal mechanism of error recovery.

## Step TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| [Best available baseline](../../../reports/main_result/comparison_random_split_candidate.md) | **81.03** (SAPLMA) | **83.05** (SAPLMA) | 89.18 (SAPLMA) | 93.60 (HaMI) | 90.00 (HaMI) |
| [Original Streaming](../../streaming_random622_all_combinations_20261120/README.md) | 78.93 | 82.09 | 87.73 | 93.34 | 89.63 |
| Original A2 | 78.64 | 82.28 | 88.90 | **93.94** | 90.24 |
| Trajectory-equal VAL selection |  |  | **89.83** |  | **90.66** |

## Prefix TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| [Best available baseline](../../../reports/main_result/comparison_random_split_candidate.md) | **78.70** (SAPLMA) | 76.94 (Global Mean) | 86.80 (Global Mean) | 89.25 (Global Mean) | 89.59 (Global Mean) |
| [Original Streaming](../../streaming_random622_all_combinations_20261120/README.md) | 78.20 | **77.14** | 87.07 | 90.24 | 89.66 |
| Original A2 | 74.12 | 76.42 | **88.69** | **91.83** | 91.48 |
| Trajectory-equal VAL selection |  |  | 88.33 |  | **91.92** |
