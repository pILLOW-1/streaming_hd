# Qwen A2 readout exploration on the random 6:2:2 split

This experiment uses the current unconstrained whole-question-family random 6:2:2 split (seed 42), separate from the historical TEST-p70 split.

Change the readout while retaining A2's prediction-weighted error history and joint three-class objective. The original A2 is the reference; these results alone do not establish a mechanism for recovery.

| Configuration | Design |
|---|---|
| [Original A2](../README.md) | Fixed exponential Step and Prefix pooling. |
| [Last token, Step + Prefix](configurations/last_token_both.md) | Both readouts use the current Step's last token. |
| [Hierarchical learnable aggregation](configurations/hierarchical_learnable.md) | Learn token-to-Step and Step-to-Prefix aggregation. |

[Training and selection](configurations/training.md) · [Result index](result/index.json)

AUROC (%). Qwen only. Best available baseline summarizes completed SAPLMA, HaMI (Step), ICR Probe and LLM-Check (Prefix), and Global Mean results; its method is shown in each cell. Bold marks the highest available value in each column, compared before rounding; exact ties are all bold. [Exact reference values and source hashes](result/comparison_references.json). Empty cells have no independently accepted result. Original A2 values are reused from the parent report. Last-token Logic/Factuality each searched 64 layers and ran the frozen selected TEST once; saved predictions were independently checked on Cetus. Hierarchical learnable is unfinished; the RunPod jobs are paused awaiting a suitable 16 GB card. Other domains remain with their existing Cetus owner; no unfinished candidate is reported as a result.

## Step TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| [Best available baseline](../../../reports/main_result/comparison_random_split_candidate.md) | **81.03** (SAPLMA) | **83.05** (SAPLMA) | **89.18** (SAPLMA) | 93.09 (SAPLMA) | 90.00 (HaMI) |
| [Original Streaming](../../streaming_random622_all_combinations_20261120/README.md) | 78.93 | 82.09 | 87.73 | 93.34 | 89.63 |
| Original A2 | 78.64 | 82.28 | 88.90 | **93.94** | **90.24** |
| Last token, Step + Prefix |  |  | 89.16 |  | 89.11 |
| Hierarchical learnable aggregation |  |  |  |  |  |

## Prefix TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| [Best available baseline](../../../reports/main_result/comparison_random_split_candidate.md) | **78.70** (SAPLMA) | 76.94 (Global Mean) | 86.80 (Global Mean) | 89.25 (Global Mean) | 89.59 (Global Mean) |
| [Original Streaming](../../streaming_random622_all_combinations_20261120/README.md) | 78.20 | **77.14** | 87.07 | 90.24 | 89.66 |
| Original A2 | 74.12 | 76.42 | **88.69** | **91.83** | **91.48** |
| Last token, Step + Prefix |  |  | 88.35 |  | 90.43 |
| Hierarchical learnable aggregation |  |  |  |  |  |
