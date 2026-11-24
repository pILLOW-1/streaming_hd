# Qwen A2 readout exploration on the random 6:2:2 split

This experiment uses the current unconstrained whole-question-family random 6:2:2 split (seed 42), separate from the historical TEST-p70 split.

Change the readout while retaining A2's prediction-weighted error history and joint three-class objective. The original A2 is the reference; these results alone do not establish a mechanism for recovery.

| Configuration | Design |
|---|---|
| [Original A2](../README.md) | Fixed exponential Step and Prefix pooling. |
| [Last token, Step + Prefix](configurations/last_token_both.md) | Both readouts use the current Step's last token. |
| [Hierarchical learnable aggregation](configurations/hierarchical_learnable.md) | Learn token-to-Step and Step-to-Prefix aggregation. |

[Training and selection](configurations/training.md) · [Result index](result/index.json)

AUROC (%). Qwen only. Empty cells have no independently accepted result. Original A2 values are reused from the parent report. Last-token Logic/Factuality each searched 64 layers and ran the frozen selected TEST once; saved predictions were independently checked on Cetus. Hierarchical learnable is unfinished; the RunPod jobs are paused awaiting a suitable 16 GB card. Other domains remain with their existing Cetus owner; no unfinished candidate is reported as a result.

## Step TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 | **78.64** | **82.28** | 88.90 | **93.94** | **90.24** |
| Last token, Step + Prefix | | | **89.16** | | 89.11 |
| Hierarchical learnable aggregation | | | | | |

## Prefix TEST AUROC

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 | **74.12** | **76.42** | **88.69** | **91.83** | **91.48** |
| Last token, Step + Prefix | | | 88.35 | | 90.43 |
| Hierarchical learnable aggregation | | | | | |
