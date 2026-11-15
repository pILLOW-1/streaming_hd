# Qwen C18 error-memory follow-up

Compare A2 with A1 across native layers, and test probability-weighted mean history with normalized A3 projections.

| Configuration | Readout and selection |
|---|---|
| A1 replay | Original C18 linear Step and conditional Prefix heads. Reconstruct each layer's VAL score on CPU. |
| A2 | Reuse the original 64 layer-specific VAL reports for the linear history model. |
| A3 normalized | Divide the detached-probability-weighted history by its weight sum, normalize both 64-dimensional projections with affine-free LayerNorm, then take their dot product. |
| C18 independent | Reuse the completed independent head-selection experiment. |

| Training parameter | Value |
|---|---|
| Features | Frozen Qwen3.8-27B BF16, the original Step and Prefix exponential readouts |
| Candidate layers | Native 1–64 |
| Training | C18 three-class CE, Adam, seed 42, learning rate 0.001, weight decay 0.0001, batch 256 |
| Budget and selection | Up to 100 epochs and patience 10. Joint VAL selection by mean Step/Prefix AUROC, AP, then lower BCE. |
| History normalization | LayerNorm epsilon 1e-5. An empty history contributes zero interaction. |

Both domains are independently verified. A1 is replayed on TRAIN/VAL to recover missing layer scores. A2 uses its original saved scores. A3 is evaluated once on TEST after VAL selection across all layers.

## A2 versus A1 on VAL Prefix AUROC

| Comparison | Logic | Factuality |
|---|---:|---:|
| A2 higher / tied / A1 higher | 56 / 0 / 8 | 62 / 0 / 2 |
| Median A2 minus A1 | +0.0042 | +0.0054 |

## Step AUROC

| Configuration and split | Logic | Factuality |
|---|---:|---:|
| A3 normalized, VAL | 0.9145 | 0.8842 |
| A3 normalized, TEST | 0.8775 | 0.8748 |

## Prefix AUROC

| Configuration and split | Logic | Factuality |
|---|---:|---:|
| A3 normalized, VAL | 0.8991 | 0.9445 |
| A3 normalized, TEST | 0.8320 | 0.8145 |

## Within-trajectory recovery AUROC

| Configuration and split | Logic conditional / Prefix | Factuality conditional / Prefix |
|---|---:|---:|
| A3 normalized, VAL | 0.5828 / 0.5745 | 0.5652 / 0.5792 |
| A3 normalized, TEST | 0.5572 / 0.5611 | 0.5202 / 0.5266 |

Logic normalized A3 selected layer 40, epoch 12. Factuality normalized A3 selected layer 51, epoch 6.

The independent C18 checkpoint scored 87.75% on Logic Step TEST, compared with HaMI at 88.75%. See [the independent selection results](../../qwen_c18_independent_selection_20261114/README.md) and [the baseline comparison](../../../reports/main_result/comparison.md).
